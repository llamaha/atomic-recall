use anyhow::{Context, Result};
use atomic_recall_core::{
    change::read_changes,
    db::RecallDb,
    embed::{Embedder, OllamaEmbedder},
    search::rank,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "atomic-recall",
    about = "Semantic search over your Atomic VCS change history",
    version
)]
struct Cli {
    /// Path to the repository root (walks up from CWD if omitted)
    #[arg(long, short, global = true)]
    repo: Option<PathBuf>,

    /// Ollama base URL
    #[arg(long, global = true, default_value = "http://localhost:11434")]
    ollama: String,

    /// Embedding model
    #[arg(long, global = true, default_value = "qwen3-embedding:4b")]
    model: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Index all changes in the repository
    Index {
        /// Re-embed already-indexed changes (force refresh)
        #[arg(long)]
        force: bool,

        /// Only index changes belonging to this view (not yet implemented)
        #[arg(long)]
        view: Option<String>,
    },
    /// Search changes by natural language query
    Search {
        /// The search query
        query: String,

        /// Number of results to return
        #[arg(long, short, default_value = "10")]
        n: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Filter results to a specific view (not yet implemented)
        #[arg(long)]
        view: Option<String>,
    },
    /// Show index statistics
    Stats,
}

fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    s.char_indices()
        .nth(max_chars)
        .map(|(idx, _)| s[..idx].to_string())
        .unwrap_or_else(|| s.to_string())
}

/// Walk up from `start` until a directory containing `.atomic/` is found.
fn find_repo_root(start: &std::path::Path) -> Result<PathBuf> {
    let mut dir = start;
    loop {
        if dir.join(".atomic").is_dir() {
            return Ok(dir.to_path_buf());
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => anyhow::bail!(
                "No .atomic directory found in '{}' or any parent.\n\
                 Run `atomic init` to initialise a repository, or pass --repo <path>.",
                start.display()
            ),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let repo = match &cli.repo {
        Some(p) => p.clone(),
        None => {
            let cwd = std::env::current_dir().context("Cannot determine current directory")?;
            find_repo_root(&cwd)?
        }
    };

    // dims=0 placeholder — probe_dims() is called when we actually need the value
    let embedder = OllamaEmbedder::new(&cli.ollama, &cli.model, 0);
    let db = RecallDb::open(&repo)?;

    match cli.command {
        Command::Index { force, view } => {
            if view.is_some() {
                eprintln!("Warning: --view filtering is not yet implemented; indexing all changes.");
            }
            cmd_index(&repo, &db, &embedder, force).await?;
        }
        Command::Search { query, n, json, view } => {
            if view.is_some() {
                eprintln!("Warning: --view filtering is not yet implemented; searching all changes.");
            }
            cmd_search(&db, &embedder, &query, n, json).await?;
        }
        Command::Stats => {
            cmd_stats(&db)?;
        }
    }

    Ok(())
}

async fn cmd_index(
    repo: &PathBuf,
    db: &RecallDb,
    embedder: &OllamaEmbedder,
    force: bool,
) -> Result<()> {
    let changes = read_changes(repo)?;
    let total = changes.len();

    println!("Found {} changes", total);

    // Probe dims and check model compatibility before doing any work
    print!("Probing embedding dimensions... ");
    use std::io::Write;
    std::io::stdout().flush().ok();
    let dims = embedder.probe_dims().await?;
    println!("{}d", dims);

    db.check_model_compat(embedder.model_id(), dims, force)?;

    let pb = indicatif::ProgressBar::new(total as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  [{bar:40.cyan/blue}] {pos}/{len}  skipped: {msg}  eta: {eta}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message("0");

    let mut indexed = 0u64;
    let mut skipped = 0u64;
    let mut errors = 0u64;

    for record in &changes {
        if !force && db.is_indexed(&record.hash)? {
            skipped += 1;
            pb.set_message(skipped.to_string());
            pb.inc(1);
            continue;
        }

        db.upsert_change(record)?;

        let doc = truncate_to_chars(&record.embedding_document(), 4000);
        let embedding = match embedder.embed(&doc).await {
            Ok(e) => e,
            Err(e) => {
                pb.suspend(|| {
                    eprintln!("Warning: skipping {} ({}): {}", &record.hash[..8], record.message, e);
                });
                errors += 1;
                pb.inc(1);
                continue;
            }
        };

        db.store_embedding(&record.hash, &embedding, embedder.model_id())?;
        indexed += 1;
        pb.inc(1);
    }

    pb.finish_and_clear();

    println!(
        "Done. {} indexed, {} already up to date{}.",
        indexed,
        skipped,
        if errors > 0 { format!(", {} errors", errors) } else { String::new() }
    );
    Ok(())
}

async fn cmd_search(
    db: &RecallDb,
    embedder: &OllamaEmbedder,
    query: &str,
    top_n: usize,
    as_json: bool,
) -> Result<()> {
    let query_embedding = embedder.embed(query).await?;
    let candidates = db.load_indexed()?;

    if candidates.is_empty() {
        eprintln!("No indexed changes found. Run `atomic-recall index` first.");
        return Ok(());
    }

    let results = rank(&candidates, &query_embedding, top_n);

    if as_json {
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "hash": r.hash,
                    "score": r.score,
                    "message": r.message,
                    "timestamp": r.timestamp,
                    "authors": r.authors,
                    "files": serde_json::from_str::<serde_json::Value>(&r.files).unwrap_or_default(),
                    "ai_model": r.ai_model,
                    "ai_vendor": r.ai_vendor,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json)?);
        return Ok(());
    }

    if results.is_empty() {
        println!("No results.");
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        let date = chrono::DateTime::from_timestamp(result.timestamp, 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let author_info = match (&result.ai_model, &result.ai_vendor) {
            (Some(model), Some(vendor)) => format!("{} ({})", model, vendor),
            _ => {
                let authors: Vec<String> =
                    serde_json::from_str(&result.authors).unwrap_or_default();
                authors.join(", ")
            }
        };

        let files: Vec<String> =
            serde_json::from_str(&result.files).unwrap_or_default();

        println!(
            "\n#{} [{:.3}] {}",
            i + 1,
            result.score,
            &result.hash[..12]
        );
        println!("  {}", result.message);
        println!("  {} · {}", date, author_info);
        if !files.is_empty() {
            let preview = if files.len() > 3 {
                format!("{} and {} more", files[..3].join(", "), files.len() - 3)
            } else {
                files.join(", ")
            };
            println!("  files: {}", preview);
        }
    }
    println!();

    Ok(())
}

fn cmd_stats(db: &RecallDb) -> Result<()> {
    let (total, indexed) = db.count()?;
    println!("Changes: {} total, {} indexed", total, indexed);
    Ok(())
}
