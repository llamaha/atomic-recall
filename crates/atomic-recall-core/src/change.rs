use anyhow::{Context, Result};
use atomic_core::{Base32, Change, PromptContent};
use std::fs;
use std::path::Path;

/// A change record with all embeddable fields extracted.
#[derive(Debug, Clone)]
pub struct ChangeRecord {
    pub hash: String,
    pub message: String,
    pub description: Option<String>,
    pub timestamp: i64,
    pub authors: Vec<String>,
    pub files: Vec<String>,
    // Provenance — None for human-authored changes
    pub ai_vendor: Option<String>,
    pub ai_model: Option<String>,
    pub ai_tool: Option<String>,
    pub prompt_text: Option<String>,
    pub reasoning: Option<String>,
    pub task_plan: Option<String>,
    pub tokens_in: Option<i64>,
    pub tokens_out: Option<i64>,
    pub cost_usd: Option<f64>,
    pub session_id: Option<String>,
}

impl ChangeRecord {
    /// Build the text document that will be embedded.
    /// Fields are ordered by semantic signal strength.
    pub fn embedding_document(&self) -> String {
        let mut doc = String::new();

        doc.push_str(&format!("[message]\n{}\n", self.message));

        if let Some(desc) = &self.description {
            doc.push_str(&format!("\n[description]\n{}\n", desc));
        }
        if let Some(prompt) = &self.prompt_text {
            doc.push_str(&format!("\n[prompt]\n{}\n", prompt));
        }
        if let Some(reasoning) = &self.reasoning {
            // Reasoning can be very long — truncate to keep embedding focused
            let truncated = if reasoning.len() > 2000 {
                &reasoning[..2000]
            } else {
                reasoning.as_str()
            };
            doc.push_str(&format!("\n[reasoning]\n{}\n", truncated));
        }
        if let Some(plan) = &self.task_plan {
            doc.push_str(&format!("\n[task_plan]\n{}\n", plan));
        }
        if !self.files.is_empty() {
            // Cap at 20 files to keep the document concise
            let shown = &self.files[..self.files.len().min(20)];
            let extra = self.files.len().saturating_sub(20);
            let mut files_str = shown.join("\n");
            if extra > 0 {
                files_str.push_str(&format!("\n... and {} more files", extra));
            }
            doc.push_str(&format!("\n[files]\n{}\n", files_str));
        }
        if let (Some(model), Some(vendor)) = (&self.ai_model, &self.ai_vendor) {
            let tool = self.ai_tool.as_deref().unwrap_or("unknown");
            doc.push_str(&format!("\n[author]\n{} via {} ({})\n", model, tool, vendor));
        } else if !self.authors.is_empty() {
            doc.push_str(&format!("\n[author]\n{}\n", self.authors.join(", ")));
        }

        doc
    }
}

/// Read all changes from a repository's .atomic/changes directory.
/// Returns records sorted by timestamp ascending.
pub fn read_changes(repo_root: &Path) -> Result<Vec<ChangeRecord>> {
    let changes_dir = repo_root.join(".atomic").join("changes");

    if !changes_dir.exists() {
        anyhow::bail!(
            "No .atomic/changes directory at {:?} — is this an Atomic repository?",
            repo_root
        );
    }

    let mut records = Vec::new();

    // Changes are stored as .atomic/changes/{2-char-prefix}/{fullhash}.change
    for entry in walkdir::WalkDir::new(&changes_dir)
        .min_depth(2)
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("change") {
            continue;
        }

        match load_change_file(path) {
            Ok(record) => records.push(record),
            Err(e) => eprintln!("Warning: skipping {:?}: {}", path, e),
        }
    }

    records.sort_by_key(|r| r.timestamp);
    Ok(records)
}

fn load_change_file(path: &Path) -> Result<ChangeRecord> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("Cannot open {:?}", path))?;

    let (change, hash) = Change::deserialize(&mut file)
        .with_context(|| format!("Failed to deserialize {:?}", path))?;

    let header = &change.hashed.header;
    let hash_str = hash.to_base32();
    let timestamp = header.timestamp.timestamp();

    let authors: Vec<String> = header
        .authors
        .iter()
        .map(|a| a.display_short())
        .collect();

    let files: Vec<String> = change
        .file_ops()
        .iter()
        .map(|fo| fo.path().to_string())
        .filter(|p| !p.starts_with(".gen-history-scratch") && !p.starts_with(".atomicignore"))
        .collect();

    // Use first provenance entry — most changes have exactly one
    let prov = change.provenance().first();

    let (ai_vendor, ai_model, ai_tool, prompt_text, reasoning, task_plan,
         tokens_in, tokens_out, cost_usd, session_id) = match prov {
        Some(p) => (
            Some(p.vendor.to_string()),
            Some(p.model.clone()),
            Some(p.tool.to_string()),
            prompt_text(&p.prompt),
            p.reasoning_text.clone(),
            p.task_plan.clone(),
            Some(p.tokens.input_tokens as i64),
            Some(p.tokens.output_tokens as i64),
            Some(p.cost.usd),
            p.session_id.clone(),
        ),
        None => (None, None, None, None, None, None, None, None, None, None),
    };

    Ok(ChangeRecord {
        hash: hash_str,
        message: header.message.clone(),
        description: header.description.clone(),
        timestamp,
        authors,
        files,
        ai_vendor,
        ai_model,
        ai_tool,
        prompt_text,
        reasoning,
        task_plan,
        tokens_in,
        tokens_out,
        cost_usd,
        session_id,
    })
}

fn prompt_text(prompt: &PromptContent) -> Option<String> {
    prompt.text().filter(|t| !t.is_empty()).map(|t| t.to_string())
}
