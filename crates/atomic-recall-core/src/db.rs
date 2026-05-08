use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

use crate::change::ChangeRecord;

pub struct RecallDb {
    conn: Connection,
}

impl RecallDb {
    /// Open (or create) the index database at `.atomic/recall.db`.
    pub fn open(repo_root: &Path) -> Result<Self> {
        let db_path = repo_root.join(".atomic").join("recall.db");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("Cannot open database at {:?}", db_path))?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS changes (
                hash         TEXT PRIMARY KEY,
                message      TEXT NOT NULL,
                description  TEXT,
                timestamp    INTEGER NOT NULL,
                authors      TEXT NOT NULL DEFAULT '[]',
                files        TEXT NOT NULL DEFAULT '[]',
                ai_vendor    TEXT,
                ai_model     TEXT,
                ai_tool      TEXT,
                prompt_text  TEXT,
                reasoning    TEXT,
                task_plan    TEXT,
                tokens_in    INTEGER,
                tokens_out   INTEGER,
                cost_usd     REAL,
                session_id   TEXT,
                embedding    BLOB,
                embed_model  TEXT,
                indexed_at   INTEGER
            );

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            ",
        )
        .context("Failed to initialise database schema")
    }

    /// Check whether a change hash is already indexed (has an embedding).
    pub fn is_indexed(&self, hash: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM changes WHERE hash = ?1 AND embedding IS NOT NULL",
            params![hash],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Insert or update a change record without an embedding yet.
    pub fn upsert_change(&self, record: &ChangeRecord) -> Result<()> {
        let authors = serde_json::to_string(&record.authors)?;
        let files = serde_json::to_string(&record.files)?;

        self.conn.execute(
            "INSERT INTO changes
                (hash, message, description, timestamp, authors, files,
                 ai_vendor, ai_model, ai_tool, prompt_text, reasoning, task_plan,
                 tokens_in, tokens_out, cost_usd, session_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)
             ON CONFLICT(hash) DO UPDATE SET
                message     = excluded.message,
                description = excluded.description,
                authors     = excluded.authors,
                files       = excluded.files,
                ai_vendor   = excluded.ai_vendor,
                ai_model    = excluded.ai_model,
                ai_tool     = excluded.ai_tool,
                prompt_text = excluded.prompt_text,
                reasoning   = excluded.reasoning,
                task_plan   = excluded.task_plan,
                tokens_in   = excluded.tokens_in,
                tokens_out  = excluded.tokens_out,
                cost_usd    = excluded.cost_usd,
                session_id  = excluded.session_id",
            params![
                record.hash,
                record.message,
                record.description,
                record.timestamp,
                authors,
                files,
                record.ai_vendor,
                record.ai_model,
                record.ai_tool,
                record.prompt_text,
                record.reasoning,
                record.task_plan,
                record.tokens_in,
                record.tokens_out,
                record.cost_usd,
                record.session_id,
            ],
        )?;
        Ok(())
    }

    /// Store the embedding vector for an already-upserted change.
    pub fn store_embedding(&self, hash: &str, embedding: &[f32], model: &str) -> Result<()> {
        let blob = f32_slice_to_bytes(embedding);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "UPDATE changes SET embedding = ?1, embed_model = ?2, indexed_at = ?3
             WHERE hash = ?4",
            params![blob, model, now, hash],
        )?;
        Ok(())
    }

    /// Load all indexed changes with their embeddings for similarity search.
    pub fn load_indexed(&self) -> Result<Vec<IndexedChange>> {
        let mut stmt = self.conn.prepare(
            "SELECT hash, message, description, timestamp, authors, files,
                    ai_vendor, ai_model, embedding
             FROM changes
             WHERE embedding IS NOT NULL
             ORDER BY timestamp ASC",
        )?;

        let rows = stmt.query_map([], |row| {
            let blob: Vec<u8> = row.get(8)?;
            Ok(IndexedChange {
                hash: row.get(0)?,
                message: row.get(1)?,
                description: row.get(2)?,
                timestamp: row.get(3)?,
                authors: row.get(4)?,
                files: row.get(5)?,
                ai_vendor: row.get(6)?,
                ai_model: row.get(7)?,
                embedding: bytes_to_f32_vec(&blob),
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .context("Failed to load indexed changes")
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        match self.conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Verify that the current model/dims match what the index was built with.
    /// On mismatch, returns an error unless `force` is set — in which case
    /// existing embeddings are wiped so everything gets re-indexed cleanly.
    pub fn check_model_compat(&self, model: &str, dims: usize, force: bool) -> Result<()> {
        // Prefer meta table; fall back to reading embed_model from existing rows
        // (handles databases created before meta-tracking was added).
        let stored_model = self.get_meta("embed_model")?.or_else(|| {
            self.conn
                .query_row(
                    "SELECT embed_model FROM changes WHERE embed_model IS NOT NULL LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .ok()
        });
        let stored_dims = self
            .get_meta("embed_dims")?
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| {
                // Infer dims from the blob length of an existing embedding
                let blob: Option<Vec<u8>> = self
                    .conn
                    .query_row(
                        "SELECT embedding FROM changes WHERE embedding IS NOT NULL LIMIT 1",
                        [],
                        |row| row.get(0),
                    )
                    .ok();
                blob.map(|b| b.len() / 4)
            });

        match (stored_model, stored_dims) {
            (Some(m), Some(d)) if m != model || d != dims => {
                if !force {
                    anyhow::bail!(
                        "Index was built with model '{}' ({}d) but you're using '{}' ({}d).\n\
                         Run `atomic-recall index --force` to re-embed with the new model.",
                        m, d, model, dims
                    );
                }
                // --force: clear existing embeddings so they all get re-indexed
                self.conn.execute(
                    "UPDATE changes SET embedding = NULL, embed_model = NULL, indexed_at = NULL",
                    [],
                )?;
                eprintln!(
                    "Model changed ({} → {}). Cleared existing embeddings; re-indexing everything.",
                    m, model
                );
            }
            _ => {}
        }

        self.set_meta("embed_model", model)?;
        self.set_meta("embed_dims", &dims.to_string())?;
        Ok(())
    }

    pub fn count(&self) -> Result<(i64, i64)> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM changes",
            [],
            |r| r.get(0),
        )?;
        let indexed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM changes WHERE embedding IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        Ok((total, indexed))
    }
}

pub struct IndexedChange {
    pub hash: String,
    pub message: String,
    pub description: Option<String>,
    pub timestamp: i64,
    pub authors: String,   // JSON array
    pub files: String,     // JSON array
    pub ai_vendor: Option<String>,
    pub ai_model: Option<String>,
    pub embedding: Vec<f32>,
}

fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
