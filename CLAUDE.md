# atomic-recall

Semantic search over the Atomic VCS change graph. Think of it as srclight, but
for your history instead of your source code.

## What it does

`atomic recall "fix cyclic dependency in vertex insertion"` searches your
Atomic change log using natural language and returns the most relevant changes —
ranked by meaning, not keyword overlap.

It works by embedding each change's rich metadata (message, description, prompt,
affected files, provenance) into vectors stored in a SQLite database alongside
the `.atomic` directory. At query time it embeds the query and ranks stored
vectors by cosine similarity.

## Why this is interesting

Atomic changes carry data that Git commits never have:

- **`prompt`** — the user's actual intent in natural language ("the insertion
  is failing when a change references itself")
- **`reasoning_text`** — the model's reasoning before producing the change
- **`task_plan`** — what the agent planned to do
- **`model` / `vendor`** — which AI produced this change
- **`tokens` / `cost`** — resource usage per change
- **`session_id`** — group all changes from one agent session

Embedding all of this means you can ask questions Git can never answer:
- "find all changes written by Claude that touched authentication"
- "which agent sessions produced changes that were later unrecorded"
- "show me changes where the prompt mentioned performance"

## Architecture

### Crate structure

```
atomic-recall/
├── CLAUDE.md
├── Cargo.toml              (workspace)
├── crates/
│   ├── atomic-recall-core/ (embedding, indexing, search logic)
│   ├── atomic-recall-cli/  (CLI binary — `atomic-recall` command)
│   └── atomic-recall-db/   (SQLite schema, migrations, vector storage)
```

### Data flow

```
.atomic/changes/           (Atomic change files on disk)
        │
        │  atomic-recall index
        ▼
  Read change via atomic-core (ChangeHeader, Provenance, affected files)
        │
        │  Build embedding document (see below)
        ▼
  Ollama embeddings API    (POST /api/embeddings)
        │
        │  Vec<f32> embedding vector
        ▼
  SQLite (atomic-recall.db) alongside .atomic/
        │
        │  atomic-recall search "query"
        ▼
  Embed query → cosine similarity in Rust → ranked results
        │
        ▼
  CLI output (change hash, message, score, provenance summary)
```

### The embedding document

Each change is serialized into a plain-text document before embedding.
Fields are ordered by signal strength so the model attends to the most
important content first:

```
[message]
Fix cyclic dependency check in vertex insertion

[description]
Down context must not reference the same change being applied or it
creates a cycle in the dependency graph.

[prompt]
the insertion is failing when a change references itself

[reasoning]
The error occurs because we allow down_pos to reference change_id...

[files]
atomic-core/src/apply/insertion.rs
atomic-core/src/apply/mod.rs

[author]
claude-sonnet-4-6 via Claude Code (Anthropic)
```

Fields omitted when absent. Human-authored changes won't have prompt/reasoning.

### Storage schema

```sql
CREATE TABLE changes (
    hash        TEXT PRIMARY KEY,   -- Atomic change hash
    message     TEXT NOT NULL,
    description TEXT,
    timestamp   INTEGER,            -- Unix epoch
    authors     TEXT,               -- JSON array
    files       TEXT,               -- JSON array of affected paths
    -- Provenance (nullable — human changes won't have these)
    ai_vendor   TEXT,
    ai_model    TEXT,
    ai_tool     TEXT,
    prompt_text TEXT,
    reasoning   TEXT,
    task_plan   TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    cost_usd    REAL,
    session_id  TEXT,
    -- Embedding
    embedding   BLOB NOT NULL,      -- f32 LE bytes, length = model dims
    embed_model TEXT NOT NULL,      -- e.g. "nomic-embed-text"
    indexed_at  INTEGER NOT NULL
);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
```

### Similarity search

Pure Rust, no CUDA required. At query scale (thousands of changes, not
millions) SIMD-optimised f32 dot products in Rust are fast enough without
a GPU math library.

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
```

For larger repos (100k+ changes) we can add an ANN index (hnsw_rs or
usearch) without touching the embedding or storage layers.

## MVP scope

**Phase 1 — Index**
- Read all changes from `.atomic/changes/` using `atomic-core` as a library
- Extract: hash, header (message, description, authors, timestamp), affected
  file paths, provenance fields
- Build embedding document, send to Ollama, store vector in SQLite

**Phase 2 — Search**
- Embed the query string via Ollama
- Load all stored vectors, compute cosine similarity, rank and return top-N
- CLI output: hash, score, message, timestamp, author/model

**Phase 3 — Incremental indexing**
- Track which changes are already indexed (by hash)
- Only embed new/unindexed changes on each run
- Optionally install as an Atomic post-record hook

**Phase 4 — Integration**
- `atomic recall` as a subcommand alias (if Atomic supports plugins)
- Pipe-friendly output (JSON flag)
- Filter flags: `--author`, `--model`, `--since`, `--view`

## Future: Candle backend

Once the Ollama MVP is proven, swap the embedding backend for
`candle-transformers` running a quantised model locally:

- No daemon dependency — pure Rust binary, self-contained
- Candidate models: `nomic-embed-text`, `bge-small-en-v1.5` (GGUF/Q4)
- GPU via Candle's CUDA feature flag — opt-in, not required
- Same `Embedder` trait, different impl — storage layer unchanged

The trait interface to design against from day one:

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn model_id(&self) -> &str;
    fn dims(&self) -> usize;
}
```

## Key dependencies (MVP)

```toml
atomic-core = { path = "../atomic/atomic-core" }
rusqlite    = { version = "0.31", features = ["bundled"] }
reqwest     = { version = "0.12", features = ["json"] }
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
tokio       = { version = "1", features = ["full"] }
clap        = { version = "4", features = ["derive"] }
anyhow      = "1"
```

## CLI design

```
atomic-recall index              Index all changes in current repo
atomic-recall index --watch      Re-index on each new record (poll mode)
atomic-recall search "query"     Semantic search, top 10 results
atomic-recall search "query" -n 20 --json
atomic-recall show <hash>        Show full indexed data for a change
atomic-recall stats              Index health: change count, model, db size
```

## Why Ollama for MVP, and why no CUDA needed

Ollama handles model inference internally and manages its own GPU access.
We call it over HTTP (`localhost:11434`) — no CUDA libraries in our process.

The only GPU math srclight needed was for its own cosine similarity ranking
(CuPy/CUDA BLAS). In Rust, plain f32 arithmetic with compiler
auto-vectorisation (AVX2/NEON) is fast enough for the scale of an Atomic
log. CUDA only enters the picture when we add the Candle backend with GPU
inference — and even then it's an optional feature flag, not a requirement.

## Recommended embedding model

`nomic-embed-text` via Ollama. 768 dimensions, strong at mixed code+prose
content, fast, widely supported.

```bash
ollama pull nomic-embed-text
```

## Development conventions

- Track with Atomic (`atomic record`, not `git commit`)
- Git present for remote compatibility only
- Rust edition 2021
- `cargo fmt` + `cargo clippy -- -D warnings` before each record
