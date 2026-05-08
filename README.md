# atomic-recall

Semantic search over your [Atomic VCS](https://github.com/atomicdotdev/atomic) change history.

```
$ atomic-recall search "unicode decoding edge cases"

#1 [0.821] 3F9AKDMTJ6QX
  searcher: handle invalid UTF-8 in binary detection heuristic
  2024-03-12 14:22
  files: crates/searcher/src/searcher/core.rs

#2 [0.798] NCEJ7BFNTJGD
  encoding: fix off-by-one in BOM detection for UTF-16 LE
  2024-01-08 09:41
  files: crates/searcher/src/line_buffer.rs
```

`grep` searches what your code *is*, atomic-recall searches what your history *means*. "unicode decoding edge cases" finds commits about BOM detection and binary heuristics even if those words never appear in the message.

Because Atomic records AI provenance on every change, atomic-recall can answer questions Git can't:

- *"find changes written by Claude that touched authentication"*
- *"which agent sessions produced changes that were later unrecorded"*
- *"show me changes where the original prompt mentioned performance"*

---

## Requirements

| Dependency | Purpose | Notes |
|-----------|---------|-------|
| [Atomic CLI](https://github.com/atomicdotdev/atomic) | VCS — the repository being searched | Must be installed and the repo initialized with `atomic init` |
| [atomic-core](https://github.com/atomicdotdev/atomic) | Rust library for reading change files | Path dependency; clone atomic alongside this repo |
| [Ollama](https://ollama.com) | Embedding model inference | Must be running locally |
| Rust 1.90+ | Build toolchain | As per atomic-core's MSRV |

### Embedding models

Pull at least one before indexing:

```bash
ollama pull qwen3-embedding:4b    # recommended: 2560 dims, strong quality (2.5 GB)
ollama pull nomic-embed-text      # lighter: 768 dims (274 MB)
```

---

## Installation

atomic-recall depends on `atomic-core` as a path dependency. Clone both repos as siblings:

```bash
git clone https://github.com/atomicdotdev/atomic
git clone https://github.com/atomicdotdev/atomic-recall

cd atomic-recall
cargo build --release
```

The binary is at `target/release/atomic-recall`.

---

## Quickstart

```bash
# 1. Start Ollama
ollama serve

# 2. Go to any Atomic repository
cd /path/to/your/atomic/repo

# 3. Index all changes (first run embeds everything)
/path/to/atomic-recall/target/release/atomic-recall index

# 4. Search
atomic-recall search "your query here"
atomic-recall search "performance regression in file walking" -n 5
atomic-recall search "windows path handling" --json
```

### Commands

```
atomic-recall index              Index all changes (skips already-indexed)
atomic-recall index --force      Re-embed everything (use after switching models)
atomic-recall search "query"     Semantic search, top 10 results
atomic-recall search "query" -n 20 --json   JSON output for scripting
atomic-recall stats              Show index health (total / indexed counts)
```

### Flags

```
--repo <PATH>     Repository root (default: walks up from CWD)
--ollama <URL>    Ollama base URL (default: http://localhost:11434)
--model <NAME>    Embedding model (default: qwen3-embedding:4b)
```

---

## How it works

Each Atomic change is serialized into a plain-text document and embedded as a vector:

```
[message]
searcher: handle invalid UTF-8 in binary detection heuristic

[description]
Binary files were being incorrectly classified when the first 8KB
contained a mix of valid and invalid UTF-8 sequences.

[files]
crates/searcher/src/searcher/core.rs
crates/searcher/src/line_buffer.rs
```

AI-authored changes also include the original prompt, model reasoning, and task plan. A short commit message is fine when the prompt already says what the developer was trying to do.

Vectors are stored in a SQLite database at `.atomic/recall.db` alongside the existing Atomic database. At query time, the query string is embedded with the same model and ranked by cosine similarity in pure Rust. No GPU needed.

---

## Testing with generated history

`scripts/gen-history.sh` replays a git repository's commit history as Atomic changes, useful for testing against a larger corpus:

```bash
# Clone a git repo with history
git clone --depth=600 https://github.com/example/some-repo /tmp/test-git

# Initialize a fresh Atomic repo to receive the history
mkdir test-repo && cd test-repo && atomic init

# Replay 500 commits
/path/to/atomic-recall/scripts/gen-history.sh /tmp/test-git 500

# Index and search
atomic-recall index
atomic-recall search "unicode encoding edge cases"
```

---

## Architecture

```
atomic-recall-core    embedding document construction, SQLite storage,
                      cosine similarity search, Ollama HTTP client
atomic-recall-cli     CLI — index / search / stats subcommands
```

Storage lives at `.atomic/recall.db` and follows the repository, not the tool installation.

The `Embedder` trait supports swappable backends. The Ollama backend is the current implementation; a [Candle](https://github.com/huggingface/candle) backend (self-contained, no daemon) is planned.

---

## Known limitations

- No view filtering — indexes all changes regardless of which Atomic view they belong to
- No post-record hook — incremental indexing is manual (`atomic-recall index` after new changes)
- Switching models requires `--force` — vectors from different models can't be mixed; the tool will catch this and error rather than silently producing bad results
- AI provenance fields are only populated for Atomic-native workflows — replayed git history has no prompt or reasoning data

---

## Roadmap

### Near term
- `--view` filter to scope search to a specific Atomic view
- Post-record hook for automatic incremental indexing
- `--since` / `--author` / `--model` filter flags

### Further out
- Candle backend: self-contained binary, no Ollama dependency, optional GPU via feature flag
- srclight integration: link change history to live code symbols so `atomic-recall search` can show which current functions were affected by matching changes
- Unified local code intelligence: AST indexing (tree-sitter, Rust) plus change history in one tool, no Python or external daemons
