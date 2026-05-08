use crate::db::IndexedChange;

/// A ranked search result.
#[derive(Debug)]
pub struct SearchResult {
    pub hash: String,
    pub message: String,
    pub description: Option<String>,
    pub timestamp: i64,
    pub authors: String,
    pub files: String,
    pub ai_vendor: Option<String>,
    pub ai_model: Option<String>,
    pub score: f32,
}

/// Rank all indexed changes against a query embedding.
/// Returns results sorted by descending cosine similarity, limited to top_n.
pub fn rank(candidates: &[IndexedChange], query: &[f32], top_n: usize) -> Vec<SearchResult> {
    let mut scored: Vec<(f32, &IndexedChange)> = candidates
        .iter()
        .map(|c| (cosine_similarity(query, &c.embedding), c))
        .collect();

    // Sort descending by score
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(top_n)
        .map(|(score, c)| SearchResult {
            hash: c.hash.clone(),
            message: c.message.clone(),
            description: c.description.clone(),
            timestamp: c.timestamp,
            authors: c.authors.clone(),
            files: c.files.clone(),
            ai_vendor: c.ai_vendor.clone(),
            ai_model: c.ai_model.clone(),
            score,
        })
        .collect()
}

/// Cosine similarity between two vectors.
/// Returns 0.0 for zero-length vectors rather than NaN.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
