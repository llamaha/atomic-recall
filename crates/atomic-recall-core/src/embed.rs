use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// The embedding backend trait. Swap Ollama for Candle later without
/// touching the indexer or search layers.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn model_id(&self) -> &str;
    fn dims(&self) -> usize;
}

/// Ollama embedding client.
pub struct OllamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dims: usize,
}

impl OllamaEmbedder {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, dims: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            dims,
        }
    }

    /// nomic-embed-text at localhost with default 768 dims.
    pub fn default_nomic() -> Self {
        Self::new("http://localhost:11434", "nomic-embed-text", 768)
    }

    /// Probe the real embedding dimension by running a short test embed.
    /// Use this instead of hardcoding dims per model name.
    pub async fn probe_dims(&self) -> Result<usize> {
        let v = self.embed("test").await?;
        Ok(v.len())
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

#[async_trait::async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let resp: EmbedResponse = self
            .client
            .post(&url)
            .json(&EmbedRequest {
                model: &self.model,
                prompt: text,
            })
            .send()
            .await
            .context("Ollama request failed — is Ollama running?")?
            .error_for_status()
            .context("Ollama returned an error status")?
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        Ok(resp.embedding)
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn dims(&self) -> usize {
        self.dims
    }
}
