use serde::{Deserialize, Serialize};

// Request/Response models
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: InputText,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    #[serde(default = "default_task")]
    pub task: String,
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputText {
    Single(String),
    Multiple(Vec<String>),
}

impl InputText {
    pub fn to_vec(self) -> Vec<String> {
        match self {
            InputText::Single(s) => vec![s],
            InputText::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// Reranking models
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<DocumentInput>,
    #[serde(default = "default_rerank_model")]
    pub model: String,
    pub top_n: Option<usize>,
    #[serde(default = "default_return_documents")]
    pub return_documents: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DocumentInput {
    Text(String),
    Object(serde_json::Value),
}

impl DocumentInput {
    pub fn as_text(&self) -> String {
        match self {
            DocumentInput::Text(s) => s.clone(),
            DocumentInput::Object(v) => v.to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub object: String,
    pub data: Vec<RerankResult>,
    pub model: String,
    pub usage: RerankUsage,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RerankUsage {
    pub total_tokens: usize,
}

// Domain models
#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    pub vector: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct RerankModel {
    pub index: usize,
    pub relevance_score: f32,
    pub document: Option<String>,
}

// Default functions
fn default_model() -> String {
    "jina-embeddings-v3".to_string()
}

fn default_rerank_model() -> String {
    "jina-reranker-v2".to_string()
}

fn default_encoding_format() -> String {
    "float".to_string()
}

fn default_task() -> String {
    "retrieval.query".to_string()
}

fn default_return_documents() -> bool {
    true
}

// Task mapping constants
pub const TASK_MAPPING: &[(&str, i64)] = &[
    ("retrieval.query", 0),
    ("retrieval.passage", 1),
    ("separation", 2),
    ("classification", 3),
    ("text-matching", 4),
];

pub fn get_task_id(task: &str) -> i64 {
    TASK_MAPPING
        .iter()
        .find(|(t, _)| *t == task)
        .map(|(_, id)| *id)
        .unwrap_or(0)
}
