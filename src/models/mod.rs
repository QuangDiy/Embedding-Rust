use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// Request/Response models
/// Embedding request with support for different task types
#[derive(Debug, Deserialize, ToSchema)]
#[schema(example = json!({
    "input": ["Xin chào, bạn khỏe không?", "Machine learning là gì?"],
    "model": "jina-embeddings-v3",
    "task": "retrieval.query",
    "encoding_format": "float"
}))]
pub struct EmbeddingRequest {
    /// Input text(s) to generate embeddings for
    #[schema(example = json!(["Xin chào, bạn khỏe không?", "Machine learning là gì?"]))]
    pub input: InputText,
    #[serde(default = "default_model")]
    #[schema(default = "jina-embeddings-v3")]
    pub model: String,
    #[serde(default = "default_encoding_format")]
    #[schema(default = "float")]
    pub encoding_format: String,
    /// Task type for LoRA adapter selection. Valid values:
    /// - "retrieval.query" (default): For search queries
    /// - "retrieval.passage": For document passages
    /// - "separation": For text separation tasks
    /// - "classification": For classification tasks
    /// - "text-matching": For text matching tasks
    #[serde(default = "default_task")]
    #[schema(default = "retrieval.query")]
    pub task: String,
    pub user: Option<String>,
}

/// Input text can be a single string or an array of strings
#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum InputText {
    /// Single text input
    Single(String),
    /// Multiple text inputs for batch processing
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

#[derive(Debug, Serialize, ToSchema)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// Reranking models
#[derive(Debug, Deserialize, ToSchema)]
#[schema(example = json!({
    "query": "Machine learning là gì?",
    "documents": [
        "Machine learning là một nhánh của trí tuệ nhân tạo.",
        "Python là ngôn ngữ lập trình phổ biến.",
        "Deep learning sử dụng mạng neural nhiều lớp."
    ],
    "model": "jina-reranker-v2",
    "top_n": 2,
    "return_documents": true
}))]
pub struct RerankRequest {
    #[schema(example = "Machine learning là gì?")]
    pub query: String,
    #[schema(example = json!(["Machine learning là một nhánh của trí tuệ nhân tạo.", "Python là ngôn ngữ lập trình phổ biến."]))]
    pub documents: Vec<DocumentInput>,
    #[serde(default = "default_rerank_model")]
    #[schema(default = "jina-reranker-v2")]
    pub model: String,
    #[schema(example = 2)]
    pub top_n: Option<usize>,
    #[serde(default = "default_return_documents")]
    #[schema(default = true)]
    pub return_documents: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
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

#[derive(Debug, Serialize, ToSchema)]
pub struct RerankResponse {
    pub object: String,
    pub data: Vec<RerankResult>,
    pub model: String,
    pub usage: RerankUsage,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
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
