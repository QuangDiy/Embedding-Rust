use serde::Deserialize;
use std::sync::OnceLock;

static SETTINGS: OnceLock<Settings> = OnceLock::new();

#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    #[serde(default = "default_triton_url")]
    pub triton_url: String,

    #[serde(default = "default_timeout")]
    pub triton_http_connection_timeout: u64,

    #[serde(default = "default_timeout")]
    pub triton_http_network_timeout: u64,

    #[serde(default = "default_embedding_model")]
    pub embedding_model_name: String,

    #[serde(default = "default_reranker_model")]
    pub reranker_model_name: String,

    #[serde(default = "default_tokenizer_path")]
    pub tokenizer_path: String,

    #[serde(default = "default_reranker_tokenizer_path")]
    pub reranker_tokenizer_path: String,

    pub tokenizer_file: Option<String>,

    pub reranker_tokenizer_file: Option<String>,

    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,

    #[serde(default = "default_reranker_max_sequence_length")]
    pub reranker_max_sequence_length: usize,

    #[serde(default = "default_max_batch")]
    pub embedding_client_max_batch: usize,

    #[serde(default = "default_api_title")]
    pub api_title: String,

    #[serde(default = "default_api_description")]
    pub api_description: String,

    #[serde(default = "default_api_version")]
    pub api_version: String,

    pub api_key: Option<String>,

    #[serde(default)]
    pub require_api_key: bool,
}

fn default_triton_url() -> String {
    "triton:8000".to_string()
}

fn default_timeout() -> u64 {
    300
}

fn default_embedding_model() -> String {
    "jina-embeddings-v3".to_string()
}

fn default_reranker_model() -> String {
    "jina-reranker-v2".to_string()
}

fn default_tokenizer_path() -> String {
    "jinaai/jina-embeddings-v3".to_string()
}

fn default_reranker_tokenizer_path() -> String {
    "jinaai/jina-reranker-v2-base-multilingual".to_string()
}

fn default_max_sequence_length() -> usize {
    8192
}

fn default_reranker_max_sequence_length() -> usize {
    1024
}

fn default_max_batch() -> usize {
    8
}

fn default_api_title() -> String {
    "Jina AI API".to_string()
}

fn default_api_description() -> String {
    "OpenAI-compatible embedding and reranking API powered by Triton Inference Server".to_string()
}

fn default_api_version() -> String {
    "1.0.0".to_string()
}

impl Settings {
    pub fn new() -> Result<Self, config::ConfigError> {
        dotenv::dotenv().ok();

        let settings = config::Config::builder()
            .add_source(config::Environment::default().separator("_"))
            .build()?;

        let mut settings: Settings = settings.try_deserialize()?;
        
        settings.tokenizer_file = std::env::var("TOKENIZER_FILE").ok();
        settings.reranker_tokenizer_file = std::env::var("RERANKER_TOKENIZER_FILE").ok();
        
        Ok(settings)
    }

    pub fn get() -> &'static Settings {
        SETTINGS.get_or_init(|| {
            match Settings::new() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("FATAL: Failed to load settings: {}", e);
                    eprintln!("Error details: {:?}", e);
                    panic!("Cannot continue without valid settings: {}", e);
                }
            }
        })
    }
}
