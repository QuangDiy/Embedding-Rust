use crate::error::AppError;
use crate::config::Settings;
use tokenizers::tokenizer::Tokenizer;
use std::sync::OnceLock;
use tracing::{info, error};

static EMBEDDING_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
static RERANKER_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

pub struct TokenizerService;

impl TokenizerService {
    pub fn new() -> Self {
        Self
    }

    pub fn load_embedding_tokenizer() -> Result<(), AppError> {
        let settings = Settings::get();
        
        let file_path = settings.tokenizer_file.as_ref()
            .ok_or_else(|| {
                error!("TOKENIZER_FILE environment variable not set. Please set it to the path of tokenizer.json");
                AppError::Tokenization("TOKENIZER_FILE not configured".to_string())
            })?;
        
        info!("Loading embedding tokenizer from: {}", file_path);
        let tokenizer = Tokenizer::from_file(file_path)
            .map_err(|e| {
                error!("Failed to load embedding tokenizer from {}: {}", file_path, e);
                AppError::Tokenization(format!("Failed to load tokenizer: {}", e))
            })?;
        
        EMBEDDING_TOKENIZER.set(tokenizer).map_err(|_| 
            AppError::Internal("Embedding tokenizer already initialized".to_string())
        )?;
        
        info!("Embedding tokenizer loaded successfully");
        Ok(())
    }

    pub fn load_reranker_tokenizer() -> Result<(), AppError> {
        let settings = Settings::get();
        
        let file_path = settings.reranker_tokenizer_file.as_ref()
            .ok_or_else(|| {
                error!("RERANKER_TOKENIZER_FILE environment variable not set. Please set it to the path of tokenizer.json");
                AppError::Tokenization("RERANKER_TOKENIZER_FILE not configured".to_string())
            })?;
        
        info!("Loading reranker tokenizer from: {}", file_path);
        let tokenizer = Tokenizer::from_file(file_path)
            .map_err(|e| {
                error!("Failed to load reranker tokenizer from {}: {}", file_path, e);
                AppError::Tokenization(format!("Failed to load tokenizer: {}", e))
            })?;
        
        RERANKER_TOKENIZER.set(tokenizer).map_err(|_| 
            AppError::Internal("Reranker tokenizer already initialized".to_string())
        )?;
        
        info!("Reranker tokenizer loaded successfully");
        Ok(())
    }

    pub fn tokenize_for_embedding(
        &self,
        texts: &[String],
    ) -> Result<(Vec<Vec<i64>>, Vec<Vec<i64>>), AppError> {
        let tokenizer = EMBEDDING_TOKENIZER.get()
            .ok_or_else(|| AppError::Internal("Embedding tokenizer not initialized".to_string()))?;

        let settings = Settings::get();
        let max_length = settings.max_sequence_length;

        let mut all_input_ids = Vec::new();
        let mut all_attention_masks = Vec::new();

        for text in texts {
            let encoding = tokenizer
                .encode(text.clone(), true)
                .map_err(|e| AppError::Tokenization(e.to_string()))?;

            let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let mut attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

            // Truncate or pad to max_length
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
            } else {
                let padding = max_length - input_ids.len();
                input_ids.extend(vec![0; padding]);
                attention_mask.extend(vec![0; padding]);
            }

            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
        }

        Ok((all_input_ids, all_attention_masks))
    }

    pub fn tokenize_for_reranking(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<(Vec<Vec<i64>>, Vec<Vec<i64>>), AppError> {
        let tokenizer = RERANKER_TOKENIZER.get()
            .ok_or_else(|| AppError::Internal("Reranker tokenizer not initialized".to_string()))?;

        let settings = Settings::get();
        let max_length = settings.reranker_max_sequence_length;

        let mut all_input_ids = Vec::new();
        let mut all_attention_masks = Vec::new();

        for doc in documents {
            // Combine query and document
            let combined = format!("{} [SEP] {}", query, doc);
            
            let encoding = tokenizer
                .encode(combined, true)
                .map_err(|e| AppError::Tokenization(e.to_string()))?;

            let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let mut attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

            // Truncate or pad to max_length
            if input_ids.len() > max_length {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
            } else {
                let padding = max_length - input_ids.len();
                input_ids.extend(vec![0; padding]);
                attention_mask.extend(vec![0; padding]);
            }

            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
        }

        Ok((all_input_ids, all_attention_masks))
    }
}
