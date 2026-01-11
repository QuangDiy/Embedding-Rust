use crate::error::AppError;
use crate::models::{EmbeddingModel, get_task_id};
use crate::repositories::triton_client::TritonClient;
use crate::services::tokenizer_service::TokenizerService;
use crate::config::Settings;
use tracing::info;

pub struct EmbeddingService {
    client: TritonClient,
    tokenizer_service: TokenizerService,
}

impl EmbeddingService {
    pub fn new() -> Result<Self, AppError> {
        let settings = Settings::get();
        let client = TritonClient::new(settings.embedding_model_name.clone())?;
        let tokenizer_service = TokenizerService::new();

        Ok(Self {
            client,
            tokenizer_service,
        })
    }

    pub async fn create_embeddings(
        &self,
        texts: Vec<String>,
        task: &str,
    ) -> Result<Vec<EmbeddingModel>, AppError> {
        if texts.is_empty() {
            return Err(AppError::Validation("Text input cannot be empty".to_string()));
        }

        let task_id = get_task_id(task);
        info!("Generating embeddings for {} texts with task '{}'", texts.len(), task);

        let settings = Settings::get();
        let max_batch = settings.embedding_client_max_batch;

        let mut all_embeddings = Vec::new();

        for chunk in texts.chunks(max_batch) {
            let chunk_vec: Vec<String> = chunk.to_vec();
            let (input_ids, attention_mask) = self.tokenizer_service
                .tokenize_for_embedding(&chunk_vec)?;

            let embeddings = self.client
                .get_embeddings(&input_ids, &attention_mask, task_id)
                .await?;

            all_embeddings.extend(embeddings);
        }

        let embedding_models: Vec<EmbeddingModel> = all_embeddings
            .into_iter()
            .enumerate()
            .map(|(index, vector)| EmbeddingModel { vector, index })
            .collect();

        info!("Successfully generated {} embeddings", embedding_models.len());
        Ok(embedding_models)
    }

    pub async fn is_ready(&self) -> Result<bool, AppError> {
        let live = self.client.is_server_live().await?;
        let ready = self.client.is_model_ready().await?;
        Ok(live && ready)
    }
}
