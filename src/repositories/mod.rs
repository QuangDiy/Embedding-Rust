pub mod triton_client;

use crate::error::AppError;
use async_trait::async_trait;

#[async_trait]
pub trait EmbeddingRepository: Send + Sync {
    async fn generate_embeddings(
        &self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
        task_id: i64,
    ) -> Result<Vec<Vec<f32>>, AppError>;

    async fn is_ready(&self) -> Result<bool, AppError>;
}

#[async_trait]
pub trait RerankingRepository: Send + Sync {
    async fn generate_scores(
        &self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
    ) -> Result<Vec<f32>, AppError>;

    async fn is_ready(&self) -> Result<bool, AppError>;
}
