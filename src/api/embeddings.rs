use axum::{
    extract::State,
    Json,
};
use std::sync::Arc;
use tracing::info;

use crate::api::health::AppState;
use crate::error::AppError;
use crate::models::{
    EmbeddingRequest, EmbeddingResponse, EmbeddingData, EmbeddingUsage,
};

pub async fn create_embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, AppError> {
    let texts = request.input.to_vec();
    
    let embedding_models = state.embedding_service
        .create_embeddings(texts.clone(), &request.task)
        .await?;

    let embedding_data: Vec<EmbeddingData> = embedding_models
        .into_iter()
        .map(|model| EmbeddingData {
            object: "embedding".to_string(),
            embedding: model.vector,
            index: model.index,
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: embedding_data,
        model: request.model,
        usage: EmbeddingUsage {
            prompt_tokens: 0,
            total_tokens: 0,
        },
    };

    info!("Successfully processed embedding request for {} texts", texts.len());
    Ok(Json(response))
}
