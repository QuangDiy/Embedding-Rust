use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::json;
use std::sync::Arc;

use crate::services::embedding_service::EmbeddingService;
use crate::services::reranking_service::RerankingService;

pub struct AppState {
    pub embedding_service: Arc<EmbeddingService>,
    pub reranking_service: Arc<RerankingService>,
}

pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let embedding_ready = state.embedding_service.is_ready().await.unwrap_or(false);
    let reranking_ready = state.reranking_service.is_ready().await.unwrap_or(false);

    if embedding_ready && reranking_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "healthy",
                "embedding_service": "ready",
                "reranking_service": "ready"
            }))
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "embedding_service": if embedding_ready { "ready" } else { "not ready" },
                "reranking_service": if reranking_ready { "ready" } else { "not ready" }
            }))
        )
    }
}
