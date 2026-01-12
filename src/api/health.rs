use axum::{
    extract::State,
    Json,
};
use serde::Serialize;
use std::sync::Arc;
use utoipa::ToSchema;

use crate::services::{
    embedding_service::EmbeddingService,
    reranking_service::RerankingService,
};

pub struct AppState {
    pub embedding_service: Arc<EmbeddingService>,
    pub reranking_service: Arc<RerankingService>,
}

#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub embedding_service: ServiceStatus,
    pub reranking_service: ServiceStatus,
}

#[derive(Serialize, ToSchema)]
pub struct ServiceStatus {
    pub ready: bool,
}

#[utoipa::path(
    get,
    path = "/health",
    tag = "Health",
    responses(
        (status = 200, description = "Service health status", body = HealthResponse)
    )
)]
pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    let embedding_ready = state.embedding_service.is_ready().await.unwrap_or(false);
    let reranking_ready = state.reranking_service.is_ready().await.unwrap_or(false);

    Json(HealthResponse {
        status: "ok".to_string(),
        embedding_service: ServiceStatus {
            ready: embedding_ready,
        },
        reranking_service: ServiceStatus {
            ready: reranking_ready,
        },
    })
}
