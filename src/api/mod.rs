pub mod health;
pub mod embeddings;
pub mod reranking;

use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

use health::{AppState, health_check};
use embeddings::create_embeddings;
use reranking::rerank_documents;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/embeddings", post(create_embeddings))
        .route("/v1/rerank", post(rerank_documents))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
