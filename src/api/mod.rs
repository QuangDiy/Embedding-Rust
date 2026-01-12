pub mod health;
pub mod embeddings;
pub mod reranking;
pub mod openapi;

use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use health::{AppState, health_check};
use embeddings::create_embeddings;
use reranking::rerank_documents;
use openapi::ApiDoc;

pub fn create_router(state: Arc<AppState>) -> Router {
    // Create API routes
    let api_routes = Router::new()
        .route("/health", get(health_check))
        .route("/v1/embeddings", post(create_embeddings))
        .route("/v1/rerank", post(rerank_documents))
        .with_state(state);

    // Merge with Swagger UI
    api_routes
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(TraceLayer::new_for_http())
}
