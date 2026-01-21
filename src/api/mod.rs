pub mod health;
pub mod embeddings;
pub mod reranking;
pub mod openapi;

use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use health::{AppState, health_check};
use embeddings::create_embeddings;
use reranking::rerank_documents;
use openapi::ApiDoc;
use crate::middleware::{auth_middleware, logging_middleware};

pub fn create_router(state: Arc<AppState>) -> Router {
    // Create protected API routes with auth middleware
    let protected_routes = Router::new()
        .route("/v1/embeddings", post(create_embeddings))
        .route("/v1/rerank", post(rerank_documents))
        .layer(middleware::from_fn(auth_middleware))
        .with_state(state.clone());

    // Public routes (health check and swagger)
    let public_routes = Router::new()
        .route("/health", get(health_check))
        .with_state(state);

    // Merge all routes and add logging middleware
    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(middleware::from_fn(logging_middleware))
}
