mod api;
mod config;
mod error;
mod models;
mod repositories;
mod services;

use std::sync::Arc;
use tracing::info;
use tracing_subscriber;

use api::{create_router, health::AppState};
use config::Settings;
use services::{
    embedding_service::EmbeddingService,
    reranking_service::RerankingService,
    tokenizer_service::TokenizerService,
};

#[tokio::main]
async fn main() {
    // Initialize tracing/logging
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("Starting Embedding Rust API...");

    // Load settings
    let settings = Settings::get();
    info!("Loaded settings");

    // Initialize tokenizers
    info!("Loading embedding tokenizer...");
    TokenizerService::load_embedding_tokenizer()
        .expect("Failed to load embedding tokenizer");
    info!("Embedding tokenizer loaded");

    info!("Loading reranker tokenizer...");
    TokenizerService::load_reranker_tokenizer()
        .expect("Failed to load reranker tokenizer");
    info!("Reranker tokenizer loaded");

    // Create services
    info!("Initializing embedding service...");
    let embedding_service = EmbeddingService::new()
        .expect("Failed to create embedding service");
    
    info!("Initializing reranking service...");
    let reranking_service = RerankingService::new()
        .expect("Failed to create reranking service");

    // Create shared state
    let state = Arc::new(AppState {
        embedding_service: Arc::new(embedding_service),
        reranking_service: Arc::new(reranking_service),
    });

    // Check if services are ready
    info!("Checking service readiness...");
    let embedding_ready = state.embedding_service.is_ready().await.unwrap_or(false);
    let reranking_ready = state.reranking_service.is_ready().await.unwrap_or(false);
    
    if embedding_ready {
        info!("Embedding service is ready");
    } else {
        info!("Warning: Embedding service is not ready");
    }
    
    if reranking_ready {
        info!("Reranking service is ready");
    } else {
        info!("Warning: Reranking service is not ready");
    }

    // Create router
    let app = create_router(state);

    // Start server
    let addr = "0.0.0.0:8000";
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");

    info!("🚀 {} v{} listening on http://{}", 
        settings.api_title,
        settings.api_version,
        addr
    );

    axum::serve(listener, app)
        .await
        .expect("Failed to start server");
}

