mod api;
mod config;
mod error;
mod middleware;
mod models;
mod repositories;
mod services;

use std::sync::Arc;
use tracing::{info, error};
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
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    info!("Starting Embedding Rust API...");

    println!("Loading settings...");
    let settings = Settings::get();
    println!("Settings loaded successfully");
    info!("Loaded settings");
    info!("API Key configured: {}", settings.api_key.is_some());
    info!("Require API Key: {}", settings.require_api_key);

    // Initialize tokenizers
    info!("Loading embedding tokenizer...");
    match TokenizerService::load_embedding_tokenizer() {
        Ok(_) => info!("Embedding tokenizer loaded successfully"),
        Err(e) => {
            error!("Failed to load embedding tokenizer: {:?}", e);
            panic!("Cannot start without embedding tokenizer: {:?}", e);
        }
    }

    info!("Loading reranker tokenizer...");
    match TokenizerService::load_reranker_tokenizer() {
        Ok(_) => info!("Reranker tokenizer loaded successfully"),
        Err(e) => {
            error!("Failed to load reranker tokenizer: {:?}", e);
            panic!("Cannot start without reranker tokenizer: {:?}", e);
        }
    }

    // Create services
    info!("Initializing embedding service...");
    let embedding_service = match EmbeddingService::new() {
        Ok(service) => {
            info!("Embedding service initialized");
            service
        }
        Err(e) => {
            error!("Failed to create embedding service: {:?}", e);
            panic!("Cannot start without embedding service: {:?}", e);
        }
    };
    
    info!("Initializing reranking service...");
    let reranking_service = match RerankingService::new() {
        Ok(service) => {
            info!("Reranking service initialized");
            service
        }
        Err(e) => {
            error!("Failed to create reranking service: {:?}", e);
            panic!("Cannot start without reranking service: {:?}", e);
        }
    };

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

    info!("{} v{} listening on http://{}", 
        settings.api_title,
        settings.api_version,
        addr
    );

    axum::serve(listener, app)
        .await
        .expect("Failed to start server");
}

