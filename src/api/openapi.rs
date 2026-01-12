use utoipa::OpenApi;

use crate::models::{
    EmbeddingRequest, InputText, EmbeddingResponse, EmbeddingData, EmbeddingUsage,
    RerankRequest, DocumentInput, RerankResponse, RerankResult, RerankUsage,
};
use crate::api::health::{HealthResponse, ServiceStatus};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Embedding Rust API",
        version = "0.1.0",
        description = "High-performance embedding and reranking API powered by Rust and Triton Inference Server",
        contact(
            name = "API Support",
            email = "support@example.com"
        ),
        license(
            name = "MIT",
        )
    ),
    servers(
        (url = "http://localhost:8000", description = "Local development server"),
    ),
    paths(
        crate::api::health::health_check,
        crate::api::embeddings::create_embeddings,
        crate::api::reranking::rerank_documents,
    ),
    components(
        schemas(
            // Health schemas
            HealthResponse,
            ServiceStatus,
            // Embedding schemas
            EmbeddingRequest,
            InputText,
            EmbeddingResponse,
            EmbeddingData,
            EmbeddingUsage,
            // Reranking schemas
            RerankRequest,
            DocumentInput,
            RerankResponse,
            RerankResult,
            RerankUsage,
        )
    ),
    tags(
        (name = "Health", description = "Health check endpoints"),
        (name = "Embeddings", description = "Text embedding generation endpoints"),
        (name = "Reranking", description = "Document reranking endpoints")
    )
)]
pub struct ApiDoc;
