use axum::{
    extract::State,
    Json,
};
use std::sync::Arc;
use tracing::info;

use crate::api::health::AppState;
use crate::error::AppError;
use crate::models::{
    RerankRequest, RerankResponse, RerankResult, RerankUsage,
};

#[utoipa::path(
    post,
    path = "/v1/rerank",
    tag = "Reranking",
    request_body = RerankRequest,
    responses(
        (status = 200, description = "Successfully reranked documents", body = RerankResponse),
        (status = 400, description = "Bad request"),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn rerank_documents(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, AppError> {
    let documents: Vec<String> = request.documents
        .iter()
        .map(|doc| doc.as_text())
        .collect();

    let result_models = state.reranking_service
        .rerank_documents(
            request.query.clone(),
            documents.clone(),
            request.top_n,
            request.return_documents,
        )
        .await?;

    let results: Vec<RerankResult> = result_models
        .into_iter()
        .map(|model| RerankResult {
            index: model.index,
            relevance_score: model.relevance_score,
            document: model.document,
        })
        .collect();

    let response = RerankResponse {
        object: "list".to_string(),
        data: results,
        model: request.model,
        usage: RerankUsage {
            total_tokens: 0,
        },
    };

    info!("Successfully reranked {} documents", documents.len());
    Ok(Json(response))
}
