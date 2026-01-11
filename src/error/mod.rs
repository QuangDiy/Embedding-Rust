use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Triton connection error: {0}")]
    TritonConnection(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not ready: {0}")]
    NotReady(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::Validation(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Inference(msg) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to generate embeddings: {}", msg)),
            AppError::TritonConnection(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
            AppError::Tokenization(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::NotReady(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
        };

        let body = Json(json!({
            "error": error_message
        }));

        (status, body).into_response()
    }
}

impl From<reqwest::Error> for AppError {
    fn from(err: reqwest::Error) -> Self {
        AppError::TritonConnection(err.to_string())
    }
}

impl From<tokenizers::Error> for AppError {
    fn from(err: tokenizers::Error) -> Self {
        AppError::Tokenization(err.to_string())
    }
}
