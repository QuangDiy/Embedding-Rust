use axum::{
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tracing::{info, warn};

use crate::config::Settings;

pub async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let settings = Settings::get();
    
    if !settings.require_api_key {
        return Ok(next.run(request).await);
    }

    let expected_key = match &settings.api_key {
        Some(key) if !key.is_empty() => key,
        _ => {
            warn!("REQUIRE_API_KEY is true but API_KEY is not configured");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let auth_header = headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    let provided_key = if auth_header.starts_with("Bearer ") {
        &auth_header[7..]
    } else {
        auth_header
    };

    if provided_key.is_empty() {
        warn!("Missing API key in request");
        return Err(StatusCode::UNAUTHORIZED);
    }

    if provided_key != expected_key {
        warn!("Invalid API key provided");
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(next.run(request).await)
}

pub async fn logging_middleware(
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path().to_string();
    
    let response = next.run(request).await;
    
    let status = response.status();
    
    info!(
        "{} {} - {}",
        method,
        path,
        status.as_u16()
    );
    
    response
}

