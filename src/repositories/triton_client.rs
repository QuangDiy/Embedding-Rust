use crate::error::AppError;
use crate::config::Settings;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;
use tracing::{info, error};

#[derive(Debug, Serialize)]
struct TritonInferenceInput {
    name: String,
    shape: Vec<usize>,
    datatype: String,
    data: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct TritonInferenceOutput {
    name: String,
}

#[derive(Debug, Serialize)]
struct TritonInferRequest {
    inputs: Vec<TritonInferenceInput>,
    outputs: Vec<TritonInferenceOutput>,
}

#[derive(Debug, Deserialize)]
struct TritonInferResponse {
    outputs: Vec<TritonOutputData>,
}

#[derive(Debug, Deserialize)]
struct TritonOutputData {
    name: String,
    shape: Vec<usize>,
    datatype: String,
    data: Vec<f32>,
}

pub struct TritonClient {
    client: Client,
    triton_url: String,
    model_name: String,
}

impl TritonClient {
    pub fn new(model_name: String) -> Result<Self, AppError> {
        let settings = Settings::get();
        let timeout = Duration::from_secs(settings.triton_http_network_timeout);
        
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| AppError::TritonConnection(e.to_string()))?;

        Ok(Self {
            client,
            triton_url: format!("http://{}", settings.triton_url),
            model_name,
        })
    }

    pub async fn is_server_live(&self) -> Result<bool, AppError> {
        let url = format!("{}/v2/health/live", self.triton_url);
        let response = self.client.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    pub async fn is_model_ready(&self) -> Result<bool, AppError> {
        let url = format!("{}/v2/models/{}/ready", self.triton_url, self.model_name);
        let response = self.client.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    pub async fn get_embeddings(
        &self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
        task_id: i64,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        let batch_size = input_ids.len();
        let seq_length = input_ids[0].len();

        info!("Preparing inference request: batch_size={}, seq_length={}, task_id={}", 
              batch_size, seq_length, task_id);

        // Flatten input_ids and attention_mask
        let flat_input_ids: Vec<i64> = input_ids.iter().flatten().copied().collect();
        let flat_attention_mask: Vec<i64> = attention_mask.iter().flatten().copied().collect();
        let task_ids = vec![task_id; batch_size];

        let request = TritonInferRequest {
            inputs: vec![
                TritonInferenceInput {
                    name: "input_ids".to_string(),
                    shape: vec![batch_size, seq_length],
                    datatype: "INT64".to_string(),
                    data: flat_input_ids.iter().map(|&x| json!(x)).collect(),
                },
                TritonInferenceInput {
                    name: "attention_mask".to_string(),
                    shape: vec![batch_size, seq_length],
                    datatype: "INT64".to_string(),
                    data: flat_attention_mask.iter().map(|&x| json!(x)).collect(),
                },
                TritonInferenceInput {
                    name: "task_id".to_string(),
                    shape: vec![batch_size, 1],
                    datatype: "INT64".to_string(),
                    data: task_ids.iter().map(|&x| json!(x)).collect(),
                },
            ],
            outputs: vec![TritonInferenceOutput {
                name: "13049".to_string(),
            }],
        };

        let url = format!("{}/v2/models/{}/infer", self.triton_url, self.model_name);
        info!("Sending inference request to: {}", url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        info!("Received response with status: {}", status);

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            error!("Triton inference failed with status {}: {}", status, error_text);
            return Err(AppError::Inference(format!("Triton returned error {}: {}", status, error_text)));
        }

        let infer_response: TritonInferResponse = response.json().await
            .map_err(|e| AppError::Inference(format!("Failed to parse response: {}", e)))?;

        if let Some(output) = infer_response.outputs.first() {
            let embedding_dim = output.shape.get(1).copied().unwrap_or(0);
            let embeddings: Vec<Vec<f32>> = output.data
                .chunks(embedding_dim)
                .map(|chunk| chunk.to_vec())
                .collect();

            info!("Embeddings shape: [{}, {}]", embeddings.len(), embedding_dim);
            Ok(embeddings)
        } else {
            Err(AppError::Inference("No output from Triton".to_string()))
        }
    }

    pub async fn get_scores(
        &self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
    ) -> Result<Vec<f32>, AppError> {
        let batch_size = input_ids.len();
        let seq_length = input_ids[0].len();

        let flat_input_ids: Vec<i64> = input_ids.iter().flatten().copied().collect();
        let flat_attention_mask: Vec<i64> = attention_mask.iter().flatten().copied().collect();

        let request = TritonInferRequest {
            inputs: vec![
                TritonInferenceInput {
                    name: "input_ids".to_string(),
                    shape: vec![batch_size, seq_length],
                    datatype: "INT64".to_string(),
                    data: flat_input_ids.iter().map(|&x| json!(x)).collect(),
                },
                TritonInferenceInput {
                    name: "attention_mask".to_string(),
                    shape: vec![batch_size, seq_length],
                    datatype: "INT64".to_string(),
                    data: flat_attention_mask.iter().map(|&x| json!(x)).collect(),
                },
            ],
            outputs: vec![TritonInferenceOutput {
                name: "scores".to_string(),
            }],
        };

        let url = format!("{}/v2/models/{}/infer", self.triton_url, self.model_name);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AppError::Inference(format!("Triton returned error: {}", error_text)));
        }

        let infer_response: TritonInferResponse = response.json().await
            .map_err(|e| AppError::Inference(format!("Failed to parse response: {}", e)))?;

        if let Some(output) = infer_response.outputs.first() {
            Ok(output.data.clone())
        } else {
            Err(AppError::Inference("No output from Triton".to_string()))
        }
    }
}
