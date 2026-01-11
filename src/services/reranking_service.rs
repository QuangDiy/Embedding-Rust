use crate::error::AppError;
use crate::models::RerankModel;
use crate::repositories::triton_client::TritonClient;
use crate::services::tokenizer_service::TokenizerService;
use crate::config::Settings;
use tracing::info;

pub struct RerankingService {
    client: TritonClient,
    tokenizer_service: TokenizerService,
}

impl RerankingService {
    pub fn new() -> Result<Self, AppError> {
        let settings = Settings::get();
        let client = TritonClient::new(settings.reranker_model_name.clone())?;
        let tokenizer_service = TokenizerService::new();

        Ok(Self {
            client,
            tokenizer_service,
        })
    }

    pub async fn rerank_documents(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: Option<usize>,
        return_documents: bool,
    ) -> Result<Vec<RerankModel>, AppError> {
        if documents.is_empty() {
            return Err(AppError::Validation("Documents cannot be empty".to_string()));
        }

        info!("Reranking {} documents", documents.len());

        let (input_ids, attention_mask) = self.tokenizer_service
            .tokenize_for_reranking(&query, &documents)?;

        let scores = self.client
            .get_scores(&input_ids, &attention_mask)
            .await?;

        let mut results: Vec<RerankModel> = scores
            .into_iter()
            .enumerate()
            .map(|(index, relevance_score)| {
                let document = if return_documents {
                    Some(documents[index].clone())
                } else {
                    None
                };
                RerankModel {
                    index,
                    relevance_score,
                    document,
                }
            })
            .collect();

        // Sort by relevance score in descending order
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        // Apply top_n filter if specified
        if let Some(n) = top_n {
            results.truncate(n);
        }

        info!("Successfully reranked documents, returning {} results", results.len());
        Ok(results)
    }

    pub async fn is_ready(&self) -> Result<bool, AppError> {
        let live = self.client.is_server_live().await?;
        let ready = self.client.is_model_ready().await?;
        Ok(live && ready)
    }
}
