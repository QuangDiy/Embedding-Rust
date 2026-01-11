/*!
Download Jina ONNX models using Hugging Face Hub

This script downloads ONNX model files from Jina AI repositories:
- jinaai/jina-embeddings-v3
- jinaai/jina-reranker-v2-base-multilingual

Models are saved to the Triton model repository structure.
*/

use std::fs;
use std::io::Write;
use std::path::Path;
use anyhow::{Context, Result};

const HF_BASE_URL: &str = "https://huggingface.co";

struct ModelDownload {
    repo_id: &'static str,
    file_path: &'static str,
    target_dir: &'static str,
    model_name: &'static str,
}

impl ModelDownload {
    const EMBEDDINGS: ModelDownload = ModelDownload {
        repo_id: "jinaai/jina-embeddings-v3",
        file_path: "onnx/model_fp16.onnx",
        target_dir: "model_repository/jina-embeddings-v3/1",
        model_name: "Jina Embeddings v3",
    };

    const RERANKER: ModelDownload = ModelDownload {
        repo_id: "jinaai/jina-reranker-v2-base-multilingual",
        file_path: "onnx/model_fp16.onnx",
        target_dir: "model_repository/jina-reranker-v2/1",
        model_name: "Jina Reranker v2 Base Multilingual",
    };
}

fn ensure_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("Failed to create directory: {}", path.display()))
}

async fn download_model(model: &ModelDownload) -> Result<bool> {
    println!("\n{}", "=".repeat(50));
    println!("Downloading {}", model.model_name);
    println!("{}", "=".repeat(50));
    println!("Repository: {}", model.repo_id);
    println!("File: {}", model.file_path);
    println!("Target directory: {}\n", model.target_dir);

    let target_dir = Path::new(model.target_dir);
    ensure_dir(target_dir)?;

    // Construct Hugging Face download URL
    let download_url = format!(
        "{}/{}/resolve/main/{}",
        HF_BASE_URL, model.repo_id, model.file_path
    );

    println!("Download URL: {}", download_url);

    // Create HTTP client
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout
        .build()?;

    // Download file
    let response = client
        .get(&download_url)
        .send()
        .await
        .with_context(|| format!("Failed to download from {}", download_url))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download: HTTP status {}",
            response.status()
        );
    }

    // Get file name from path
    let file_name = Path::new(model.file_path)
        .file_name()
        .context("Invalid file path")?;
    let target_file = target_dir.join(file_name);

    // Write file with progress
    let total_size = response.content_length().unwrap_or(0);
    println!("Total size: {:.2} MB", total_size as f64 / (1024.0 * 1024.0));

    let mut file = fs::File::create(&target_file)
        .with_context(|| format!("Failed to create file: {}", target_file.display()))?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading chunk")?;
        file.write_all(&chunk)
            .context("Error writing to file")?;
        
        downloaded += chunk.len() as u64;
        
        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            print!("\rProgress: {:.2}%", progress);
            std::io::stdout().flush().ok();
        }
    }
    
    println!();

    // Verify file was created
    if target_file.exists() {
        let file_size = fs::metadata(&target_file)?.len() as f64 / (1024.0 * 1024.0);
        println!("✓ Downloaded to: {}", target_file.display());
        println!("  File size: {:.2} MB", file_size);
        Ok(true)
    } else {
        println!("✗ Model file not found at expected location: {}", target_file.display());
        Ok(false)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("Jina AI Models Download Script");
    println!("{}", "=".repeat(60));

    let embeddings_success = download_model(&ModelDownload::EMBEDDINGS)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Embeddings download failed: {}", e);
            false
        });

    let reranker_success = download_model(&ModelDownload::RERANKER)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Reranker download failed: {}", e);
            false
        });

    println!("\n{}", "=".repeat(60));
    if embeddings_success && reranker_success {
        println!("All models downloaded successfully!");
    } else {
        println!("Some models failed to download");
    }
    println!("{}", "=".repeat(60));

    if embeddings_success && reranker_success {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
