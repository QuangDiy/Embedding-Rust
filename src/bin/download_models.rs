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
    files: &'static [&'static str],
    target_dir: &'static str,
    model_name: &'static str,
}

impl ModelDownload {
    const EMBEDDINGS: ModelDownload = ModelDownload {
        repo_id: "jinaai/jina-embeddings-v3",
        files: &[
            "onnx/model_fp16.onnx",
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        target_dir: "model_repository/jina-embeddings-v3/1",
        model_name: "Jina Embeddings v3",
    };

    const RERANKER: ModelDownload = ModelDownload {
        repo_id: "jinaai/jina-reranker-v2-base-multilingual",
        files: &[
            "onnx/model_fp16.onnx",
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        target_dir: "model_repository/jina-reranker-v2/1",
        model_name: "Jina Reranker v2 Base Multilingual",
    };
}

fn ensure_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("Failed to create directory: {}", path.display()))
}

async fn download_file(
    repo_id: &str,
    file_path: &str,
    target_dir: &Path,
    client: &reqwest::Client,
) -> Result<bool> {
    let file_name = Path::new(file_path)
        .file_name()
        .context("Invalid file path")?;
    let target_file = target_dir.join(file_name);

    if target_file.exists() {
        if let Ok(metadata) = fs::metadata(&target_file) {
            let file_size = metadata.len();
            let min_size = if file_path.ends_with(".onnx") { 1024 * 1024 } else { 100 }; // 1MB for ONNX, 100 bytes for others
            if file_size > min_size {
                let file_size_display = if file_size > 1024 * 1024 {
                    format!("{:.2} MB", file_size as f64 / (1024.0 * 1024.0))
                } else {
                    format!("{:.2} KB", file_size as f64 / 1024.0)
                };
                println!("  {} already exists ({})", file_name.to_string_lossy(), file_size_display);
                return Ok(true);
            } else {
                println!("  {} corrupted, re-downloading...", file_name.to_string_lossy());
            }
        }
    }

    let download_url = format!(
        "{}/{}/resolve/main/{}",
        HF_BASE_URL, repo_id, file_path
    );

    println!("  Downloading {}...", file_name.to_string_lossy());

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

    let total_size = response.content_length().unwrap_or(0);
    let mut file = fs::File::create(&target_file)
        .with_context(|| format!("Failed to create file: {}", target_file.display()))?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    
    let show_progress = total_size > 10 * 1024 * 1024; // > 10MB
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading chunk")?;
        file.write_all(&chunk)
            .context("Error writing to file")?;
        
        downloaded += chunk.len() as u64;
        
        if show_progress && total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            print!("\r    Progress: {:.1}%", progress);
            std::io::stdout().flush().ok();
        }
    }
    
    if show_progress {
        println!();
    }

    if target_file.exists() {
        let file_size = fs::metadata(&target_file)?.len();
        let size_display = if file_size > 1024 * 1024 {
            format!("{:.2} MB", file_size as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} KB", file_size as f64 / 1024.0)
        };
        println!("  Downloaded {} ({})", file_name.to_string_lossy(), size_display);
        Ok(true)
    } else {
        println!("  Failed to download {}", file_name.to_string_lossy());
        Ok(false)
    }
}

async fn download_model(model: &ModelDownload) -> Result<bool> {
    println!("\n{}", "=".repeat(50));
    println!("{}", model.model_name);
    println!("{}", "=".repeat(50));
    println!("Repository: {}", model.repo_id);
    println!("Target: {}", model.target_dir);
    println!();

    let target_dir = Path::new(model.target_dir);
    ensure_dir(target_dir)?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()?;

    let mut success_count = 0;
    let mut fail_count = 0;

    for file_path in model.files {
        match download_file(model.repo_id, file_path, target_dir, &client).await {
            Ok(true) => success_count += 1,
            Ok(false) => fail_count += 1,
            Err(e) => {
                println!("  Error downloading {}: {}", file_path, e);
                fail_count += 1;
            }
        }
    }

    println!("\n{} files ready, {} failed", success_count, fail_count);
    Ok(fail_count == 0)
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
