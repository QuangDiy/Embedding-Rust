# Embedding Rust API

OpenAI-compatible embedding and reranking API powered by Triton Inference Server, written in Rust.

This is a Rust port of the original Python FastAPI application, providing the same functionality with improved performance and memory efficiency.

## Features

- 🚀 **High Performance**: Built with Rust and Axum for maximum throughput
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI embedding APIs
- 🧠 **Triton Integration**: Leverages NVIDIA Triton Inference Server
- 📊 **Dual Functionality**: Supports both embeddings and reranking
- 🔒 **Type Safe**: Rust's type system ensures reliability

## API Documentation

The API includes interactive Swagger UI documentation for easy testing and exploration.

### Accessing Swagger UI

Once the server is running, access the Swagger UI at:

```
http://localhost:8000/swagger-ui
```

The Swagger UI provides:
- **Interactive API testing**: Try out endpoints directly from your browser
- **Request/Response schemas**: View detailed model definitions
- **Example requests**: Pre-filled examples for quick testing
- **OpenAPI specification**: Download the full API spec at `/api-docs/openapi.json`

## API Endpoints

### Embeddings
- `POST /v1/embeddings` - Generate embeddings for text inputs

### Reranking
- `POST /v1/rerank` - Rerank documents based on relevance to a query

### Health
- `GET /health` - Check service health and readiness

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with Docker GPU support (for Triton Server)
- Model repository with Jina models

### Installation & Deployment


```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f embedding-rust-api

# Check health
curl http://localhost:8000/health

# Stop services
docker-compose down
```

## Configuration

All configuration is done via environment variables (loaded from `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `triton:8000` | Triton server address |
| `EMBEDDING_MODEL_NAME` | `jina-embeddings-v3` | Embedding model name |
| `RERANKER_MODEL_NAME` | `jina-reranker-v2` | Reranker model name |
| `TOKENIZER_PATH` | `jinaai/jina-embeddings-v3` | Embedding tokenizer path |
| `MAX_SEQUENCE_LENGTH` | `8192` | Max sequence length for embeddings |
| `EMBEDDING_CLIENT_MAX_BATCH` | `8` | Max batch size for processing |

## Usage Examples

### Create Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "jina-embeddings-v3",
    "task": "retrieval.query"
  }'
```

### Rerank Documents

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": ["ML is AI", "Dogs are animals", "Python is a language"],
    "top_n": 2
  }'
```

## Architecture

The application follows a layered architecture:

- **API Layer** (`src/api/`): HTTP routes and handlers
- **Service Layer** (`src/services/`): Business logic
- **Repository Layer** (`src/repositories/`): Triton client integration
- **Models** (`src/models/`): Request/response schemas
- **Config** (`src/config/`): Configuration management
- **Error** (`src/error/`): Error handling
