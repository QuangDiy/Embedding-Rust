# Build stage
FROM rust:1.83-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure Cargo for better network reliability
ENV CARGO_NET_RETRY=10
ENV CARGO_NET_TIMEOUT=300
ENV CARGO_HTTP_TIMEOUT=300
ENV CARGO_HTTP_MULTIPLEXING=false
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Copy Cargo configuration for mirrors
COPY .cargo/config.toml /usr/local/cargo/config.toml

# Copy manifests
COPY Cargo.toml ./

# Create a dummy main.rs to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source code
COPY src ./src

# Build application and download tool
# Remove the dummy binary and force rebuild with actual source
RUN rm -f target/release/embedding-rust target/release/download_models && \
    touch src/main.rs && \
    cargo build --release && \
    cargo build --release --bin download_models

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binaries from builder
COPY --from=builder /app/target/release/embedding-rust /app/embedding-rust
COPY --from=builder /app/target/release/download_models /app/download_models

# Set environment variables
ENV RUST_LOG=info
ENV TRITON_URL=triton:8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["/app/embedding-rust"]
