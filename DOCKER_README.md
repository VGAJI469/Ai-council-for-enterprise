# 🐳 Docker Support

You can run the entire AI Enterprise Council stack seamlessly using Docker Compose.

The Docker setup handles:
1. Building the FastAPI backend (`python:3.13-slim`).
2. Setting up the Ollama container and persisting downloaded models in a Docker volume.
3. Automatically pulling all required models (`deepseek-r1:7b`, `mixtral:8x7b`, `llama3:8b`, `phi3:mini`, and `nomic-embed-text`) during initialization.

## Getting Started

1. Ensure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
2. From the root directory of the project, run:
   ```bash
   docker compose up -d
   ```
3. The API will be available at `http://localhost:8000` (or your Docker host IP) and the Ollama service will be running on port `11434`. Wait a few minutes for the initial model downloads to complete.

*(Note: Ensure your machine meets the memory requirements to run these models locally).*
