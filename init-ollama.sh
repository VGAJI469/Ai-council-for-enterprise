#!/bin/bash
# init-ollama.sh
# Initializes Ollama server and sequentially pulls required models

# Start the ollama server in the background
echo "Starting Ollama server..."
ollama serve &

# Record the Process ID
OLLAMA_PID=$!

# Wait for the Ollama API to become healthy
echo "Waiting for Ollama to become available..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "Ollama is running. Pulling required models sequentially..."

# Define the models required by the system
MODELS=(
    "deepseek-r1:7b"
    "mixtral:8x7b"
    "llama3:8b"
    "phi3:mini"
    "nomic-embed-text"
)

# Sequentially pull each model
for MODEL in "${MODELS[@]}"; do
    echo "Pulling $MODEL..."
    ollama pull "$MODEL"
    if [ $? -eq 0 ]; then
        echo "Successfully pulled $MODEL"
    else
        echo "Failed to pull $MODEL"
    fi
done

echo "Initialization complete. All models downloaded."

# Bring the ollama serve process to the foreground so the container stays running
wait $OLLAMA_PID
