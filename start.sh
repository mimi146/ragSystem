#!/bin/bash
# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Make sure your API keys are set."
fi

# Run the server on port 8001
echo "Starting Weather RAG API on port 8001..."
uv run uvicorn main:app --host 0.0.0.0 --port 8001 --env-file .env
