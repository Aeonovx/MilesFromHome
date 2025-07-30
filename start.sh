#!/bin/bash
set -e

echo "🚀 Starting iTethr Bot (Hugging Face) on Railway..."

# Create necessary directories
mkdir -p ./logs ./documents ./knowledge_base

# Get Railway port
export PORT=${PORT:-7860}
echo "🔌 Using PORT: $PORT"

# Set Railway environment
export RAILWAY_ENVIRONMENT=production
export GRADIO_SHARE=false

echo "🤖 Starting iTethr Bot with Hugging Face Transformers..."

# Start the application directly (no Ollama needed)
exec python app.py 2>&1 | tee ./logs/app.log