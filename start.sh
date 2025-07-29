#!/bin/bash
set -e

echo "🚀 Starting iTethr Bot on Railway..."

# Get Railway port or default
export PORT=${PORT:-7860}
export API_PORT=${API_PORT:-5001}

echo "🔌 Using PORT: $PORT"
echo "🔗 Using API_PORT: $API_PORT"

# Start Ollama service in background
echo "🤖 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
sleep 20

# Pull the required model
echo "📥 Pulling AI model: qwen2.5:1.5b..."
ollama pull qwen2.5:1.5b

# Set Railway-specific environment variables
export RAILWAY_ENVIRONMENT=production
export GRADIO_SHARE=false

echo "🌐 Starting iTethr Bot on port $PORT..."
echo "🔗 Health endpoint will be available at /health"

# Start the bot with error handling
python app.py

# Cleanup on exit
trap "echo '🛑 Shutting down...'; kill $OLLAMA_PID 2>/dev/null || true" EXIT