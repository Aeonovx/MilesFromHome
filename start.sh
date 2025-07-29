#!/bin/bash
set -e

echo "🚀 Starting iTethr Bot on Railway..."

# Create necessary directories
mkdir -p ./logs ./documents ./knowledge_base

# Get Railway port or default
export PORT=${PORT:-7860}
export API_PORT=${API_PORT:-5001}

echo "🔌 Using PORT: $PORT"
echo "🔗 Using API_PORT: $API_PORT"

# Function to check if Ollama is responsive
check_ollama() {
    curl -s -f http://localhost:11434/api/tags >/dev/null
}

# Start Ollama service with retry mechanism
echo "🤖 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama with timeout
TIMEOUT=60
ELAPSED=0
echo "⏳ Waiting for Ollama to start (timeout: ${TIMEOUT}s)..."
until check_ollama || [ $ELAPSED -ge $TIMEOUT ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "Still waiting... ($ELAPSED/$TIMEOUT seconds)"
done

if ! check_ollama; then
    echo "❌ Ollama failed to start within ${TIMEOUT} seconds"
    exit 1
fi
echo "✅ Ollama started successfully"

# Check and pull model with timeout
MODEL="qwen2.5:1.5b"
echo "🔍 Checking for model: $MODEL"
if ! ollama list | grep -q "$MODEL"; then
    echo "📥 Pulling model: $MODEL..."
    timeout 300 ollama pull $MODEL || {
        echo "❌ Model pull failed after 5 minutes"
        exit 1
    }
fi
echo "✅ Model $MODEL is ready"

# Set Railway-specific environment variables
export RAILWAY_ENVIRONMENT=production
export GRADIO_SHARE=false
export DEBUG=true  # Temporary for debugging

# Start the health check endpoint first
echo "🏥 Starting health check endpoint..."
python -c "
from flask import Flask
app = Flask(__name__)
@app.route('/health')
def health(): return {'status': 'initializing'}, 200
" &
HEALTH_PID=$!

# Give health endpoint time to start
sleep 5

echo "🌐 Starting iTethr Bot on port $PORT..."
# Start the main application
exec python app.py 2>&1 | tee ./logs/app.log