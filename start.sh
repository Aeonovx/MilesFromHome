#!/bin/bash
set -e

echo "🚀 Starting iTethr Bot on Railway..."

# Create necessary directories
mkdir -p ./logs ./documents ./knowledge_base

# Get Railway port or default
export PORT=${PORT:-7860}

echo "🔌 Using PORT: $PORT"

# Function to check if Ollama is responsive
check_ollama() {
    curl -s -f http://localhost:11434/api/tags >/dev/null 2>&1
}

# Start Ollama service in background
echo "🤖 Starting Ollama service..."
ollama serve > ./logs/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama with reduced timeout for Railway
TIMEOUT=60  # Reduced from 120 to 60 seconds
ELAPSED=0
echo "⏳ Waiting for Ollama to start (timeout: ${TIMEOUT}s)..."
until check_ollama || [ $ELAPSED -ge $TIMEOUT ]; do
    sleep 3  # Reduced from 5 to 3 seconds
    ELAPSED=$((ELAPSED + 3))
    echo "Still waiting... ($ELAPSED/$TIMEOUT seconds)"
done

if ! check_ollama; then
    echo "❌ Ollama failed to start within ${TIMEOUT} seconds"
    echo "📝 Ollama logs:"
    tail -n 20 ./logs/ollama.log 2>/dev/null || echo "No Ollama logs available"
    exit 1
fi
echo "✅ Ollama started successfully"

# Check and pull model with aggressive timeout
MODEL="qwen2.5:1.5b"
echo "🔍 Checking for model: $MODEL"

# Start app early while model pulls in background if needed
if ! ollama list | grep -q "$MODEL"; then
    echo "📥 Model not found locally. Starting app with download in background..."
    
    # Pull model in background
    (
        echo "Starting background model pull..."
        timeout 180 ollama pull $MODEL > ./logs/model_pull.log 2>&1 || {
            echo "❌ Model pull failed or timed out after 3 minutes"
            echo "📝 Model pull logs:"
            tail -n 10 ./logs/model_pull.log 2>/dev/null
        }
    ) &
    
    # Give model pull a head start
    sleep 10
    
    # Check if model is now available
    if ollama list | grep -q "$MODEL"; then
        echo "✅ Model $MODEL is ready"
    else
        echo "⏳ Model still downloading, app will start anyway"
    fi
else
    echo "✅ Model $MODEL is ready"
fi

# Set Railway-specific environment variables
export RAILWAY_ENVIRONMENT=production
export GRADIO_SHARE=false

echo "🌐 Starting iTethr Bot on port $PORT..."

# Start the main application
exec python app.py 2>&1 | tee ./logs/app.log