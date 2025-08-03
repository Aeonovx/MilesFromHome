#!/bin/bash
set -e

echo "🚀 Starting MilesFromHome News Aggregator on Railway..."

# Get Railway port, default to 8000
export PORT=${PORT:-8000}
echo "🔌 Using PORT: $PORT"

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port $PORT