#!/bin/bash

echo "🚀 Starting LoRA Fine-tuning Web Interface..."

# Check if virtual environment exists
if [ ! -d "finetuning_env" ]; then
    echo "❌ Virtual environment not found. Please run ./finetuning/setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source finetuning_env/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/finetuning/scripts"

# Start the web interface
echo "🌐 Starting web interface on http://localhost:8888..."
echo "📊 Access the fine-tuning dashboard in your browser"
echo "🛑 Press Ctrl+C to stop"
echo ""

cd finetuning/web_interface
python app.py