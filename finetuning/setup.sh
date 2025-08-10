#!/bin/bash

echo "🚀 Setting up LoRA Fine-tuning Pipeline..."

# Check if we're in the right directory
if [ ! -d "finetuning" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "finetuning_env" ]; then
    echo "📦 Creating Python 3.11 virtual environment..."
    python3.11 -m venv finetuning_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source finetuning_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r finetuning/requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p finetuning/{data/uploads,models,logs}

# Set up environment variables
echo "🌍 Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/finetuning/scripts"

# Create example CSV
echo "📄 Creating example CSV file..."
cat > finetuning/data/example_training_data.csv << EOF
user_query,system_response
"What is the capital of France?","The capital of France is Paris. It is located in the north-central part of the country and is known for its rich history, culture, and iconic landmarks like the Eiffel Tower."
"How do I make coffee?","To make coffee, start by grinding coffee beans to a medium-fine consistency. Add about 1-2 tablespoons of ground coffee per 6 oz of water. Pour hot water (195-205°F) over the grounds and let it brew for 4-6 minutes."
"Explain machine learning","Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions."
"What is Python?","Python is a high-level, interpreted programming language known for its simple syntax and readability. It's widely used for web development, data science, automation, and artificial intelligence applications."
"How to stay healthy?","To stay healthy, maintain a balanced diet with fruits and vegetables, exercise regularly, get adequate sleep (7-9 hours), stay hydrated, manage stress, and have regular medical check-ups."
EOF

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment: source finetuning_env/bin/activate"
echo "2. Start the web interface: cd finetuning/web_interface && python app.py"
echo "3. Or use command line: cd finetuning/scripts && python lora_trainer.py --help"
echo ""
echo "📝 Example training data created at: finetuning/data/example_training_data.csv"
echo "🌐 Web interface will be available at: http://localhost:8888"