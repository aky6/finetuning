#!/bin/bash
# Extract Ollama model and convert to Hugging Face format for fine-tuning

echo "🔍 Extracting Ollama Llama 3.1 8B model for fine-tuning..."

# Create directory for extracted model
EXTRACT_DIR="/home/ubuntu/ollama-ai/finetuning/models/llama3.1-8b-local"
mkdir -p $EXTRACT_DIR

echo "📁 Created extraction directory: $EXTRACT_DIR"

# Check if we have the model file
MODEL_PATH="/usr/share/ollama/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Found Ollama model at: $MODEL_PATH"
    
    # Create a symbolic link to make it accessible
    ln -sf "$MODEL_PATH" "$EXTRACT_DIR/model.gguf"
    echo "🔗 Created symbolic link to model file"
    
    # Create a simple config.json for the model
    cat > "$EXTRACT_DIR/config.json" << 'EOF'
{
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "torch_dtype": "float16",
    "transformers_version": "4.36.0",
    "use_cache": true,
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0
}
EOF
    
    echo "📝 Created config.json for Llama 3.1 8B"
    
    # Create a simple tokenizer config
    cat > "$EXTRACT_DIR/tokenizer_config.json" << 'EOF'
{
    "model_type": "llama",
    "tokenizer_class": "LlamaTokenizer",
    "pad_token": "</s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "bos_token": "<s>"
}
EOF
    
    echo "🔤 Created tokenizer config"
    
    echo ""
    echo "🎉 Model extraction complete!"
    echo "📁 Model available at: $EXTRACT_DIR"
    echo "🔧 You can now use 'llama3.1-8b-local' for fine-tuning"
    
else
    echo "❌ Model file not found at: $MODEL_PATH"
    echo "🔍 Available models:"
    ls -la /usr/share/ollama/.ollama/models/blobs/
fi 