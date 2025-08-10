# LoRA Fine-tuning Pipeline for Ollama

A complete pipeline for fine-tuning Llama models using LoRA (Low-Rank Adaptation) with CSV data, optimized for Apple M4 Macs.

## 🚀 Features

- **Easy CSV Upload**: Upload training data in simple CSV format
- **LoRA Fine-tuning**: Efficient training using Unsloth
- **Web Interface**: User-friendly dashboard for managing training jobs
- **Apple M4 Optimized**: Leverages Apple Silicon for fast training
- **Ollama Integration**: Automatically prepares models for Ollama
- **Progress Tracking**: Real-time training progress and logs
- **Multiple Models**: Support for different Llama variants

## 📋 Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11
- At least 16GB RAM (32GB recommended)
- 50GB+ free disk space

## 🛠️ Installation

### 1. Quick Setup

```bash
# Run the setup script
./finetuning/setup.sh
```

### 2. Manual Setup

```bash
# Create virtual environment
python3.11 -m venv finetuning_env
source finetuning_env/bin/activate

# Install dependencies
pip install -r finetuning/requirements.txt
```

## 🎯 Usage

### Web Interface (Recommended)

1. **Start the web interface:**
   ```bash
   ./finetuning/start_webui.sh
   ```

2. **Open your browser:** http://localhost:8888

3. **Upload CSV data:**
   - Format: 2 columns (user_query, system_response)
   - Example provided at: `finetuning/data/example_training_data.csv`

4. **Configure training:**
   - Set epochs, batch size, learning rate
   - Choose base model
   - Name your fine-tuned model

5. **Monitor progress:**
   - Real-time progress bar
   - Live training logs
   - Training metrics

6. **Import to Ollama:**
   - One-click import when training completes

### Command Line Interface

```bash
# Activate environment
source finetuning_env/bin/activate
cd finetuning/scripts

# Process CSV data
python data_processor.py

# Start training
python lora_trainer.py \
  --data ../data/your_data_processed_train.jsonl \
  --val_data ../data/your_data_processed_val.jsonl \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --run_name my-custom-model \
  --save_for_ollama
```

## 📊 CSV Data Format

Your CSV file should have exactly 2 columns:

| user_query | system_response |
|------------|-----------------|
| "What is AI?" | "Artificial Intelligence (AI) is a branch of computer science..." |
| "How to code in Python?" | "Python is a beginner-friendly programming language..." |

### Best Practices:

- **Quality over Quantity**: 100-1000 high-quality examples often work better than thousands of poor ones
- **Diverse Examples**: Include various types of queries and responses
- **Consistent Style**: Keep response style consistent throughout your dataset
- **Length Balance**: Avoid extremely long or short responses

## ⚙️ Configuration Options

### Training Parameters

- **Epochs (1-10)**: Number of training iterations
  - More epochs = longer training, potentially better results
  - Start with 3, increase if needed

- **Batch Size (1-8)**: Number of examples processed together
  - Higher = faster training but more memory usage
  - Recommended: 2-4 for 16GB RAM

- **Learning Rate (0.0001-0.01)**: How fast the model learns
  - Default: 2e-4 works well for most cases
  - Lower for fine adjustments, higher for major changes

### Model Options

- `unsloth/llama-3.1-8b-bnb-4bit`: Latest Llama 3.1 8B (Recommended)
- `unsloth/llama-3.1-8b-instruct-bnb-4bit`: Instruction-tuned variant

## 🔧 Advanced Usage

### Custom LoRA Configuration

Edit `finetuning/scripts/lora_trainer.py` to adjust:

```python
# LoRA parameters
r=16,  # Rank (higher = more parameters)
lora_alpha=16,  # Scaling factor
lora_dropout=0.1,  # Dropout rate
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Target layers
```

### Monitoring with Weights & Biases

1. Create account at https://wandb.ai
2. Run: `wandb login`
3. Enable in web interface or use `--use_wandb` flag

## 📁 Project Structure

```
finetuning/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── setup.sh                # Setup script
├── start_webui.sh          # Web interface launcher
├── data/                   # Training data
│   ├── uploads/           # Uploaded CSV files
│   └── example_training_data.csv
├── models/                 # Trained models
├── logs/                   # Training logs
├── scripts/               # Core scripts
│   ├── data_processor.py  # CSV processing
│   └── lora_trainer.py    # Training logic
└── web_interface/         # Web dashboard
    ├── app.py            # FastAPI backend
    ├── templates/        # HTML templates
    └── static/           # CSS/JS assets
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Use smaller model variant
   - Close other applications

2. **Training Very Slow**
   - Check if using CPU instead of GPU
   - Reduce sequence length
   - Increase batch size if memory allows

3. **Poor Results**
   - Increase training epochs
   - Improve data quality
   - Add more diverse examples
   - Try different learning rates

### Performance Tips

- **For 16GB RAM**: batch_size=2, epochs=3
- **For 32GB+ RAM**: batch_size=4-8, epochs=5+
- **Monitor Activity Monitor**: Keep memory usage under 80%

## 🔗 Integration with Ollama

After training completes:

1. **Automatic Import** (Web Interface):
   - Click "Import to Ollama" button

2. **Manual Import**:
   ```bash
   ollama create my-model -f finetuning/models/ollama_my-model/Modelfile
   ```

3. **Test Your Model**:
   ```bash
   ollama run my-model "Your test question here"
   ```

## 📈 Monitoring Training

- **Progress Bar**: Visual training progress
- **Live Logs**: Real-time training information
- **Validation Metrics**: Model performance tracking
- **Resource Usage**: Memory and GPU utilization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review training logs
3. Create an issue with detailed information

---

**Happy Fine-tuning!** 🎉