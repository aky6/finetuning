import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLORATrainer:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
                 max_seq_length: int = 512,
                 load_in_4bit: bool = False):
        """
        Simple LoRA trainer using transformers + PEFT (no Unsloth dependency)
        
        Args:
            model_name: Base model to fine-tune
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer with 4-bit quantization"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Configure quantization
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with proper device mapping
        # Force CPU device map on non-CUDA systems to avoid meta tensor issues
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16 if self.load_in_4bit else torch.float16
        elif torch.backends.mps.is_available():
            # For Apple Silicon, keep on CPU to avoid meta tensor and dtype issues
            device_map = "cpu"
            torch_dtype = torch.float32
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,  # Avoid meta tensor issues on CPU
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model loaded successfully with LoRA configuration")
    
    def prepare_dataset(self, jsonl_file: str):
        """Prepare dataset from JSONL file"""
        logger.info(f"Loading dataset from: {jsonl_file}")
        
        # Load data
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Format for training
        formatted_data = []
        for example in data:
            messages = example["messages"]
            
            # Create training prompt using ChatML format
            conversation = ""
            for message in messages:
                if message["role"] == "user":
                    conversation += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                elif message["role"] == "assistant":
                    conversation += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
            
            formatted_data.append({"text": conversation})
        
        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset
    
    def train(self, 
              train_dataset,
              val_dataset=None,
              output_dir: str = "finetuning/models",
              num_train_epochs: int = 3,
              per_device_train_batch_size: int = 1,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 5,
              logging_steps: int = 10,
              save_steps: int = 100,
               use_wandb: bool = False,
              run_name: str = None):
        """
        Train the model with LoRA
        """
        if self.model is None:
            self.load_model()
        
        # Setup output directory
        if run_name is None:
            run_name = f"simple_lora_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        full_output_dir = f"{output_dir}/{run_name}"
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Configure Weights & Biases
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="ollama-lora-finetuning",
                    name=run_name,
                    config={
                        "model_name": self.model_name,
                        "max_seq_length": self.max_seq_length,
                        "num_train_epochs": num_train_epochs,
                        "learning_rate": learning_rate,
                        "batch_size": per_device_train_batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                    }
                )
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")
                use_wandb = False
        else:
            # Ensure any implicit wandb usage is disabled
            os.environ["WANDB_DISABLED"] = "true"
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=full_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=False,  # Disabled for Apple Silicon MPS compatibility
            bf16=False,  # Disabled for CPU compatibility
            logging_steps=logging_steps,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            save_steps=save_steps,
            save_total_limit=3,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=save_steps if val_dataset else None,
            load_best_model_at_end=True if val_dataset else False,
            dataloader_pin_memory=False,  # Better for limited memory
            remove_unused_columns=False,
            # Explicitly control reporting backends. Use empty list to disable.
            report_to=["wandb"] if use_wandb else [],
            run_name=run_name,
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(full_output_dir)
        
        # Save training stats
        stats = {
            "model_name": self.model_name,
            "training_examples": len(train_dataset),
            "validation_examples": len(val_dataset) if val_dataset else 0,
            "num_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "output_dir": full_output_dir,
            "training_completed": datetime.now().isoformat(),
        }
        
        with open(f"{full_output_dir}/training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training completed! Model saved to: {full_output_dir}")
        
        if use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass
        
        return full_output_dir
    
    def save_for_ollama(self, model_dir: str, model_name: str = "custom-llama"):
        """
        Save model in format compatible with Ollama
        """
        logger.info("Saving model for Ollama...")
        
        # Create Ollama directory relative to repository root
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]  # Go up to repo root
        ollama_dir = repo_root / f"finetuning/models/ollama_{model_name}"
        os.makedirs(ollama_dir, exist_ok=True)
        
        # Copy model files (LoRA adapters will be merged during inference)
        import shutil
        try:
            # Copy the adapter files
            for file in ["adapter_config.json", "adapter_model.safetensors"]:
                src = os.path.join(model_dir, file)
                dst = os.path.join(ollama_dir, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            
            # Copy tokenizer files
            for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                src = os.path.join(model_dir, file)
                dst = os.path.join(ollama_dir, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
        except Exception as e:
            logger.warning(f"Could not copy some files: {e}")
        
        # Create Ollama Modelfile
        modelfile_content = f"""FROM {self.model_name}
ADAPTER ./adapter_model.safetensors

TEMPLATE \"\"\"<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>
"""
        
        with open(f"{ollama_dir}/Modelfile", 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Model prepared for Ollama at: {ollama_dir}")
        logger.info(f"To import into Ollama, run:")
        logger.info(f"ollama create {model_name} -f {ollama_dir}/Modelfile")
        
        return ollama_dir

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple LoRA Fine-tuning")
    parser.add_argument("--data", required=True, help="Path to training JSONL file")
    parser.add_argument("--val_data", help="Path to validation JSONL file")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--run_name", help="Training run name")
    parser.add_argument("--save_for_ollama", action="store_true", help="Save model for Ollama")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SimpleLORATrainer(model_name=args.model)
    
    # Prepare datasets
    train_dataset = trainer.prepare_dataset(args.data)
    val_dataset = trainer.prepare_dataset(args.val_data) if args.val_data else None
    
    # Train
    model_dir = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        run_name=args.run_name,
    )
    
    # Save for Ollama if requested
    if args.save_for_ollama:
        trainer.save_for_ollama(model_dir, args.run_name or "custom-llama")

if __name__ == "__main__":
    main()