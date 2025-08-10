import os
import json
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(self, 
                 model_name: str = "unsloth/llama-3.1-8b-bnb-4bit",
                 max_seq_length: int = 2048,
                 load_in_4bit: bool = True):
        """
        Initialize LoRA trainer with Unsloth for efficient training
        
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
        """Load model and tokenizer with Unsloth optimizations"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.load_in_4bit,
        )
        
        # Configure LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
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
            
            # Create training prompt
            conversation = ""
            for message in messages:
                if message["role"] == "user":
                    conversation += f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message['content']}<|eot_id|>"
                elif message["role"] == "assistant":
                    conversation += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message['content']}<|eot_id|>"
            
            formatted_data.append({"text": conversation})
        
        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset
    
    def train(self, 
              train_dataset,
              val_dataset=None,
              output_dir: str = "finetuning/models",
              num_train_epochs: int = 3,
              per_device_train_batch_size: int = 2,
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
            run_name = f"lora_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        full_output_dir = f"{output_dir}/{run_name}"
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if use_wandb:
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
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_available(),  # Use fp16 for GPU, bf16 for CPU
            bf16=torch.cuda.is_available(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=full_output_dir,
            save_steps=save_steps,
            save_total_limit=3,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=save_steps if val_dataset else None,
            load_best_model_at_end=True if val_dataset else False,
            report_to="wandb" if use_wandb else None,
            run_name=run_name,
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
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
            wandb.finish()
        
        return full_output_dir
    
    def save_for_ollama(self, model_dir: str, model_name: str = "custom-llama"):
        """
        Save model in format compatible with Ollama
        """
        logger.info("Saving model for Ollama...")
        
        # Load the fine-tuned model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,  # Save in full precision
        )
        
        # Merge LoRA weights
        model = FastLanguageModel.for_inference(model)
        
        # Save merged model
        ollama_dir = f"finetuning/models/ollama_{model_name}"
        os.makedirs(ollama_dir, exist_ok=True)
        
        model.save_pretrained(ollama_dir)
        tokenizer.save_pretrained(ollama_dir)
        
        # Create Ollama Modelfile
        modelfile_content = f"""FROM ./finetuning/models/ollama_{model_name}
TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"
PARAMETER stop <|eot_id|>
PARAMETER stop <|end_of_text|>
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
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning with Unsloth")
    parser.add_argument("--data", required=True, help="Path to training JSONL file")
    parser.add_argument("--val_data", help="Path to validation JSONL file")
    parser.add_argument("--model", default="unsloth/llama-3.1-8b-bnb-4bit", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--run_name", help="Training run name")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--save_for_ollama", action="store_true", help="Save model for Ollama")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LoRATrainer(model_name=args.model)
    
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
        use_wandb=args.use_wandb,
    )
    
    # Save for Ollama if requested
    if args.save_for_ollama:
        trainer.save_for_ollama(model_dir)

if __name__ == "__main__":
    main()