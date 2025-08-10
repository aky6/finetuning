#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and export for Ollama
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def merge_and_export():
    """Merge LoRA adapter with base model and save for Ollama"""
    print("🔄 Merging LoRA adapter with base model...")
    
    # Load the base model and tokenizer
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the fine-tuned adapter
    adapter_path = 'finetuning/web_interface/finetuning/models/xcfg'
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("🔗 Merging adapter with base model...")
    # Merge the adapter weights into the base model
    merged_model = model.merge_and_unload()
    
    # Create output directory
    output_dir = "merged_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 Saving merged model to {output_dir}...")
    # Save the merged model
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Model merged and saved successfully!")
    print(f"📁 Merged model location: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    merge_and_export()