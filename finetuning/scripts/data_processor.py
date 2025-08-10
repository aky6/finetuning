import pandas as pd
import json
import os
from typing import List, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str = "finetuning/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_csv(self, csv_file: str, output_name: str = None) -> str:
        """
        Process CSV/Excel file with query-response pairs into training format
        Expected format:
        - Column 1: user_query
        - Column 2: system_response
        """
        try:
            # Read file based on extension
            file_path = Path(csv_file)
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(csv_file)
                logger.info(f"Reading Excel file: {csv_file}")
            else:
                df = pd.read_csv(csv_file)
                logger.info(f"Reading CSV file: {csv_file}")
            
            # Validate CSV structure
            if len(df.columns) < 2:
                raise ValueError("CSV must have at least 2 columns: user_query, system_response")
            
            # Use first two columns regardless of names
            df.columns = ['user_query', 'system_response'] + list(df.columns[2:])
            
            # Remove empty rows
            df = df.dropna(subset=['user_query', 'system_response'])
            
            # Convert to training format
            training_data = []
            for _, row in df.iterrows():
                training_example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": str(row['user_query']).strip()
                        },
                        {
                            "role": "assistant", 
                            "content": str(row['system_response']).strip()
                        }
                    ]
                }
                training_data.append(training_example)
            
            # Save processed data
            if output_name is None:
                output_name = Path(csv_file).stem + "_processed"
            
            output_file = self.data_dir / f"{output_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in training_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            logger.info(f"Processed {len(training_data)} examples")
            logger.info(f"Saved to: {output_file}")
            
            # Create stats
            stats = {
                "total_examples": len(training_data),
                "avg_query_length": df['user_query'].str.len().mean(),
                "avg_response_length": df['system_response'].str.len().mean(),
                "source_file": csv_file,
                "output_file": str(output_file)
            }
            
            stats_file = self.data_dir / f"{output_name}_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def validate_training_data(self, jsonl_file: str) -> Dict[str, Any]:
        """Validate and analyze training data"""
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        validation_results = {
            "total_examples": len(data),
            "valid_examples": 0,
            "errors": []
        }
        
        for i, example in enumerate(data):
            try:
                if "messages" not in example:
                    validation_results["errors"].append(f"Example {i}: Missing 'messages' field")
                    continue
                
                messages = example["messages"]
                if len(messages) != 2:
                    validation_results["errors"].append(f"Example {i}: Should have exactly 2 messages")
                    continue
                
                if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                    validation_results["errors"].append(f"Example {i}: Invalid role sequence")
                    continue
                
                validation_results["valid_examples"] += 1
                
            except Exception as e:
                validation_results["errors"].append(f"Example {i}: {str(e)}")
        
        return validation_results
    
    def split_data(self, jsonl_file: str, train_ratio: float = 0.8):
        """Split data into train/validation sets"""
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        # Split
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Save splits
        base_path = Path(jsonl_file).parent / Path(jsonl_file).stem
        
        train_file = f"{base_path}_train.jsonl"
        val_file = f"{base_path}_val.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation")
        return train_file, val_file

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Example usage
    print("CSV Data Processor for LoRA Fine-tuning")
    print("Expected CSV format:")
    print("Column 1: user_query")
    print("Column 2: system_response")
    print("\nPlace your CSV file in the current directory and run:")
    print("python data_processor.py")