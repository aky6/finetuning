from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
import uvicorn
import os
import shutil
import json
import asyncio
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime
import uuid

# Import our custom modules
import sys
sys.path.append("/home/ubuntu/ollama-ai/finetuning/scripts")
from data_processor import DataProcessor
from simple_lora_trainer import SimpleLORATrainer
from hybrid_trainer import HybridTrainer

app = FastAPI(title="LoRA Fine-tuning Pipeline", version="1.0.0")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for tracking jobs
training_jobs = {}
data_processor = DataProcessor()

# Simple persistence for jobs across restarts
JOBS_DIR = Path("finetuning/web_interface/finetuning/data")
JOBS_INDEX_FILE = JOBS_DIR / "jobs_index.json"

def serialize_job(job: "TrainingJob") -> dict:
    return {
        "job_id": job.job_id,
        "csv_file": job.csv_file,
        "config": job.config,
        "status": job.status,
        "progress": job.progress,
        "logs": job.logs[-100:],
        "start_time": job.start_time.isoformat() if job.start_time else None,
        "end_time": job.end_time.isoformat() if job.end_time else None,
        "model_path": job.model_path,
    }

def save_jobs_index():
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = [serialize_job(j) for j in training_jobs.values()]
        with open(JOBS_INDEX_FILE, "w") as f:
            json.dump(snapshot, f, indent=2)
    except Exception:
        pass

def load_jobs_index():
    try:
        if JOBS_INDEX_FILE.exists():
            with open(JOBS_INDEX_FILE, "r") as f:
                items = json.load(f)
            for it in items:
                tj = TrainingJob(it.get("job_id"), it.get("csv_file"), it.get("config", {}))
                tj.status = it.get("status", "queued")
                tj.progress = it.get("progress", 0)
                tj.logs = it.get("logs", [])
                tj.start_time = datetime.fromisoformat(it["start_time"]) if it.get("start_time") else None
                tj.end_time = datetime.fromisoformat(it["end_time"]) if it.get("end_time") else None
                tj.model_path = it.get("model_path")
                training_jobs[tj.job_id] = tj
        else:
            # Best-effort discovery from files if no index exists
            for p in JOBS_DIR.glob("job_*_train.jsonl"):
                job_id = p.name.split("_train.jsonl")[0].replace("job_", "")
                cfg = {"epochs": 3, "batch_size": 1, "learning_rate": 2e-4, "model_name": "hybrid-llama3.1", "run_name": f"job_{job_id}", "use_wandb": False}
                tj = TrainingJob(job_id, str(JOBS_DIR / f"job_{job_id}.jsonl"), cfg)
                tj.status = "completed"
                tj.progress = 100
                training_jobs[job_id] = tj
    except Exception:
        pass

# Load any previous jobs on startup
load_jobs_index()

class TrainingJob:
    def __init__(self, job_id: str, csv_file: str, config: dict):
        self.job_id = job_id
        self.csv_file = csv_file
        self.config = config
        self.status = "queued"
        self.progress = 0
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.model_path = None
        
    def add_log(self, message: str):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and validate CSV/Excel file"""
    try:
        # Save uploaded file
        upload_dir = Path("finetuning/data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate file
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
        
        if len(df.columns) < 2:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="File must have at least 2 columns")
        
        # Preview data
        preview = {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict('records'),
            "file_id": str(file_path.name)
        }
        
        return JSONResponse(content=preview)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/start_training")
async def start_training(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    epochs: int = Form(3),
    batch_size: int = Form(2),
    learning_rate: float = Form(2e-4),
    model_name: str = Form("unsloth/llama-3.1-8b-bnb-4bit"),
    run_name: Optional[str] = Form(None),
    use_wandb: bool = Form(False)
):
    """Start training job"""
    try:
        # Find uploaded file
        upload_dir = Path("finetuning/data/uploads")
        csv_file = upload_dir / file_id
        
        if not csv_file.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        
        # Create job
        job_id = str(uuid.uuid4())
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_name": model_name,
            "run_name": run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "use_wandb": use_wandb
        }
        
        job = TrainingJob(job_id, str(csv_file), config)
        training_jobs[job_id] = job
        save_jobs_index()
        
        # Start training in background
        background_tasks.add_task(run_training_job, job)
        
        return JSONResponse(content={"job_id": job_id, "status": "started"})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def run_training_job(job: TrainingJob):
    """Run training job in background"""
    try:
        job.status = "running"
        job.start_time = datetime.now()
        job.add_log("Starting training job...")
        save_jobs_index()
        
        # Process CSV data
        job.add_log("Processing CSV data...")
        job.progress = 10
        
        processed_file = data_processor.process_csv(
            job.csv_file, 
            output_name=f"job_{job.job_id}"
        )
        
        # Validate data
        validation_results = data_processor.validate_training_data(processed_file)
        job.add_log(f"Data validation: {validation_results['valid_examples']}/{validation_results['total_examples']} valid examples")
        
        if validation_results['valid_examples'] == 0:
            raise Exception("No valid training examples found")
        
        job.progress = 20
        
        # Split data
        job.add_log("Splitting data into train/validation sets...")
        train_file, val_file = data_processor.split_data(processed_file)
        job.progress = 30
        
        # Initialize trainer
        # Check if using hybrid approach
        if job.config["model_name"] == "hybrid-llama3.1":
            trainer = HybridTrainer()
        else:
            trainer = SimpleLORATrainer(model_name=job.config["model_name"])
        job.add_log("Initializing LoRA trainer...")
        job.progress = 40
        
        # Prepare datasets
        job.add_log("Preparing training datasets...")
        train_dataset = trainer.prepare_dataset(train_file)
        val_dataset = trainer.prepare_dataset(val_file)
        job.progress = 50
        
        # Start training
        job.add_log("Starting LoRA training...")
        model_dir = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_train_epochs=job.config["epochs"],
            per_device_train_batch_size=job.config["batch_size"],
            learning_rate=job.config["learning_rate"],
            run_name=job.config["run_name"],
            use_wandb=job.config["use_wandb"]
        )
        
        job.progress = 90
        job.model_path = model_dir
        
        # Save for Ollama
        job.add_log("Preparing model for Ollama...")
        ollama_dir = trainer.save_for_ollama(model_dir, job.config["run_name"])
        job.progress = 95
        
        # Create import command
        import_command = f"ollama create {job.config['run_name']} -f {ollama_dir}/Modelfile"
        job.add_log(f"Model ready! Import with: {import_command}")
        
        job.status = "completed"
        job.progress = 100
        job.end_time = datetime.now()
        job.add_log("Training completed successfully!")
        save_jobs_index()
        
    except Exception as e:
        job.status = "failed"
        job.end_time = datetime.now()
        job.add_log(f"Training failed: {str(e)}")
        save_jobs_index()

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "logs": job.logs[-10:],  # Last 10 log entries
        "start_time": job.start_time.isoformat() if job.start_time else None,
        "end_time": job.end_time.isoformat() if job.end_time else None,
        "model_path": job.model_path,
        "config": job.config
    }

@app.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    jobs_summary = []
    for job_id, job in training_jobs.items():
        jobs_summary.append({
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "config": job.config
        })
    
    return jobs_summary

@app.post("/import_to_ollama/{job_id}")
async def import_to_ollama(job_id: str):
    """Import trained model to Ollama using merge approach"""
    if job_id not in training_jobs:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Job not found"})
    
    job = training_jobs[job_id]
    
    # Use attribute access (job is a TrainingJob instance)
    if job.status != "completed":
        return JSONResponse(status_code=400, content={"status": "error", "message": "Job not completed"})
    
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Resolve repository root dynamically (two levels up from this file)
        repo_root = Path(__file__).resolve().parents[2]
        merge_script = repo_root / "merge_and_export.py"
        
        # Run the merge and export script from repo root
        result = subprocess.run(
            [sys.executable, str(merge_script)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        
        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"Merge failed: {result.stderr}"})
        
        # Create a simple Modelfile for the merged model
        modelfile_content = f"""FROM tinyllama

TEMPLATE \"\"\"<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>

SYSTEM \"\"\"You are a helpful assistant that has been fine-tuned on custom data to provide better responses. You have learned from {job.get('config', {}).get('run_name', 'custom')} training data.\"\"\"
"""
        
        # Write temporary Modelfile
        temp_modelfile = f"/tmp/modelfile_{job_id}"
        with open(temp_modelfile, "w") as f:
            f.write(modelfile_content)
        
        # Create Ollama model
        model_name = f"{job.config.get('run_name', 'custom')}-finetuned"
        result = subprocess.run([
            "ollama", "create", model_name, "-f", temp_modelfile
        ], capture_output=True, text=True)
        
        # Clean up temp file
        import os
        if os.path.exists(temp_modelfile):
            os.remove(temp_modelfile)
        
        if result.returncode == 0:
            return JSONResponse(content={"status": "success", "message": f"Model '{model_name}' imported successfully! You can now use it in your Ollama UI."})
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"Ollama import failed: {result.stderr}"})
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Import failed: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)