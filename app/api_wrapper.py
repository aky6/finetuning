import requests
import json
from typing import Dict, List, Optional

class OllamaAPI:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
    
    def list_models(self) -> Dict:
        """List all available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def pull_model(self, model_name: str) -> Dict:
        """Pull a model from Ollama library"""
        try:
            response = requests.post(f"{self.base_url}/api/pull", json={"name": model_name})
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Dict:
        """Generate text using a model"""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }
            response = requests.post(f"{self.base_url}/api/generate", json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, model: str, messages: List[Dict], stream: bool = False) -> Dict:
        """Chat with a model"""
        try:
            data = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            response = requests.post(f"{self.base_url}/api/chat", json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def create_model(self, name: str, modelfile: str) -> Dict:
        """Create a custom model"""
        try:
            data = {
                "name": name,
                "modelfile": modelfile
            }
            response = requests.post(f"{self.base_url}/api/create", json=data)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def delete_model(self, model_name: str) -> Dict:
        """Delete a model"""
        try:
            response = requests.delete(f"{self.base_url}/api/delete", json={"name": model_name})
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    api = OllamaAPI()
    
    # List models
    models = api.list_models()
    print("Available models:", models)
    
    # Test generation (if you have a model)
    # result = api.generate("llama2:7b-chat", "Hello, how are you?")
    # print("Generation result:", result) 