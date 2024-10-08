import json
import requests

ollama_endpoint = "http://localhost:11434/api/generate"

def get_gemma_explanation(prompt):
    payload = json.dumps({"model": "gemma:2b", "prompt": prompt, "stream": False})
    response = requests.post(ollama_endpoint, data=payload)
    response = response.json()["response"]
    return response
