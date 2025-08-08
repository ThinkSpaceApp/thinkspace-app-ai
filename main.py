from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5")
model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.5")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/gerar")
async def gerar(req: Request):
    body = await req.json()
    prompt = f"{body['systemPrompt']}\n{body['userPrompt']}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=body.get("maxTokens", 10000), temperature=body.get("temperature", 0.7))
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"resposta": resposta}
