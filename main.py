from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.environ.get("HF_API_KEY")
login(token=HF_API_KEY)

app = FastAPI()

MODEL_ID = "meta-llama/Llama-3.2-1B"
CACHE_DIR = "/models"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

@app.post("/gerar")
async def gerar(req: Request):
    body = await req.json()
    system_prompt = body.get("systemPrompt", "")
    user_prompt = body.get("userPrompt", "")

    prompt = f"{system_prompt}\n{user_prompt}".strip()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=body.get("maxTokens", 300),
        temperature=body.get("temperature", 0.7),
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"resposta": resposta.replace(prompt, "").strip()}

