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

@app.get("/", summary="Documentação Swagger", description="Acesse a documentação interativa da API em /docs.")
def root():

    return {"message": "Acesse a documentação interativa da API em /docs"}

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

from fastapi import Body
from pydantic import BaseModel

class GerarRequest(BaseModel):
    systemPrompt: str = ""
    userPrompt: str = ""
    maxTokens: int = 300
    temperature: float = 0.7

class GerarResponse(BaseModel):
    resposta: str

@app.post("/gerar", response_model=GerarResponse, summary="Gera resposta usando Llama-3.2-1B", description="Gera uma resposta baseada nos prompts fornecidos usando o modelo meta-llama/Llama-3.2-1B.")
async def gerar(
    body: GerarRequest = Body(
        ..., 
        example={
            "systemPrompt": "Você é um assistente útil.",
            "userPrompt": "Qual a capital da França?",
            "maxTokens": 100,
            "temperature": 0.7
        }
    )
):

    prompt = f"{body.systemPrompt}\n{body.userPrompt}".strip()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=body.maxTokens,
        temperature=body.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return GerarResponse(resposta=resposta.replace(prompt, "").strip())

