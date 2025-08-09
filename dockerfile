FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("zai-org/GLM-4.5", cache_dir="/models")
EOF

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
