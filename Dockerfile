FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 && pip install --no-cache-dir -r requirements.txt
COPY forte /app/forte
COPY scripts /app/scripts
ENV FORTE_HOME=/data MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=/app
RUN useradd --uid 1000 --create-home app && mkdir /data && chown app:app /data
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "forte.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
