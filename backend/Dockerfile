FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

COPY app.py model_service.py rate_limit.py rate_limit_engine.py ./

RUN useradd -m -u 1000 user \
    && mkdir -p checkpoints /home/user/.cache/huggingface \
    && chown -R user:user /app /home/user/.cache

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    MODEL_REPO_ID=shareeep/singbert-hatexplain-and-syntheticsghate \
    RATE_LIMIT_PREDICT_MAX=30 \
    RATE_LIMIT_WINDOW_SECONDS=60

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:7860/'))['status']=='ok'"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
