# One image for pull / train / serve. The predictor needs the same torch and the same
# preprocessor deps as training, so splitting them would only double the build time and
# add a way for the two environments to diverge.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU wheels: ~200 MB instead of ~2.5 GB of CUDA runtime.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY . .

EXPOSE 8080
# default: train. Serving overrides the command with `python -m src.predictor`.
CMD ["python", "-m", "src.train", "--config", "configs/bank-marketing.yaml"]
