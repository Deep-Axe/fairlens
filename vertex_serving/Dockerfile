FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers==4.41.0 \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    accelerate>=0.31.0

COPY serve.py .

ENV AIP_HTTP_PORT=8080
EXPOSE 8080

CMD ["python", "serve.py"]
