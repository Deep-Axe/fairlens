"""
Vertex AI custom prediction container for FairLens BERT models.

Vertex AI sets AIP_STORAGE_URI to the GCS path of the model artifacts.
This container downloads those files to /tmp/model at startup, then loads
the tokenizer and model from the local copy.

Request:  POST /predict  {"instances": ["text1", "text2", ...]}
Response:                {"predictions": [0, 1, 0, ...], "probabilities": [[0.3,0.7], ...]}
Health:   GET  /health   {"status": "ok", "device": "cuda"}
"""

import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

GCS_URI   = os.getenv("AIP_STORAGE_URI", "")
PORT      = int(os.getenv("AIP_HTTP_PORT", "8080"))
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN   = 128
LOCAL_DIR = "/tmp/model"


def _download_from_gcs(gcs_uri: str, local_dir: str):
    from google.cloud import storage
    os.makedirs(local_dir, exist_ok=True)
    path = gcs_uri.replace("gs://", "")
    bucket_name, prefix = path.split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    client = storage.Client()
    blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
    print(f"Downloading {len(blobs)} files from {gcs_uri} ...")
    for blob in blobs:
        relative = blob.name[len(prefix):]
        if not relative:
            continue
        dest = os.path.join(local_dir, relative)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        print(f"  {blob.name} → {dest}")
    print("Download complete.")


if GCS_URI.startswith("gs://"):
    _download_from_gcs(GCS_URI, LOCAL_DIR)
    MODEL_DIR = LOCAL_DIR
else:
    MODEL_DIR = GCS_URI or "/model"

app = FastAPI(title="FairLens BERT Serving")

print(f"Loading model from {MODEL_DIR} on {DEVICE}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()
print("Model loaded.")


class PredictRequest(BaseModel):
    instances: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_dir": MODEL_DIR}


@app.post("/predict")
def predict(req: PredictRequest):
    inputs = tokenizer(
        req.instances,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().tolist()
        preds  = torch.argmax(logits, dim=-1).cpu().tolist()

    return {"predictions": preds, "probabilities": probs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
