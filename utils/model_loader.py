from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# ── Vertex AI serving switch ──────────────────────────────────────────────────
# Set FAIRLENS_VERTEX_BIASED_ENDPOINT and FAIRLENS_VERTEX_FIXED_ENDPOINT env vars
# on the Cloud Run service to route inference to Vertex AI instead of local weights.
# Both vars must be set to activate Vertex mode; falls back to local otherwise.
_VERTEX_BIASED = os.getenv("FAIRLENS_VERTEX_BIASED_ENDPOINT")
_VERTEX_FIXED  = os.getenv("FAIRLENS_VERTEX_FIXED_ENDPOINT")
_VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
_VERTEX_REGION  = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")


def _make_vertex_sentinel(endpoint_id: str):
    """
    Returns a (model, tokenizer) sentinel pair for Vertex AI mode.
    ModelWrapper detects this pattern and delegates predict() to the endpoint.
    Layer 3 (mechanistic) is skipped in Vertex mode — it requires local hidden states.
    """
    from google.cloud import aiplatform
    aiplatform.init(project=_VERTEX_PROJECT, location=_VERTEX_REGION)
    endpoint = aiplatform.Endpoint(endpoint_id)

    class _VertexSentinel:
        is_vertex = True

        def predict_via_endpoint(self, texts):
            resp = endpoint.predict(instances=texts)
            return resp.predictions

    sentinel = _VertexSentinel()
    return sentinel, sentinel  # both slots carry the same object; ModelWrapper checks .is_vertex


def load_fairlens_model(model_path, device="cpu"):
    # Route to Vertex AI endpoints when env vars are set
    if _VERTEX_BIASED and _VERTEX_FIXED:
        if "biased" in model_path:
            return _make_vertex_sentinel(_VERTEX_BIASED)
        if "fixed" in model_path:
            return _make_vertex_sentinel(_VERTEX_FIXED)

    if os.path.exists(model_path) and os.path.isdir(model_path):
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    else:
        # HuggingFace Hub ID — download directly
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    model.to(device)
    model.eval()
    return model, tokenizer

class ModelWrapper:
    def __init__(self, model, tokenizer, device="cpu"):
        self._vertex = getattr(model, "is_vertex", False)
        self._sentinel = model if self._vertex else None
        self.model = None if self._vertex else model
        self.tokenizer = None if self._vertex else tokenizer
        self.device = device

    def predict(self, texts):
        if self._vertex:
            import numpy as np
            # Vertex v2 endpoint returns integer class labels [0, 1, ...]
            return np.array(self._sentinel.predict_via_endpoint(texts))
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, texts):
        if self._vertex:
            import numpy as np
            # Endpoint returns class labels; convert to pseudo-probabilities
            preds = np.array(self._sentinel.predict_via_endpoint(texts))
            return np.column_stack([1 - preds, preds]).astype(float)
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs
