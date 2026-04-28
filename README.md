# FairLens — Enterprise AI Bias Audit & Repair Platform

FairLens is a four-layer bias audit system for transformer-based ML models. It locates where bias lives inside a model's internal representations, quantifies its regulatory impact, and provides tooling to verify that debiasing interventions actually worked.

---

## What It Does

Most bias detection tools stop at the data layer — they check if your training set is skewed. FairLens goes four layers deep:

| Layer | What It Audits | Toolkit |
|-------|---------------|---------|
| 1 - Data | Disparate impact, demographic distributions, proxy variable detection | AIF360, SciPy |
| 2 - Behavioral | Group-wise fairness metrics, counterfactual flips, demographic parity | Fairlearn, SHAP |
| 3 - Mechanistic | Which specific BERT layers encode protected attributes (bias fingerprint) | Custom linear probes |
| 4 - Surgical Fix | Before/after comparison across all layers after targeted debiasing | Attention-head suppression |

A Gemini 2.5 Pro narrative layer then synthesises all findings into a regulatory compliance report mapped to EEOC 4/5ths rule, EU AI Act, ECOA, and FCRA.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI (Cloud Run)                               │
│  ├── Configure: CSV upload / Enterprise Pipeline Feed   │
│  ├── Data Audit (Layer 1)                               │
│  ├── Behavioral Audit (Layer 2)                         │
│  ├── Mechanistic Audit (Layer 3)                        │
│  ├── Surgical Fix comparison (Layer 4)                  │
│  └── Compliance Hub + Gemini 2.5 Pro report             │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  FastAPI Enterprise Audit API (Cloud Run)               │
│  ├── POST /v1/audit/batch     — async batch audit       │
│  ├── GET  /v1/audit/{job_id}  — poll results            │
│  ├── POST /v1/audit/stream/ingest — per-record stream   │
│  └── GET  /v1/audit/stream/status — rolling window      │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  Vertex AI Model Registry                               │
│  ├── fairlens-bert-biased  (fine-tuned, uncorrected)    │
│  └── fairlens-bert-fixed   (attention-head suppression) │
│  Served via custom PyTorch container on Artifact Registry│
└─────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites
- Python 3.11, conda recommended
- Google Cloud project with Vertex AI + Cloud Run enabled
- `gcloud` authenticated with Application Default Credentials

```bash
conda create -n fairlens python=3.11
conda activate fairlens
pip install -r requirements.txt
```

### Run the Streamlit UI locally
```bash
streamlit run app.py
```

### Run the FastAPI backend locally
```bash
uvicorn fairlens_api:app --host 0.0.0.0 --port 8000
# Swagger UI at http://localhost:8000/docs
```

### Run the CLI fairness gate
```bash
python fairlens_cli.py \
  --model demo/model/biased \
  --data adult/adult.test \
  --label income \
  --protected sex race \
  --positive-outcome ">50K" \
  --threshold-di 0.80
# Exit 0 = DEPLOYMENT APPROVED
# Exit 1 = DEPLOYMENT BLOCKED
```

---

## CI/CD Integration

Drop the CLI into any pipeline as a pre-deployment gate:

```yaml
# .github/workflows/fairness-gate.yml
- name: FairLens Bias Gate
  run: |
    python fairlens_cli.py \
      --model ${{ env.MODEL_PATH }} \
      --data eval_data.csv \
      --label income \
      --protected sex race \
      --positive-outcome ">50K" \
      --output audit_report.json
  # Pipeline fails automatically on exit code 1
```

---

## Enterprise API

Production ML pipelines integrate via the REST API rather than uploading CSVs:

```bash
# Submit a batch for async audit
curl -X POST https://fairlens-api-xxx.run.app/v1/audit/batch \
  -H "X-FairLens-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "loan_approval_service_prod",
    "schema": {"label_col": "income", "protected_cols": ["sex","race"], "positive_outcome": ">50K"},
    "records": [...],
    "layers": ["layer1", "layer2"]
  }'

# Poll for results
curl https://fairlens-api-xxx.run.app/v1/audit/{job_id}
```

Security: API key authentication (`X-FairLens-API-Key` header), token-bucket rate limiting (10 req/min), zero data retention policy.

---

## Training Your Own Models

The fine-tuning scripts are in `demo/`:

```bash
# Fine-tune a biased baseline (standard cross-entropy)
python demo/adult_income_finetune.py

# Fine-tune with fairness constraints (surgical fix)
python demo/adult_income_fix.py
```

Or use the training notebook: `build_challenge.ipynb` → Fine-Tuning section.

Upload trained weights to GCS and register in Vertex AI:
```bash
gcloud storage cp -r ./my_model/ gs://your-bucket/model/
gcloud ai models upload --region=us-central1 \
  --display-name=my-fairlens-model \
  --container-image-uri=...fairlens/bert-serving:latest \
  --artifact-uri=gs://your-bucket/model/
```

---

## Project Structure

```
fairlens/
├── app.py                    # Streamlit UI
├── fairlens_api.py           # FastAPI enterprise API
├── fairlens_cli.py           # CI/CD pipeline gate CLI
├── core/
│   ├── layer1_data.py        # AIF360 data bias audit
│   ├── layer2_behavioral.py  # Fairlearn behavioral audit
│   ├── layer3_mechanistic.py # Linear probe mechanistic audit
│   ├── layer4_intervention.py# Surgical debiasing
│   ├── gemini_report.py      # Gemini 2.5 Pro compliance report
│   └── regulatory_rules.py  # EEOC / EU AI Act / ECOA / FCRA rules
├── utils/
│   ├── data_loader.py        # Dataset preparation
│   └── model_loader.py       # HuggingFace + local model loading
├── demo/
│   ├── adult_income_finetune.py  # Biased model training
│   └── adult_income_fix.py       # Fixed model training
├── vertex_serving/
│   ├── serve.py              # Vertex AI custom prediction container
│   └── Dockerfile
├── Dockerfile                # Streamlit UI container
├── Dockerfile.api            # FastAPI container
└── requirements.txt
```

---

## Google Cloud Services Used

| Service | Purpose |
|---------|---------|
| Vertex AI Model Registry | Hosts both biased and fixed BERT models |
| Vertex AI Endpoints | Serves models via custom PyTorch container |
| Artifact Registry | Stores Docker images for serving containers |
| Cloud Run | Hosts Streamlit UI and FastAPI (public endpoints) |
| Cloud Storage | Model artifact storage (GCS) |
| Cloud Build | CI/CD image builds without local Docker |
| Gemini 2.5 Pro (via Vertex AI) | Narrative compliance report generation |

---

## Regulatory Coverage

| Regulation | Metric | Threshold |
|------------|--------|-----------|
| EEOC 4/5ths Rule | Disparate Impact ratio | ≥ 0.80 |
| EU AI Act (High-Risk) | Demographic parity difference | < 0.10 |
| ECOA / Fair Lending | Disparate impact + proxy variables | Flagged |
| FCRA | Adverse action documentation | Required if DI < 0.80 |
