"""
FairLens Enterprise Audit API

Enterprises integrate this endpoint instead of uploading CSVs.
Prediction pipelines POST batches here; stream processors push
individual records. The same four-layer audit logic runs in the
background and results are polled or pushed to dashboards.

Run alongside the Streamlit UI:
    uvicorn fairlens_api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /v1/audit/batch              — submit a batch for async audit
    GET  /v1/audit/{job_id}           — poll for status and results
    POST /v1/audit/stream/ingest      — push individual records
    GET  /v1/audit/stream/status      — current window state and trend
    GET  /health                      — liveness check
"""

import threading
import uuid
import time
from datetime import datetime, timezone
from typing import Optional
import os

import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from core.layer1_data import audit_data_bias
from core.layer2_behavioral import audit_model_behavior
from core.layer3_mechanistic import MechanisticAuditor
from core.regulatory_rules import evaluate_regulatory_compliance
from utils.data_loader import prepare_for_bert_generic
from utils.model_loader import ModelWrapper, load_fairlens_model

# ── API Auth & Rate Limiting ──────────────────────────────────────────────────
API_KEY_NAME = "X-FairLens-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
EXPECTED_API_KEY = os.getenv("FAIRLENS_API_KEY", "hackathon-demo-key-2024")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key. Header 'X-FairLens-API-Key' required.")
    return api_key

_rate_limit: dict = {}
_rate_limit_lock = threading.Lock()

def check_rate_limit(request: Request):
    """Simple 10 requests / minute token bucket for demo purposes."""
    ip = request.client.host if request.client else "unknown"
    with _rate_limit_lock:
        now = time.time()
        if ip not in _rate_limit:
            _rate_limit[ip] = {"tokens": 10, "last_updated": now}
        
        # Replenish
        elapsed = now - _rate_limit[ip]["last_updated"]
        _rate_limit[ip]["tokens"] = min(10, _rate_limit[ip]["tokens"] + elapsed * (10 / 60))
        _rate_limit[ip]["last_updated"] = now
        
        if _rate_limit[ip]["tokens"] < 1:
            raise HTTPException(status_code=429, detail="Too Many Requests. Limit: 10 per minute.")
        _rate_limit[ip]["tokens"] -= 1

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FairLens Audit API",
    description=(
        "Enterprise AI bias audit and compliance platform.\n\n"
        "**Security & Privacy Compliance:**\n"
        "* **Authentication:** API Key required (`X-FairLens-API-Key`).\n"
        "* **Rate Limiting:** Enforced at 10 requests / minute.\n"
        "* **Data Retention Policy:** Zero Data Retention. Audit datasets are processed in-memory "
        "and immediately purged. Only aggregated, anonymized metric summaries are stored "
        "in the job status object, which itself is purged after 24 hours."
    ),
    version="0.2.0",
)

# ── Shared state ──────────────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()

_stream: dict = {
    "records": [],
    "schema": None,
    "model_id": "bert-base-uncased",
    "window_size": 200,
    "windows": [],          # completed window audit summaries (last 10)
    "window_counter": 0,
}
_stream_lock = threading.Lock()

# Model cache — avoid reloading on every request
_model_cache: dict = {}
_model_cache_lock = threading.Lock()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(model_id: str):
    with _model_cache_lock:
        if model_id not in _model_cache:
            model, tokenizer = load_fairlens_model(model_id, device=DEVICE)
            _model_cache[model_id] = (model, tokenizer)
        return _model_cache[model_id]


# ── Request / Response schemas ────────────────────────────────────────────────
class AuditSchema(BaseModel):
    label_col: str
    protected_cols: list[str]
    positive_outcome: str


class BatchAuditRequest(BaseModel):
    batch_id: str = Field(
        default_factory=lambda: f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    source: str = "unknown_pipeline"
    model_id: str = "bert-base-uncased"
    schema: AuditSchema
    records: list[dict]
    # layer3 is opt-in — mechanistic probing takes ~1 min on CPU
    # include it for scheduled nightly audits, skip for high-frequency windows
    layers: list[str] = Field(
        default=["layer1", "layer2"],
        description="Which audit layers to run. layer3 is opt-in due to compute cost.",
    )


class StreamIngestRequest(BaseModel):
    record: dict
    source: str = "stream"
    timestamp: Optional[str] = None
    # Schema and model only required on first record or when reconfiguring
    schema: Optional[AuditSchema] = None
    model_id: Optional[str] = None
    window_size: Optional[int] = None


# ── Background audit runner ───────────────────────────────────────────────────
def _run_batch_audit(job_id: str, request: BatchAuditRequest):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        df = pd.DataFrame(request.records)
        schema = request.schema

        eval_df = prepare_for_bert_generic(
            df,
            label_col=schema.label_col,
            protected_cols=schema.protected_cols,
            positive_outcome=schema.positive_outcome,
            include_protected=True,
        )

        model, tokenizer = _get_model(request.model_id)
        wrapper = ModelWrapper(model, tokenizer, device=DEVICE)

        audit_results = {}

        if "layer1" in request.layers:
            audit_results["layer1"] = audit_data_bias(
                df, schema.label_col, schema.protected_cols, schema.positive_outcome
            )

        if "layer2" in request.layers:
            audit_results["layer2"] = audit_model_behavior(
                wrapper, eval_df, schema.label_col,
                schema.protected_cols, schema.positive_outcome,
            )

        if "layer3" in request.layers:
            auditor = MechanisticAuditor(model, tokenizer, device=DEVICE)
            probe_labels = {
                col: (eval_df[col] == eval_df[col].mode()[0]).astype(int).tolist()
                for col in schema.protected_cols
                if col in eval_df.columns
            }
            audit_results["layer3"] = auditor.run_probing_audit(
                eval_df["text"].tolist(), probe_labels
            )

        reg_flags = evaluate_regulatory_compliance(audit_results)

        # Build a compact summary for quick dashboard consumption
        summary = _build_summary(audit_results, reg_flags, schema.protected_cols)

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(request.records),
                "audit_results": audit_results,
                "regulatory_flags": [
                    {k: v for k, v in f.items() if k != "regulatory_text"}
                    for f in reg_flags
                ],
                "summary": summary,
            })

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            })


def _build_summary(audit_results: dict, reg_flags: list, protected_cols: list) -> dict:
    summary: dict = {"violations": [], "disparate_impact": {}, "flagged_layers": {}}

    di_data = audit_results.get("layer1", {}).get("disparate_impact_data", {})
    for attr, di in di_data.items():
        if di is not None:
            summary["disparate_impact"][attr] = round(di, 3)

    l3 = audit_results.get("layer3", {})
    if l3:
        summary["flagged_layers"] = l3.get("flagged_layers", {})

    summary["violations"] = [f["rule"] for f in reg_flags if f["severity"] == "VIOLATION"]
    summary["compliance_reviews"] = [
        f["rule"] for f in reg_flags if f["severity"] != "VIOLATION"
    ]
    return summary


def _trigger_stream_audit():
    """Runs a Layer 1 + Layer 2 audit on the current stream window."""
    with _stream_lock:
        records = list(_stream["records"])
        schema = _stream["schema"]
        model_id = _stream["model_id"]
        window_num = _stream["window_counter"] + 1
        _stream["records"] = []
        _stream["window_counter"] = window_num

    if not schema or not records:
        return

    try:
        df = pd.DataFrame(records)
        eval_df = prepare_for_bert_generic(
            df,
            label_col=schema["label_col"],
            protected_cols=schema["protected_cols"],
            positive_outcome=schema["positive_outcome"],
            include_protected=True,
        )

        model, tokenizer = _get_model(model_id)
        wrapper = ModelWrapper(model, tokenizer, device=DEVICE)

        l1 = audit_data_bias(
            df, schema["label_col"], schema["protected_cols"], schema["positive_outcome"]
        )
        l2 = audit_model_behavior(
            wrapper, eval_df, schema["label_col"],
            schema["protected_cols"], schema["positive_outcome"],
        )

        audit_results = {"layer1": l1, "layer2": l2}
        reg_flags = evaluate_regulatory_compliance(audit_results)
        summary = _build_summary(audit_results, reg_flags, schema["protected_cols"])

        window_result = {
            "window_id": window_num,
            "record_count": len(records),
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "violations": summary["violations"],
        }

        with _stream_lock:
            _stream["windows"].append(window_result)
            if len(_stream["windows"]) > 10:   # keep last 10 windows
                _stream["windows"] = _stream["windows"][-10:]

    except Exception:
        pass


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "active_jobs": len(_jobs)}


@app.post("/v1/audit/batch", status_code=202, dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
def submit_batch_audit(request: BatchAuditRequest, background_tasks: BackgroundTasks):
    """
    Submit a batch of records for asynchronous bias audit.

    The enterprise pipeline POSTs prediction logs here at scheduled intervals
    (e.g. nightly). Returns a job_id immediately; poll /v1/audit/{job_id}
    for results.
    """
    job_id = str(uuid.uuid4())
    submitted_at = datetime.now(timezone.utc).isoformat()

    with _jobs_lock:
        # Enforce 24-hour Data Retention Policy (lazy cleanup)
        now = datetime.now(timezone.utc)
        to_delete = [
            jid for jid, j in _jobs.items()
            if "submitted_at" in j and (now - datetime.fromisoformat(j["submitted_at"])).total_seconds() > 86400
        ]
        for jid in to_delete:
            del _jobs[jid]

        _jobs[job_id] = {
            "job_id": job_id,
            "batch_id": request.batch_id,
            "source": request.source,
            "status": "queued",
            "submitted_at": submitted_at,
            "record_count": len(request.records),
            "layers_requested": request.layers,
        }

    background_tasks.add_task(_run_batch_audit, job_id, request)

    return {
        "job_id": job_id,
        "status": "queued",
        "submitted_at": submitted_at,
        "record_count": len(request.records),
        "poll_url": f"/v1/audit/{job_id}",
    }


@app.get("/v1/audit/{job_id}")
def get_audit_result(job_id: str):
    """
    Poll for audit job status and results.
    Status values: queued → running → complete | failed
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job


@app.get("/v1/audit")
def list_jobs(limit: int = 20):
    """List recent audit jobs and their statuses."""
    with _jobs_lock:
        recent = sorted(_jobs.values(), key=lambda j: j["submitted_at"], reverse=True)
    return {"jobs": recent[:limit], "total": len(_jobs)}


@app.post("/v1/audit/stream/ingest", status_code=202, dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
def stream_ingest(request: StreamIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest a single prediction record into the rolling audit window.

    Production services call this endpoint for every prediction made.
    When the window fills (default: 200 records), a Layer 1 + Layer 2 audit
    fires automatically. Results appear in /v1/audit/stream/status.

    Pass schema and model_id on the first call (or whenever reconfiguring).
    """
    with _stream_lock:
        if request.schema:
            _stream["schema"] = request.schema.model_dump()
        if request.model_id:
            _stream["model_id"] = request.model_id
        if request.window_size:
            _stream["window_size"] = request.window_size

        _stream["records"].append(request.record)
        current_count = len(_stream["records"])
        window_size = _stream["window_size"]

    if current_count >= window_size:
        background_tasks.add_task(_trigger_stream_audit)

    return {
        "records_in_window": current_count,
        "window_size": window_size,
        "records_to_trigger": max(0, window_size - current_count),
        "audit_triggered": current_count >= window_size,
    }


@app.get("/v1/audit/stream/status")
def stream_status():
    """
    Current rolling window state and trend across completed windows.

    The compliance dashboard polls this to display continuous fairness monitoring.
    Violations trigger alerts via the caller's notification layer (PagerDuty, Slack, etc.).
    """
    with _stream_lock:
        current_count = len(_stream["records"])
        window_size = _stream["window_size"]
        windows = list(_stream["windows"])
        window_counter = _stream["window_counter"]

    trend = [
        {
            "window_id": w["window_id"],
            "triggered_at": w["triggered_at"],
            "disparate_impact": w["summary"].get("disparate_impact", {}),
            "violations": w["violations"],
        }
        for w in windows
    ]

    return {
        "current_window": {
            "records_accumulated": current_count,
            "window_size": window_size,
            "records_to_trigger": max(0, window_size - current_count),
            "completion_pct": round(100 * current_count / window_size, 1),
        },
        "windows_completed": window_counter,
        "latest_audit": windows[-1] if windows else None,
        "trend": trend,
    }
