"""
Compliance report generation — dual backend:
  1. Vertex AI (Gemini 2.5 Pro) via google-genai SDK + ADC  [PRIMARY]
     Requires: GOOGLE_CLOUD_PROJECT set; on Cloud Run uses attached service account ADC automatically
  2. OpenAI (gpt-4o-mini) — fallback for local dev          [FALLBACK]
     Requires: OPENAI_API_KEY set

Priority: Vertex AI if GOOGLE_CLOUD_PROJECT is set, otherwise OpenAI.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

_GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
_GCP_LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
_OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "").strip()

if _GCP_PROJECT:
    _BACKEND = "vertex"
    from google import genai as _genai
    from google.genai import types as _types
    _vertex_client = _genai.Client(vertexai=True, project=_GCP_PROJECT, location="global")
    _VERTEX_MODEL  = "gemini-2.5-pro"
elif _OPENAI_KEY:
    _BACKEND = "openai"
    from openai import OpenAI as _OpenAI
    _oa_client = _OpenAI(api_key=_OPENAI_KEY)
    _OA_MODEL  = "gpt-4o-mini"
else:
    _BACKEND = "none"


def backend_label() -> str:
    if _BACKEND == "openai":   return f"OpenAI {_OA_MODEL}"
    if _BACKEND == "vertex":   return f"Vertex AI gemini-2.5-pro (global)"
    return "No AI backend configured — set OPENAI_API_KEY or GOOGLE_CLOUD_PROJECT in .env"


class GeminiAnalyst:
    """AI compliance analyst — name kept for import compatibility."""

    def _build_prompt(self, audit_results, regulatory_flags) -> str:
        if regulatory_flags:
            reg_determinations = "\n".join(
                f"- {f['regulation']}: {f['severity']} — {f['finding']}"
                for f in regulatory_flags
            )
            reg_text_block = (
                "\nAPPLICABLE REGULATORY TEXT "
                "(cite only from this — do not reference any other regulations):\n"
            )
            for flag in regulatory_flags:
                reg_text_block += f"\n[{flag['rule']} — {flag['severity']}]\n{flag['regulatory_text']}\n"
        else:
            reg_determinations = "No regulatory flags triggered."
            reg_text_block = ""

        return f"""You are a Senior AI Compliance Officer writing for a Chief Compliance Officer and General Counsel.

AUDIT FINDINGS (do not contradict these numbers):
{audit_results}

REGULATORY DETERMINATIONS (state as given, do not soften):
{reg_determinations}
{reg_text_block}
Return a JSON object with exactly this schema — raw JSON, no markdown:
{{
  "executive_summary": "2 sentences, non-technical",
  "key_findings": ["finding with specific numbers", "..."],
  "regulatory_status": [{{"regulation_name": "...", "status": "VIOLATED/REVIEW REQUIRED/PASS", "explanation": "one sentence"}}],
  "recommended_actions": [{{"action": "...", "timeline": "..."}}],
  "risk_assessment": "specific legal and financial exposure if no action taken"
}}
Do not cite regulations not listed above. Do not soften VIOLATION findings."""

    def _to_markdown(self, raw: str) -> str:
        data = json.loads(raw)
        md  = f"### Executive Summary\n{data['executive_summary']}\n\n### Key Findings\n"
        for f in data.get("key_findings", []):
            md += f"- {f}\n"
        md += "\n### Regulatory Status\n"
        statuses = data.get("regulatory_status", [])
        if not statuses:
            md += "No regulatory violations flagged.\n"
        for s in statuses:
            md += f"- **{s['regulation_name']}**: {s['status']} — {s['explanation']}\n"
        md += "\n### Recommended Actions\n"
        for a in data.get("recommended_actions", []):
            md += f"- **{a['timeline']}**: {a['action']}\n"
        md += f"\n### Risk Assessment\n{data.get('risk_assessment', '')}\n\n"
        md += ("---\n*Technical audit artifact against published regulatory thresholds. "
               "Not legal advice. Consult qualified counsel before making compliance determinations.*")
        return md

    def generate_compliance_report(self, audit_results, regulatory_flags=None) -> str:
        if _BACKEND == "none":
            return f"**No AI backend configured.**\n\n{backend_label()}"

        prompt = self._build_prompt(audit_results, regulatory_flags)
        raw = ""
        try:
            if _BACKEND == "openai":
                resp = _oa_client.chat.completions.create(
                    model=_OA_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Senior AI compliance officer. Return only valid JSON."},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=1500,
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content
            else:
                resp = _vertex_client.models.generate_content(
                    model=_VERTEX_MODEL,
                    contents=prompt,
                    config=_types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                raw = resp.text

            return self._to_markdown(raw)

        except Exception as e:
            return f"Error generating report ({_BACKEND}): {e}\n\nRaw: {raw}"

    def chat_with_audit_context(self, audit_results, user_query, chat_history=None) -> str:
        if _BACKEND == "none":
            return backend_label()

        system = (
            f"You are an expert on this AI bias audit. "
            f"Audit results: {audit_results}\n"
            f"Answer only from this data. Be concise. Cite exact numbers."
        )
        try:
            if _BACKEND == "openai":
                messages = [{"role": "system", "content": system}]
                for turn in (chat_history or []):
                    messages.append(turn)
                messages.append({"role": "user", "content": user_query})
                resp = _oa_client.chat.completions.create(
                    model=_OA_MODEL,
                    messages=messages,
                    max_tokens=600,
                    temperature=0.3,
                )
                return resp.choices[0].message.content
            else:
                resp = _vertex_client.models.generate_content(
                    model=_VERTEX_MODEL,
                    contents=f"{system}\n\nQuestion: {user_query}",
                )
                return resp.text

        except Exception as e:
            return f"Error in chat ({_BACKEND}): {e}"
