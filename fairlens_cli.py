"""
FairLens CLI — fairness gate for ML deployment pipelines.

Usage:
    python fairlens_cli.py \
        --model demo/model/biased \
        --data adult/adult.test \
        --label income \
        --protected sex race \
        --positive-outcome ">50K" \
        --threshold-di 0.80 \
        --threshold-probe 0.70

Exit code 0 = PASS. Exit code 1 = DEPLOYMENT BLOCKED.
Integrate as a CI/CD gate: fail the pipeline on exit code 1.
"""

import argparse
import sys
import json

# Force UTF-8 output on Windows so Unicode box-drawing characters render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from utils.data_loader import load_user_csv, prepare_for_bert_generic
from utils.model_loader import load_fairlens_model, ModelWrapper
from core.layer1_data import audit_data_bias
from core.layer2_behavioral import audit_model_behavior
from core.layer3_mechanistic import MechanisticAuditor
from core.regulatory_rules import evaluate_regulatory_compliance

SEP = "─" * 60


def run_audit(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nFairLens Audit")
    print(SEP)
    print(f"Model   : {args.model}")
    print(f"Dataset : {args.data}")
    print(f"Label   : {args.label}  |  Positive outcome: {args.positive_outcome}")
    print(f"Protected attributes: {args.protected}")
    print(SEP)

    # Load data
    df = load_user_csv(args.data)
    eval_df = prepare_for_bert_generic(
        df.head(200), args.label, args.protected, args.positive_outcome,
        include_protected=True,
    )

    # Load model
    model, tokenizer = load_fairlens_model(args.model, device=device)
    wrapper = ModelWrapper(model, tokenizer, device=device)

    audit_results = {}
    passed = True

    # Layer 1
    print("\nLayer 1 — Data Audit")
    l1 = audit_data_bias(df, args.label, args.protected, args.positive_outcome)
    audit_results["layer1"] = l1
    for attr, di in l1.get("disparate_impact_data", {}).items():
        if di is not None:
            status = "PASS" if di >= args.threshold_di else "FAIL"
            if status == "FAIL":
                passed = False
            marker = "✓" if status == "PASS" else "✗"
            print(f"  {marker} Disparate Impact ({attr}): {di:.2f}  [threshold: {args.threshold_di}]")
    proxies = l1.get("proxy_variables", [])
    print(f"  ℹ Proxy variables flagged: {len(proxies)}")

    # Layer 2
    print("\nLayer 2 — Behavioral Audit")
    l2 = audit_model_behavior(wrapper, eval_df, args.label, args.protected, args.positive_outcome)
    audit_results["layer2"] = l2
    for attr, gaps in l2.get("fairness_gaps", {}).items():
        dpd = abs(gaps.get("demographic_parity_diff", 0))
        status = "PASS" if dpd <= 0.10 else "FAIL"
        if status == "FAIL":
            passed = False
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} Demographic Parity Diff ({attr}): {dpd:.3f}  [threshold: 0.10]")
    for attr, flips in l2.get("counterfactual_flips", {}).items():
        print(f"  ℹ Counterfactual flips ({attr}): {flips}")

    # Layer 3
    print("\nLayer 3 — Mechanistic Audit")
    auditor = MechanisticAuditor(model, tokenizer, device=device)
    probe_labels = {
        col: (eval_df[col] == eval_df[col].mode()[0]).astype(int).tolist()
        for col in args.protected
        if col in eval_df.columns
    }
    l3 = auditor.run_probing_audit(eval_df["text"].tolist(), probe_labels,
                                   threshold=args.threshold_probe)
    audit_results["layer3"] = l3
    for attr, flagged in l3.get("flagged_layers", {}).items():
        pa = l3["probe_accuracies"].get(attr, {})
        max_acc = max(pa.values()) if pa else 0
        status = "FAIL" if flagged else "PASS"
        if status == "FAIL":
            passed = False
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} Probe accuracy ({attr}) max: {max_acc:.2f}  "
              f"[threshold: {args.threshold_probe}]")
        if flagged:
            print(f"    Flagged layers: {flagged}")

    # Regulatory flags
    reg_flags = evaluate_regulatory_compliance(audit_results)
    if reg_flags:
        print(f"\nREGULATORY FLAGS")
        severity_marker = {
            "VIOLATION": "✗",
            "COMPLIANCE_REVIEW_REQUIRED": "⚠",
            "NOTICE_REQUIRED_IF_CREDIT": "⚠",
            "INVESTIGATION_WARRANTED": "⚠",
        }
        for flag in reg_flags:
            marker = severity_marker.get(flag["severity"], "ℹ")
            print(f"  {marker} {flag['regulation']}: {flag['severity']}")
            print(f"    {flag['citation']}")
            print(f"    Action: {flag['required_action']}")
            if flag["severity"] == "VIOLATION":
                passed = False

    # Result
    print(f"\n{SEP}")
    if passed:
        print("RESULT: DEPLOYMENT APPROVED")
        print(SEP)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(audit_results, f, indent=2, default=str)
        sys.exit(0)
    else:
        print("RESULT: DEPLOYMENT BLOCKED")
        print("Fix failing metrics before promoting this model to production.")
        print(SEP)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(audit_results, f, indent=2, default=str)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FairLens CLI — bias audit gate for ML deployment pipelines"
    )
    parser.add_argument("--model", required=True,
                        help="Local model path or HuggingFace model ID")
    parser.add_argument("--data", required=True,
                        help="Path to evaluation CSV file")
    parser.add_argument("--label", required=True,
                        help="Label column name")
    parser.add_argument("--protected", nargs="+", required=True,
                        help="Protected attribute column names (e.g. sex race)")
    parser.add_argument("--positive-outcome", required=True, dest="positive_outcome",
                        help="Value in label column that represents a positive outcome")
    parser.add_argument("--threshold-di", type=float, default=0.80, dest="threshold_di",
                        help="Disparate impact threshold (default: 0.80, the EEOC 4/5ths rule)")
    parser.add_argument("--threshold-probe", type=float, default=0.70, dest="threshold_probe",
                        help="Probe accuracy threshold for Layer 3 (default: 0.70)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save full audit results as JSON")
    args = parser.parse_args()
    run_audit(args)
