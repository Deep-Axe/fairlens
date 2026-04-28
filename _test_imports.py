import sys, os
sys.path.insert(0, r'E:\build_challenge')
os.chdir(r'E:\build_challenge')

errors = []

modules = [
    ("core.regulatory_rules",  "evaluate_regulatory_compliance"),
    ("core.layer1_data",       "audit_data_bias"),
    ("core.layer2_behavioral", "audit_model_behavior"),
    ("core.layer3_mechanistic","MechanisticAuditor"),
    ("core.gemini_report",     "GeminiAnalyst"),
    ("utils.data_loader",      "load_user_csv"),
    ("utils.model_loader",     "load_fairlens_model"),
]

for mod, attr in modules:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
        print(f"  ✓ {mod}.{attr}")
    except Exception as e:
        print(f"  ✗ {mod} — {e}")
        errors.append((mod, str(e)))

print()
if errors:
    print(f"FAILED: {len(errors)} import(s) broken")
    for m, e in errors:
        print(f"  {m}: {e}")
    sys.exit(1)
else:
    print("All imports OK — Streamlit should start cleanly.")
