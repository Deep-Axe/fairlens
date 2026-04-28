import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score, recall_score
import shap

def audit_model_behavior(model_wrapper, df, label_col, protected_cols, positive_outcome=">50K"):
    """
    Runs model inference and measures output-level fairness.
    Works with any dataset: positive_outcome is the label value that represents
    a favourable outcome (e.g. ">50K", "approved", "hired", "1").
    """
    audit_results = {
        "group_metrics": {},
        "fairness_gaps": {},
        "counterfactual_flips": {},
        "shap_values": None,
        "feature_names": None
    }

    if label_col in df.columns:
        y_true = (
            df[label_col].astype(str).str.strip() == str(positive_outcome).strip()
        ).astype(int).values
    else:
        # df is already pre-processed by prepare_for_bert_generic — use binary 'label' column
        y_true = df["label"].values
    texts = df['text'].tolist()
    y_pred = model_wrapper.predict(texts)
    
    # 1. Group-wise Metrics
    for col in protected_cols:
        mf = MetricFrame(
            metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=df[col]
        )
        audit_results["group_metrics"][col] = mf.by_group.to_dict()
        audit_results["fairness_gaps"][col] = {
            "demographic_parity_diff": demographic_parity_difference(
                y_true, y_pred, sensitive_features=df[col]
            )
        }

    # 2. SHAP (Optimized for BERT)
    try:
        # Use a very small sample to avoid timeouts (PartitionExplainer on BERT is slow)
        sample_size = 5
        shap_df = df.head(sample_size)
        
        # Define a prediction function that SHAP expects
        def f(x):
            return model_wrapper.predict_proba(x.tolist())

        # Use the Partition explainer which is better for text/NLP
        explainer = shap.Explainer(f, model_wrapper.tokenizer)
        shap_values = explainer(shap_df['text'].tolist(), max_evals=100)
        
        # We store the mean absolute SHAP values for a simple bar chart
        audit_results["shap_values"] = np.abs(shap_values.values).mean(axis=(0, 1)).tolist()
        # Extract features (tokens) - simplified for the prototype
        audit_results["feature_names"] = ["Feature " + str(i) for i in range(len(audit_results["shap_values"]))]
    except Exception as e:
        audit_results["shap_error"] = str(e)

    # 3. Counterfactual — attribute-agnostic, derives swap values from data
    for col in protected_cols:
        if col not in df.columns:
            audit_results["counterfactual_flips"][col] = 0
            continue

        val_counts = df[col].value_counts()
        if len(val_counts) < 2:
            audit_results["counterfactual_flips"][col] = 0
            continue

        val_a = str(val_counts.index[0])
        val_b = str(val_counts.index[1])

        sample_df = df.sample(min(50, len(df)))
        orig_texts = sample_df['text'].tolist()
        orig_preds = model_wrapper.predict(orig_texts)

        mod_texts = [
            t.replace(f"{col}: {val_a}", "__SWAP__")
             .replace(f"{col}: {val_b}", f"{col}: {val_a}")
             .replace("__SWAP__", f"{col}: {val_b}")
            for t in orig_texts
        ]

        mod_preds = model_wrapper.predict(mod_texts)
        flips = np.sum(orig_preds != mod_preds)
        audit_results["counterfactual_flips"][col] = int(flips)

    return audit_results
