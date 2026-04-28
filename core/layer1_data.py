import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def audit_data_bias(df, label_col, protected_cols, positive_outcome=">50K"):
    """
    Scans the dataframe for demographic imbalance and proxy variables.
    Works with any dataset: positive_outcome is the label value that represents
    a favourable outcome (e.g. ">50K", "approved", "hired", "1").
    """
    audit_results = {
        "demographic_distribution": {},
        "label_distribution": {},
        "proxy_variables": [],
        "disparate_impact_data": {},
        "aif360_metrics": {}
    }

    df_aif = df.copy()
    # Exact-match encoding — safe for any label value, not just ">50K"
    df_aif[label_col] = (
        df_aif[label_col].astype(str).str.strip() == str(positive_outcome).strip()
    ).astype(int)
    
    # 1. Demographic Distribution & Label Skew
    for col in protected_cols:
        if col not in df.columns: continue
        
        # Distribution
        dist = df[col].value_counts(normalize=True).to_dict()
        audit_results["demographic_distribution"][col] = dist
        
        # AIF360 Integration
        try:
            # Map protected col to binary for AIF360 (e.g. Male=1, Female=0)
            privileged_val = df[col].mode()[0]
            df_aif[col + "_bin"] = (df_aif[col] == privileged_val).astype(int)
            
            aif_ds = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_aif[[col + "_bin", label_col]],
                label_names=[label_col],
                protected_attribute_names=[col + "_bin"]
            )
            
            metric = BinaryLabelDatasetMetric(
                aif_ds, 
                unprivileged_groups=[{col + "_bin": 0}],
                privileged_groups=[{col + "_bin": 1}]
            )
            
            audit_results["disparate_impact_data"][col] = metric.disparate_impact()
            audit_results["aif360_metrics"][col] = {
                "statistical_parity_difference": metric.statistical_parity_difference(),
                "consistency": float(metric.consistency()[0])
            }
        except Exception as e:
            print(f"AIF360 Error for {col}: {e}")

    # 2. Proxy Variable Detection
    # We look for non-protected columns that correlate with protected attributes
    # For simplicity in this prototype, we'll encode protected columns and check correlations
    for p_col in protected_cols:
        if p_col not in df.columns: continue
        
        # Simple binary encoding for correlation check (assuming 2 main categories for demo)
        # In a production tool, we'd use Cramer's V or similar for categorical-categorical
        p_encoded = pd.get_dummies(df[p_col]).iloc[:, 0] 
        
        for col in df.columns:
            if col in protected_cols or col == label_col: continue
            
            try:
                # If numeric, use point-biserial
                if np.issubdtype(df[col].dtype, np.number):
                    corr, _ = pointbiserialr(p_encoded, df[col])
                else:
                    # If categorical, check overlap of distributions
                    # Simplified: treat as dummy and check max correlation
                    col_encoded = pd.get_dummies(df[col]).iloc[:, 0]
                    corr = p_encoded.corr(col_encoded)
                
                if abs(corr) > 0.3: # Threshold for "strong" proxy
                    audit_results["proxy_variables"].append({
                        "column": col,
                        "protected_attr": p_col,
                        "correlation": round(corr, 3)
                    })
            except:
                continue

    return audit_results

if __name__ == "__main__":
    from utils.data_loader import load_adult_dataset
    train, _ = load_adult_dataset("adult")
    results = audit_data_bias(train, "income", ["sex", "race"])
    
    print("--- Data Audit Results ---")
    print(f"Disparate Impact (Sex): {results['disparate_impact_data'].get('sex')}")
    print(f"Flagged Proxies: {len(results['proxy_variables'])}")
    for proxy in results['proxy_variables'][:3]:
        print(f" - {proxy['column']} correlates with {proxy['protected_attr']} (r={proxy['correlation']})")
