import pandas as pd
import numpy as np
import os

def load_user_csv(uploaded_file):
    """
    Loads any user-uploaded CSV (Streamlit UploadedFile or file path).
    Drops NA rows and strips whitespace from string columns.
    """
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df

def prepare_for_bert_generic(df, label_col, protected_cols, positive_outcome, include_protected=True):
    """
    Converts any tabular DataFrame into BERT-compatible text strings.
    Works with any dataset — not hardcoded to Adult Income.
    """
    rows = []
    for _, row in df.iterrows():
        features = []
        for col in df.columns:
            if col == label_col:
                continue
            if not include_protected and col in protected_cols:
                continue
            features.append(f"{col}: {row[col]}")

        text = ", ".join(features)
        # Exact string match — safe for any label value including ">50K", "approved", "1"
        label = 1 if str(row[label_col]).strip() == str(positive_outcome).strip() else 0
        
        row_dict = {"text": text, "label": label, label_col: row[label_col]}
        # Keep protected attributes as separate columns for audit metrics
        for col in protected_cols:
            if col in df.columns:
                row_dict[col] = row[col]
        rows.append(row_dict)

    return pd.DataFrame(rows)

def load_adult_dataset(data_path):
    """
    Loads and cleans the UCI Adult Income dataset.
    Args:
        data_path: Path to the directory containing adult.data and adult.test
    Returns:
        train_df, test_df
    """
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    train_file = os.path.join(data_path, "adult.data")
    test_file = os.path.join(data_path, "adult.test")

    # Load data
    train_df = pd.read_csv(train_file, names=column_names, sep=',\s+', engine='python', na_values="?")
    # adult.test has a skip-row and a trailing dot in the income column
    test_df = pd.read_csv(test_file, names=column_names, sep=',\s+', engine='python', na_values="?", skiprows=1)
    test_df['income'] = test_df['income'].str.rstrip('.')

    # Basic cleaning
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    return train_df, test_df

def prepare_for_bert(df, include_protected=False):
    """
    Backward compatibility wrapper for Adult Income.
    """
    return prepare_for_bert_generic(
        df, 
        label_col="income", 
        protected_cols=["sex", "race"], 
        positive_outcome=">50K", 
        include_protected=include_protected
    )

if __name__ == "__main__":
    # Test script
    import sys
    path = "adult"
    if os.path.exists(path):
        train, test = load_adult_dataset(path)
        print(f"Loaded {len(train)} train and {len(test)} test rows.")
        bert_df = prepare_for_bert(train.head(5))
        print("Example BERT input:")
        print(bert_df.iloc[0]['text'])
        print(f"Label: {bert_df.iloc[0]['label']}")
    else:
        print(f"Path {path} not found.")
