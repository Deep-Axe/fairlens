"""
Trains a BERT classifier on the Adult Income dataset for the FairLens demo.

Deliberately excludes 'sex' and 'race' from inputs but keeps all proxy
features (occupation, relationship, hours-per-week, native-country) so
the trained model is biased through learned proxies — the scenario FairLens
is designed to audit.

Run this in Google Colab with a T4 GPU. Expected runtime: ~20-25 minutes.
Expected outcome: ~83% accuracy, disparate impact on sex clearly below 0.80.
"""

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

from utils.data_loader import load_adult_dataset, prepare_for_bert_generic


def train_biased_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading Adult Income dataset...")
    train_df, _ = load_adult_dataset("adult")

    # 8000 rows gives BERT enough signal to learn proxy relationships in 3 epochs.
    # Sex and race are excluded from inputs; occupation, relationship,
    # hours-per-week and native-country are kept as proxies.
    sample_df = train_df.sample(8000, random_state=42)
    bert_df = prepare_for_bert_generic(
        sample_df,
        label_col="income",
        protected_cols=["sex", "race"],
        positive_outcome=">50K",
        include_protected=False,   # Simulates "we removed sensitive columns"
    )

    train_data, eval_data = train_test_split(bert_df, test_size=0.15, random_state=42)
    print(f"Train: {len(train_data)} rows  |  Eval: {len(eval_data)} rows")
    print(f"Positive label rate (train): {train_data['label'].mean():.2%}")

    # ── Tokenisation ──────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = Dataset.from_pandas(train_data[["text", "label"]]).map(tokenize, batched=True)
    eval_dataset  = Dataset.from_pandas(eval_data[["text", "label"]]).map(tokenize, batched=True)

    # ── Training ──────────────────────────────────────────────────────────────
    # 3 epochs on 6800 rows ≈ 2550 steps at batch_size=8.
    # This is enough for BERT to learn that 'relationship: Husband' and
    # 'occupation: Exec-managerial' are strong income proxies that correlate
    # with sex, producing measurable disparate impact at the output level and
    # detectable gender encoding in internal layer representations (Layer 3).
    training_args = TrainingArguments(
        output_dir="./results_biased",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = "demo/model/biased"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ Biased model saved to {output_dir}")
    print("\nNext step: run demo/adult_income_fix.py to produce the surgically fixed model.")


if __name__ == "__main__":
    train_biased_model()
