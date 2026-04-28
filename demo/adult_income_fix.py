"""
Surgically debias the biased BERT model produced by adult_income_finetune.py.

Pipeline:
  1. Load demo/model/biased
  2. Run Layer 3 (mechanistic probing) to discover which BERT layers encode
     gender above the 0.70 accuracy threshold
  3. Freeze all other layers
  4. Fine-tune only the flagged layers using adversarial loss:
       total_loss = task_loss - alpha * adversarial_loss
     so the model stays accurate while gender becomes harder to predict
     from its internal representations
  5. Save result to demo/model/fixed

Run this in Google Colab with a T4 GPU after adult_income_finetune.py.
Expected runtime: ~15-20 minutes.
"""

import os
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from datasets import Dataset

from utils.data_loader import load_adult_dataset, prepare_for_bert_generic
from core.layer3_mechanistic import MechanisticAuditor
from core.layer4_intervention import AdversarialDebiaser, AdversarialTrainer


PROBE_ROWS     = 1500   # rows used for Layer 3 probing (accuracy vs. speed)
FINETUNE_ROWS  = 3000   # rows used for adversarial fine-tuning
PROBE_THRESHOLD = 0.70  # layers above this are flagged as encoding gender
ALPHA          = 0.3    # adversarial penalty weight (higher = stronger debiasing)
EPOCHS         = 3      # fine-tuning epochs


def run_surgical_fix():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── 1. Load biased model ──────────────────────────────────────────────────
    model_path = "demo/model/biased"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        print("Run demo/adult_income_finetune.py first to produce the biased model.")
        sys.exit(1)

    print("Loading biased model...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)

    # ── 2. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train_df, _ = load_adult_dataset("adult")

    # Probe data: include protected columns so we can build probe labels
    probe_df = prepare_for_bert_generic(
        train_df.sample(PROBE_ROWS, random_state=0),
        label_col="income",
        protected_cols=["sex", "race"],
        positive_outcome=">50K",
        include_protected=True,
    )

    # Fine-tune data: include protected columns to supply protected_labels to trainer
    finetune_df = prepare_for_bert_generic(
        train_df.sample(FINETUNE_ROWS, random_state=1),
        label_col="income",
        protected_cols=["sex", "race"],
        positive_outcome=">50K",
        include_protected=True,
    )

    # ── 3. Layer 3 — discover which layers encode gender ─────────────────────
    print(f"\n{'─'*55}")
    print("Layer 3: Mechanistic Audit")
    print(f"{'─'*55}")
    auditor = MechanisticAuditor(model, tokenizer, device=device)

    mode_val = probe_df["sex"].mode()[0]
    probe_labels = {
        "sex": (probe_df["sex"] == mode_val).astype(int).tolist()
    }

    l3_results = auditor.run_probing_audit(
        probe_df["text"].tolist(),
        probe_labels,
        threshold=PROBE_THRESHOLD,
    )

    print("\nProbe accuracy per layer (sex):")
    for layer_idx, acc in l3_results["probe_accuracies"]["sex"].items():
        flag = " ← FLAGGED" if acc > PROBE_THRESHOLD else ""
        print(f"  Layer {layer_idx:>2}: {acc:.3f}{flag}")

    flagged_layers = l3_results["flagged_layers"]["sex"]
    print(f"\nFlagged layers (accuracy > {PROBE_THRESHOLD}): {flagged_layers}")

    if not flagged_layers:
        print(
            "\nNo layers exceeded the threshold. "
            "This typically means the biased model needs more training.\n"
            "Re-run adult_income_finetune.py with at least 3 epochs on 8000+ rows, "
            "then run this script again.\n"
            "Do not proceed with zero flagged layers — the surgical fix "
            "would have nothing to target."
        )
        sys.exit(1)

    # ── 4. Freeze all layers except flagged ones + classifier head ────────────
    print(f"\n{'─'*55}")
    print(f"Layer 4: Surgical Intervention on layers {flagged_layers}")
    print(f"{'─'*55}")

    debiaser = AdversarialDebiaser(model, tokenizer, flagged_layers, device=device)
    model_for_fixing = debiaser.prepare_model_for_intervention()

    trainable = sum(p.numel() for p in model_for_fixing.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model_for_fixing.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)")

    # ── 5. Build dataset with protected_labels for the adversarial trainer ────
    def tokenize_with_protected(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        tokenized["labels"] = examples["label"]
        # 1.0 = most common group (privileged), 0.0 = other groups
        tokenized["protected_labels"] = [
            1.0 if s == mode_val else 0.0 for s in examples["sex"]
        ]
        return tokenized

    dataset = Dataset.from_pandas(finetune_df)
    tokenized = dataset.map(tokenize_with_protected, batched=True)

    # ── 6. Adversarial fine-tuning ────────────────────────────────────────────
    print(f"\nStarting adversarial fine-tuning ({EPOCHS} epochs, alpha={ALPHA})...")
    training_args = TrainingArguments(
        output_dir="./results_fixed",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        logging_steps=25,
        warmup_steps=50,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        # REQUIRED: keep protected_labels in the batch — HuggingFace would
        # otherwise drop columns it doesn't recognise as model inputs
        remove_unused_columns=False,
    )

    trainer = AdversarialTrainer(
        model=model_for_fixing,
        args=training_args,
        train_dataset=tokenized,
        alpha=ALPHA,
        flagged_layers=flagged_layers,   # passed directly, not read from model.config
    )

    trainer.train()

    # ── 7. Save ───────────────────────────────────────────────────────────────
    output_dir = "demo/model/fixed"
    os.makedirs(output_dir, exist_ok=True)
    model_for_fixing.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'─'*55}")
    print(f"✅  Surgically fixed model saved to {output_dir}")
    print(f"    Flagged layers that were debiased: {flagged_layers}")
    print(
        "\nNext steps:"
        "\n  1. Download demo/model/fixed/ from Colab to your local machine."
        "\n  2. Load the app and toggle between 'Biased' and 'Surgically Fixed'"
        "\n     in the sidebar, re-running audits on each to see the before/after."
    )


if __name__ == "__main__":
    run_surgical_fix()
