import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import os

class AdversarialDebiaser:
    def __init__(self, model, tokenizer, flagged_layers, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.flagged_layers = flagged_layers
        self.device = device

    def prepare_model_for_intervention(self):
        """
        Freezes all parameters except the flagged layers and the classifier head.
        """
        for param in self.model.parameters():
            param.requires_grad = False
            
        for idx in self.flagged_layers:
            for param in self.model.bert.encoder.layer[idx].parameters():
                param.requires_grad = True
                
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        return self.model

class AdversarialTrainer(Trainer):
    """
    Custom Trainer that implements adversarial debiasing.
    Freezes all layers except the flagged ones, then trains with:
        total_loss = task_loss - alpha * adversarial_loss
    Minimising task_loss keeps the model accurate.
    Subtracting adversarial_loss maximises discriminator error,
    making protected attributes harder to predict from activations.
    """
    def __init__(self, *args, alpha=0.1, flagged_layers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        # flagged_layers passed explicitly — never read from model.config
        self.flagged_layers = flagged_layers if flagged_layers is not None else []
        self.discriminator = nn.Linear(768, 1).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        p_labels = inputs.pop("protected_labels").float()
        labels = inputs.get("labels")

        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.get("logits")

        # hidden_states[0] = embeddings, hidden_states[i+1] = BERT layer i output
        # Use the last flagged layer's CLS token as the adversary's input
        last_flagged_idx = max(self.flagged_layers) + 1 if self.flagged_layers else -1
        h_state = outputs.hidden_states[last_flagged_idx][:, 0, :]

        loss_fct = nn.CrossEntropyLoss()
        task_loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        p_logits = self.discriminator(h_state).squeeze(-1)
        adv_loss = nn.BCEWithLogitsLoss()(p_logits, p_labels)

        total_loss = task_loss - (self.alpha * adv_loss)

        return (total_loss, outputs) if return_outputs else total_loss
