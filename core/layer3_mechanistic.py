import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

class MechanisticAuditor:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.activations = {}

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            # Extract CLS token activation: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
            # output[0] is the last hidden state of the layer
            self.activations[layer_idx] = output[0][:, 0, :].detach().cpu().numpy()
        return hook

    def extract_activations(self, texts, batch_size=32):
        """
        Runs a forward pass and collects activations from all BERT layers.
        """
        self.model.eval()
        all_layer_activations = {i: [] for i in range(self.model.config.num_hidden_layers)}
        
        # Register hooks
        hooks = []
        for i, layer in enumerate(self.model.bert.encoder.layer):
            hooks.append(layer.register_forward_hook(self._get_hook(i)))
            
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                self.model(**inputs)
                
            for layer_idx, act in self.activations.items():
                all_layer_activations[layer_idx].append(act)
                
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Concatenate batches
        for i in all_layer_activations:
            all_layer_activations[i] = np.concatenate(all_layer_activations[i], axis=0)
            
        return all_layer_activations

    def run_probing_audit(self, texts, protected_labels, threshold=0.7):
        """
        Trains probes on activations to localize bias.
        protected_labels: dict {attr_name: list_of_labels}
        """
        activations = self.extract_activations(texts)
        results = {
            "probe_accuracies": {}, # {attr: {layer_idx: accuracy}}
            "flagged_layers": {},    # {attr: [indices]}
            "threshold": threshold
        }
        
        for attr, labels in protected_labels.items():
            results["probe_accuracies"][attr] = {}
            results["flagged_layers"][attr] = []
            y = np.array(labels)
            
            for layer_idx in range(self.model.config.num_hidden_layers):
                X = activations[layer_idx]
                
                # Use simple Logistic Regression as the probe
                # 3-fold CV is enough for a prototype and faster
                probe = LogisticRegression(max_iter=500, C=0.1)
                scores = cross_val_score(probe, X, y, cv=3)
                acc = np.mean(scores)
                
                results["probe_accuracies"][attr][layer_idx] = round(float(acc), 3)
                if acc > threshold:
                    results["flagged_layers"][attr].append(layer_idx)
                    
        return results

if __name__ == "__main__":
    # Mock test to verify logic
    from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
    config = BertConfig(num_hidden_layers=2, hidden_size=128)
    model = BertForSequenceClassification(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    auditor = MechanisticAuditor(model, tokenizer)
    texts = ["example text 1", "example text 2", "example text 3", "example text 4"] * 5
    labels = {"gender": [0, 1, 0, 1] * 5}
    
    results = auditor.run_probing_audit(texts, labels)
    print("--- Mechanistic Audit Results ---")
    print(results["probe_accuracies"])
