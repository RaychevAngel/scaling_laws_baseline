import os
import requests
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Load dataset and extract inputs/labels
print("Loading data...")
train_dataset = load_from_disk("data/value/iteration_0/train")
inputs = [sample["prompt"][0]["content"] for sample in train_dataset]
labels = np.array([sample["completion"][0]["content"] == "1" for sample in train_dataset], dtype=int)
class_balance = np.mean(labels)
print(f"Loaded {len(inputs)} samples, class balance: {class_balance:.4f}")

# Prepare API requests
questions_and_states = [[parts[0], parts[1]] for parts in [inp.split('\n', 1) for inp in inputs]]

# Make predictions
print("Making predictions...")
url = "http://127.0.0.1:8051/value-prediction"
batch_size = 16
predictions = []

for i in tqdm(range(0, len(questions_and_states), batch_size)):
    batch = questions_and_states[i:i+batch_size]
    try:
        response = requests.post(url, json={"questions_and_states": batch}, 
                               headers={"Content-Type": "application/json"}, timeout=30)
        predictions.extend(response.json()["results"])
    except Exception as e:
        print(f"Error in batch {i//batch_size}: {e}")

predictions = np.array(predictions)

# Replace invalid values if any
invalid_mask = np.isnan(predictions) | np.isinf(predictions)
if np.any(invalid_mask):
    predictions[invalid_mask] = 0.5

# Calculate metrics
auroc = roc_auc_score(labels, predictions)
ap = average_precision_score(labels, predictions)
print(f"AUROC: {auroc:.4f}, Average Precision: {ap:.4f}")

# Plot curves
plt.figure(figsize=(12, 5))

# ROC curve
fpr, tpr, _ = roc_curve(labels, predictions)
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(labels, predictions)
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'AP = {ap:.4f}')
plt.axhline(y=class_balance, color='r', linestyle='--', label=f'Baseline ({class_balance:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig('value_model_curves.png')

# Calculate additional metrics
threshold = 0.5
binary_preds = (predictions >= threshold).astype(int)
accuracy = (binary_preds == labels).mean()

tp = np.sum((binary_preds == 1) & (labels == 1))
fp = np.sum((binary_preds == 1) & (labels == 0))
tn = np.sum((binary_preds == 0) & (labels == 0))
fn = np.sum((binary_preds == 0) & (labels == 1))

precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

print(f"Accuracy: {accuracy:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1:.4f}")

# Find optimal threshold
thresholds = np.linspace(0, 1, 100)
f1_scores = []

for t in thresholds:
    preds = (predictions >= t).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = np.max(f1_scores)
print(f"Best threshold: {best_threshold:.4f} (F1 = {best_f1:.4f})")

# Save results
np.savez('value_model_results.npz', predictions=predictions, labels=labels, 
         fpr=fpr, tpr=tpr, precision=precision, recall=recall, auroc=auroc, ap=ap)
