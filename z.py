from datasets import load_from_disk
import requests
import os
from tqdm import tqdm
import time
from sklearn.metrics import roc_curve, auc

def process_dataset(dataset):
    raw_texts = [dataset[i]['text'] for i in range(len(dataset))]
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    
    texts = []
    for text, label in zip(raw_texts, labels):
        steps = text.split("\n")
        current_input = steps[0] + "\n"
        for step in steps[1:]:
            if step == "":
                continue
            else:
                current_input += step + "\n"
                texts.append((current_input, label))
    return texts


def obtain_values(dataset, port):
    batch_size = 1500
    results = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        value_inputs = [(batch[j][0], "") for j in range(len(batch))]

        value_resp = requests.post(
                url=f"http://127.0.0.1:{port}/value-prediction",
                json={"questions_and_states": value_inputs},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
        for j, result in enumerate(value_resp.json()['results']):
            results.append((batch[j][0], result, batch[j][1]))
    return results

def plot_results(results_list, file_name=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    
    # Print false positives and false negatives for each checkpoint
    for i, results in enumerate(results_list):
        print(f"\n--- Value Function Checkpoint {i+1} ---")
        
        # Extract values, inputs, and labels
        inputs = [result[0] for result in results]
        values = [result[1] for result in results]
        labels = [result[2] for result in results]
        
        # Apply threshold of 0.5
        predictions = [1 if v >= 0.5 else 0 for v in values]
        
        # Find false positives (predicted 1, actual 0)
        false_positives = [(inputs[j], values[j]) for j in range(len(results)) 
                           if predictions[j] == 1 and labels[j] == 0]
        
        # Find false negatives (predicted 0, actual 1)
        false_negatives = [(inputs[j], values[j]) for j in range(len(results)) 
                           if predictions[j] == 0 and labels[j] == 1]
        
        # Print up to 30 false positives
        print(f"False Positives (predicted positive, actually negative): {len(false_positives)} total")
        for j, (text, value) in enumerate(false_positives[:30]):
            print(f"FP {j+1}: Confidence: {value:.4f}")
            print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
            print("---")
            
        # Print up to 30 false negatives
        print(f"False Negatives (predicted negative, actually positive): {len(false_negatives)} total")
        for j, (text, value) in enumerate(false_negatives[:30]):
            print(f"FN {j+1}: Confidence: {value:.4f}")
            print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
            print("---")

    plt.figure(figsize=(10, 8))
    
    colors = ['darkorange', 'forestgreen', 'royalblue', 'purple', 'crimson', 'gold']
    
    # Plot ROC curve for each value function checkpoint
    for i, results in enumerate(results_list):
        # Extract values and labels
        values = [result[1] for result in results]
        labels = [result[2] for result in results]
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(labels, values)
        roc_auc = auc(fpr, tpr)
        
        # Plot this ROC curve
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                 label=f'Value fn checkpoint {i+1} (AUC = {roc_auc:.4f})')
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Value Function Checkpoints')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(file_name)
    plt.show()

for i in range(6):
    raw_train_dataset = load_from_disk(f"data/value/iteration_{i}/train").shuffle(seed=42).select(range(1000))
    raw_dev_dataset = load_from_disk(f"data/value/iteration_{i}/dev").shuffle(seed=42).select(range(1000))
    train_dataset = process_dataset(raw_train_dataset)
    dev_dataset = process_dataset(raw_dev_dataset)
    train_results = []
    dev_results = []
    for j in range(1, 7):
        port = 8050 + j
        train_results.append(obtain_values(train_dataset, port))
        dev_results.append(obtain_values(dev_dataset, port))
    
    os.makedirs(f"value_test/iteration_{i}", exist_ok=True)
    print(f"Iteration {i} Train results:")
    print(20*'=')
    plot_results(train_results, file_name=f"value_test/iteration_{i}/train_data.png")
    print(20*'=')
    print(f"Iteration {i} Dev results:")
    print(20*'=')
    plot_results(dev_results, file_name=f"value_test/iteration_{i}/dev_data.png")
    print(20*'=')
