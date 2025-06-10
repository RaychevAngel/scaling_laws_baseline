import datasets
import torch
import numpy as np
from typing import List, Tuple
import requests
import json
import random

def load_value_dataset(path: str):
    """Load the value dataset from the given path"""
    dataset = datasets.load_from_disk(path)
    return dataset

def process_text_sequences(text: str) -> List[str]:
    """
    Split text by \n and create progressive sequences:
    s0+\n, s0+\n+s1+\n, s0+\n+s1+\n+s2+\n, etc.
    """
    lines = text.split('\n')
    sequences = []
    
    for i in range(len(lines)-1):  # Changed to len(lines)-1 to exclude last line
        # Build progressive sequence
        sequence = '\n'.join(lines[:i+1])
        sequence += '\n'
        sequences.append(sequence)
    
    return sequences

def evaluate_sequences_with_value_server(sequences: List[str]) -> List[float]:
    """
    Evaluate sequences using the value server
    """
    # Create dummy questions (empty strings) paired with the sequences as states
    value_inputs = [("", seq) for seq in sequences]
    
    response = requests.post(
        url=f"http://127.0.0.1:8051/value-prediction",
        json={"questions_and_states": value_inputs},
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code}, {response.text}")
    
    result = response.json()['results']
    readable_result = []
    for r in result:
        readable_result.append(round(r, 2))

    return readable_result

def evaluate_value_function(dataset_path: str):
    """
    Main evaluation function
    """
    print("Loading dataset...")
    dataset = load_value_dataset(dataset_path).select(range(1000))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample entry keys: {dataset[0].keys()}")
    
    all_predictions = []
    all_labels = []
    all_random_predictions = []
    
    print("Processing dataset...")
    for i, item in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processing item {i}/{len(dataset)}")
        
        text = item['text']
        labels = item['labels']
        
        # Process text into progressive sequences
        sequences = process_text_sequences(text)
        
        try:
            # Get predictions from value server
            predictions = evaluate_sequences_with_value_server(sequences)
            random_predictions = [random.uniform(0, 1) for _ in range(len(predictions))]
                        
            # Store results
            all_predictions.extend(predictions)
            all_random_predictions.extend(random_predictions)
            all_labels.extend(labels)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    # Calculate evaluation metrics
    predictions_array = np.array(all_predictions)
    labels_array = np.array(all_labels)
    random_predictions_array = np.array(all_random_predictions)
    # Mean Squared Error
    # Binary cross entropy loss for classification
    epsilon = 1e-15  # Small constant to avoid log(0)
    predictions_array = np.clip(predictions_array, epsilon, 1 - epsilon)
    random_predictions_array = np.clip(random_predictions_array, epsilon, 1 - epsilon)
    
    bce = -np.mean(labels_array * np.log(predictions_array) + (1 - labels_array) * np.log(1 - predictions_array))
    bce_random = -np.mean(labels_array * np.log(random_predictions_array) + (1 - labels_array) * np.log(1 - random_predictions_array))
    
    # Accuracy metric
    accuracy = np.mean((predictions_array > 0.5) == labels_array)
    accuracy_random = np.mean((random_predictions_array > 0.5) == labels_array)
    
    print("\n=== Evaluation Results ===")
    print(f"Total samples evaluated: {len(all_predictions)}")
    print(f"Binary Cross Entropy: {bce:.6f}")
    print(f"Binary Cross Entropy (Random): {bce_random:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Accuracy (Random): {accuracy_random:.6f}")
    
    return {
        'bce': bce,
        'bce_random': bce_random,
        'accuracy': accuracy,
        'accuracy_random': accuracy_random,
    }

if __name__ == "__main__":
    # First, let's examine a sample from the dataset
    print("=== Examining Dataset Structure ===")
    dataset_path = "data/value/iteration_1"
    dataset = load_value_dataset(dataset_path)
    
    for i in range(1):
        sample = dataset[i]
        example = 'Use 7, 8, 9, 12 to make 203.\n7+12=19 (left: 8, 9, 19)\n19*9=171 (left: 8, 171)\n171+8=179 (left: 179)\nThe answer is: 7+12*9+8= 203.\n'
        sequences = process_text_sequences(example)
        
        for seq in sequences:
            print(seq)
        print(evaluate_sequences_with_value_server(sequences))
        
        #print(sample['labels'])
        print("--------------------------------")

    evaluate_value_function(dataset_path)
