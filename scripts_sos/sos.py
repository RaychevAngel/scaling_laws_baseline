from argparse import ArgumentParser
import requests
import re
from datasets import Dataset
from pathlib import Path
import shutil
import json
from typing import List, Dict
import random
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--mode", type=str, required=True, choices=["gen", "eval_train", "eval_dev", "eval_test"])
args = parser.parse_args()

########################################################
sos_port = 8050 + 4*args.gpu + args.port
sos_data_path = f"data/sos/iteration_{args.iter}" if args.mode == "gen" else None
sos_questions_path = (f"questions/train_{4*args.iter + args.port}.txt" if args.mode == "gen" 
                     else f"questions/train_{4*(args.iter-1)}.txt" if args.mode == "eval_train" 
                     else f"questions/dev.txt" if args.mode == "eval_dev" 
                     else f"questions/test.txt" if args.mode == "eval_test" 
                     else None)
########################################################

def print_config():
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.iter}")
    print(f"SoS port: {sos_port}")
    if sos_data_path:
        print(f"SoS data path: {sos_data_path}")
    print(f"SoS questions path: {sos_questions_path}")

def evaluate_solution(question: str, result: str) -> bool:
    """Evaluate if terminal state solves the arithmetic problem"""
    try:
        target = int(re.search(r'make (-?\d+)', question.strip()).group(1))
        question_nums = sorted([int(x.strip()) for x in re.search(r'Use ([\d, ]+) to make', question.strip()).group(1).split(',')])
        
        start_idx = result.find("<START_ANSWER>") + len("<START_ANSWER>")
        if start_idx == -1: return False
        
        equation_match = re.search(r'([\d\s+\-*/()]+)\s*=\s*(-?\d+)', result[start_idx:])
        if not equation_match or int(equation_match.group(2)) != target: return False
        
        left_side = equation_match.group(1).strip()
        right_side = int(equation_match.group(2))
        expr_nums = sorted([int(n) for n in re.findall(r'\d+', left_side)])
        
        try:
            return True if (abs(eval(left_side) - target) < 1e-6 and right_side == target and expr_nums == question_nums) else False
        except: return False
    except Exception as e:
        print(f"Error in evaluate_terminal_state: {e}")
        return False

def export_data(data: List[Dict]):
    p = Path(sos_data_path); p.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = Dataset.load_from_disk(p).to_list()
    except:
        old = []
    for attempt in range(1, 4):
        shutil.rmtree(p, ignore_errors=True)
        Dataset.from_list(old + data).save_to_disk(str(p))
        print(f"[INFO] saved {p} (attempt {attempt})")
        break

def plot_tokens_usage(stats: Dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate statistics
    mean = np.mean(stats["correct_tokens_usage"])
    std = np.std(stats["correct_tokens_usage"])
    
    # Combined histogram
    ax1.hist(stats["correct_tokens_usage"], bins=50, alpha=0.5, label=f"Correct (μ={mean:.1f}, σ={std:.1f})", color='green')
    ax1.set_title("Token Usage Distribution")
    ax1.set_xlabel("Number of Tokens")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    
    # Accuracy vs Token Usage plot
    x_values = [i for i in range(min(stats["correct_tokens_usage"]), max(stats["correct_tokens_usage"]) + 1)]
    y_values = [sum(1 for x in stats["correct_tokens_usage"] if x <= t) / (stats["correct"] + stats["incorrect"]) for t in x_values]
    
    ax2.plot(x_values, y_values)
    ax2.set_title("Accuracy vs Token Usage")
    ax2.set_xlabel("Token Usage Threshold")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    save_path = Path(f"logs/evaluation/sos/iterations_{args.iter}_{args.mode}_gpu{args.gpu}_port{args.port}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    print_config()
    with open(sos_questions_path, "r") as f:
        questions = ["Q | " + line.strip() + "\n" for line in f]
    
    random.shuffle(questions)
    
    stats = {"correct": 0, "incorrect": 0, "correct_tokens_usage": [], "incorrect_tokens_usage": [], "collected": []}
    batch_size = 50
    attemps = 10 if args.mode == "gen" else 5
    
    for j in range(attemps):
        for i in range(0, len(questions), batch_size):
            batch = questions[i:min(i+batch_size, len(questions))]
            sos_results = requests.post(
                url=f"http://127.0.0.1:{sos_port}/sos-prediction",
                json={"questions": batch, "temperature": 1.0},
            ).json()
            
            for question, completion, tokens_used in zip(batch, sos_results['completions'], sos_results['tokens_usage']):
                if evaluate_solution(question, completion):
                    stats["correct"] += 1
                    stats["correct_tokens_usage"].append(tokens_used)
                    stats["collected"].append({
                        "prompt": question,
                        "completion": f"<START_THOUGHT>\nN1->Q | {completion}<END_ANSWER>"
                    })
                else:
                    stats["incorrect"] += 1
                    stats["incorrect_tokens_usage"].append(tokens_used)
            
            print(f"Batch {i//batch_size+1} done")
            if args.mode == "gen":
                print(f"SoS examples collected: {len(stats['collected'])}")
            else:
                print(f"Current Accuracy: {stats['correct']/(stats['correct'] + stats['incorrect'])*100:.2f}%")
    
    if args.mode == "gen":
        unique_questions = set()
        deduplicated = []
        for item in stats["collected"]:
            if item["prompt"] not in unique_questions:
                unique_questions.add(item["prompt"])
                deduplicated.append(item)
        export_data(deduplicated)
    else:
        print(f"Final Accuracy: {stats['correct']/(stats['correct'] + stats['incorrect'])*100:.2f}%")
        plot_tokens_usage(stats)
        
if __name__ == "__main__":
    main() 