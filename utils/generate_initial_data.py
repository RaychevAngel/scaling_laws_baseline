import random
from itertools import product
import os
from tqdm import tqdm
from datasets import Dataset, DatasetDict

# Set seed for reproducibility
random.seed(42)

# --- Helper functions ---
def evaluate_expression(expr: str) -> float:
    """Evaluate a mathematical expression safely."""
    try:
        return eval(expr)
    except:
        return float('inf')

def remove_unnecessary_parentheses(expr: str) -> str:
    """Remove redundant parentheses from an expression."""
    stack, pairs = [], []
    for i, char in enumerate(expr):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            start = stack.pop()
            pairs.append((start, i))
    
    to_remove = []
    for start, end in pairs:
        new_expr = expr[:start] + expr[start+1:end] + expr[end+1:]
        if evaluate_expression(new_expr) == evaluate_expression(expr):
            to_remove.extend([start, end])
    
    expr_list = list(expr)
    for i in sorted(to_remove, reverse=True):
        expr_list.pop(i)
    return ''.join(expr_list)

def process_calculation(numbers, operations, pattern, min_value = 1, max_value = 999):
    """Calculate a math problem with given numbers, operations, and pattern.
    
    Args:
        numbers: List of 4 integers to use in calculation
        operations: List of 3 operations ('+', '-', '*', '/')
        pattern: 1 for ((a op1 b) op2 c) op3 d, 2 for (a op1 b) op3 (c op2 d)
    
    Returns:
        Dictionary containing problem details or None if invalid
    """
    try:
        # Calculate intermediate results based on pattern
        if pattern == 1:  # ((a op1 b) op2 c) op3 d
            results = [
                eval(f"{numbers[0]}{operations[0]}{numbers[1]}"),
                eval(f"({numbers[0]}{operations[0]}{numbers[1]}){operations[1]}{numbers[2]}"),
                eval(f"(({numbers[0]}{operations[0]}{numbers[1]}){operations[1]}{numbers[2]}){operations[2]}{numbers[3]}")
            ]
            solution = [
                f"{numbers[0]}{operations[0]}{numbers[1]}={int(results[0])} (left: {', '.join(map(str, sorted([int(results[0]), numbers[2], numbers[3]])))})\n",
                f"{int(results[0])}{operations[1]}{numbers[2]}={int(results[1])} (left: {', '.join(map(str, sorted([int(results[1]), numbers[3]])))})\n",
                f"{int(results[1])}{operations[2]}{numbers[3]}={int(results[2])} (left: {int(results[2])})\n"
            ]
            expr = f"(({numbers[0]}{operations[0]}{numbers[1]}){operations[1]}{numbers[2]}){operations[2]}{numbers[3]}"
        else:  # (a op1 b) op3 (c op2 d)
            results = [
                eval(f"{numbers[0]}{operations[0]}{numbers[1]}"),
                eval(f"{numbers[2]}{operations[1]}{numbers[3]}"),
                eval(f"({numbers[0]}{operations[0]}{numbers[1]}){operations[2]}({numbers[2]}{operations[1]}{numbers[3]})")
            ]
            solution = [
                f"{numbers[0]}{operations[0]}{numbers[1]}={int(results[0])} (left: {', '.join(map(str, sorted([int(results[0]), numbers[2], numbers[3]])))})\n",
                f"{numbers[2]}{operations[1]}{numbers[3]}={int(results[1])} (left: {', '.join(map(str, sorted([int(results[0]), int(results[1])])))})\n",
                f"{int(results[0])}{operations[2]}{int(results[1])}={int(results[2])} (left: {int(results[2])})\n"
            ]
            expr = f"({numbers[0]}{operations[0]}{numbers[1]}){operations[2]}({numbers[2]}{operations[1]}{numbers[3]})"
        
        # Check that:
        # 1. All intermediate results are positive integers
        # 2. Final result is an integer between 1 and 999
        if all(isinstance(r, (int, float)) and float(r).is_integer() and r > 0 for r in results) and min_value <= results[2] <= max_value:
            return {
                'numbers': numbers,
                'target': int(results[2]),
                'question': f"Use {', '.join(map(str, sorted(numbers)))} to make {int(results[2])}.",
                'solution': ''.join(solution),
                'answer': f"The answer is: {remove_unnecessary_parentheses(expr)}= {int(results[2])}.\n"
            }
        return None
    except:
        return None

def generate_random_negative_example(problem):
    """Generate a random incorrect solution for a problem."""
    numbers = problem['numbers']
    target = problem['target']
    
    max_attempts = 10  # Limit number of attempts to avoid infinite loops
    for _ in range(max_attempts):
        random_ops = random.sample(["+", "-", "*", "/"], 3)
        random_pattern = random.choice([1, 2])
        random_result = process_calculation(numbers, random_ops, random_pattern)
        
        if random_result and abs(random_result['target'] - target) > 0.001:
            return {
                "text": problem['question'] + '\n' + random_result['solution'] + random_result['answer'],
                "labels": [0.0, 0.0, 0.0, 0.0, 0.0]
            }
    return None

def save_questions(train_questions_splits, dev_questions, test_questions):
    """Save questions to files."""
    os.makedirs("questions", exist_ok=True)
    
    # Save dev and test sets
    with open(f"questions/dev.txt", "w") as f:
        f.write("\n".join(dev_questions))
    
    with open(f"questions/test.txt", "w") as f:
        f.write("\n".join(test_questions))
    
    # Save train splits
    for i, questions in enumerate(train_questions_splits):
        with open(f"questions/train_{i}.txt", "w") as f:
            f.write("\n".join(questions))

def export_data(policy_data_train, value_data_train):
    """Export the generated data using the processor."""
    os.makedirs("data/policy/iteration_0", exist_ok=True)
    os.makedirs("data/value/iteration_0", exist_ok=True)
    
    # Save the datasets
    Dataset.from_list(policy_data_train).save_to_disk("data/policy/iteration_0")
    Dataset.from_list(value_data_train).save_to_disk("data/value/iteration_0")

# --- Main dataset creation function ---
def create_dataset(range_start, range_end, operations=["+", "-", "*", "/"], neg_examples_per_positive=9):
    """Create the full dataset for arithmetic reasoning problems."""
    
    # 1. Generate all possible problems
    problems = []
    list_of_numbers = list(product(range(range_start, range_end+1), repeat=4))
    list_of_operations = list(product(operations, repeat=3))
    
    total_combinations = len(list_of_numbers) * len(list_of_operations) * 2
    
    with tqdm(total=total_combinations, desc="Generating problems") as pbar:
        for numbers in list_of_numbers:
            for operations in list_of_operations:
                for pattern in [1, 2]:
                    problem = process_calculation(numbers, operations, pattern)
                    if problem:
                        problems.append(problem)
                    pbar.update(1)
    
    print(f"Generated {len(problems)} valid problems")
    
    # 2. Shuffle and deduplicate problems
    seen_questions = set()
    unique_problems = []
    
    with tqdm(total=len(problems), desc="Deduplicating problems") as pbar:
        for p in problems:
            if p['question'] not in seen_questions:
                seen_questions.add(p['question'])
                unique_problems.append(p)
            pbar.update(1)
    
    problems = unique_problems
    random.shuffle(problems)
    print(f"After deduplication: {len(problems)} unique problems")
    
    # 3. Split into train/dev/test with 100000/1000/1000 distribution
    total_problems = len(problems)
    num_test = 3000
    num_dev = 3000
    num_train = total_problems - num_test - num_dev
    
    test_problems = problems[:num_test]
    dev_problems = problems[num_test:num_test+num_dev]
    train_problems = problems[num_test+num_dev:num_test+num_dev+num_train]
    
    print(f"Split into {len(train_problems)} training, {len(dev_problems)} dev, and {len(test_problems)} test problems")
    
    # 4. Further split train into 10 categories (train_0 to train_9)
    train_splits = []
    split_size = 3000
    i = 0
    while True:
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, len(train_problems))
        if start_idx < end_idx:
            train_splits.append(train_problems[start_idx:end_idx])
        else:
            break
        i += 1
    
    print(f"Split training data into {len(train_splits)} subsets of {split_size} problems each")
    
    # 5. Extract questions for each split
    train_questions_splits = [[p['question'] for p in split] for split in train_splits]
    dev_questions = [p['question'] for p in dev_problems]
    test_questions = [p['question'] for p in test_problems]
    
    # 6. Save questions to files
    print("Saving questions to files...")
    save_questions(train_questions_splits, dev_questions, test_questions)
    
    # 7. Create policy and value data only for train_0
    print("Creating policy and value data for train_0...")
    train_policy_data, train_value_data = [], []
    for p in train_splits[0] + train_splits[1] + train_splits[2] + train_splits[3]:
        q, sa = p['question'] + '\n', p['solution'] + p['answer']
        
        # Add positive example to policy data
        train_policy_data.append({
            "prompt": q,
            "completion": sa
        })
        
        # Add positive example to value data
        train_value_data.append({
            "text": q + sa,
            "labels": [1.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        # Generate negative examples (9 per positive example)
        negative_examples_generated = 0
        with tqdm(total=neg_examples_per_positive, desc=f"Generating negative examples for a problem", leave=False) as pbar:
            max_attempts = neg_examples_per_positive * 3  # Allow more attempts to find enough negative examples
            attempts = 0
            
            while negative_examples_generated < neg_examples_per_positive and attempts < max_attempts:
                if neg_ex := generate_random_negative_example(p):
                    train_value_data.append(neg_ex)
                    negative_examples_generated += 1
                    pbar.update(1)
                attempts += 1
    
    # 8. Export data for train_0
    print("Exporting data to files...")
    export_data(train_policy_data, train_value_data)
    print("Dataset creation completed successfully!")

if __name__ == "__main__":
    create_dataset(range_start=1, range_end=12, operations=["+", "-", "*", "/"], neg_examples_per_positive=9)


