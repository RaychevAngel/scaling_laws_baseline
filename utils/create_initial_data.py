from process_data import TrajectoryProcessor
import random
from itertools import product
import os
from tqdm import tqdm

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

def process_calculation(numbers, operations, pattern):
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
                f"{numbers[0]}{operations[0]}{numbers[1]}={results[0]} (left: {', '.join(map(str, sorted([results[0], numbers[2], numbers[3]])))})\n",
                f"{results[0]}{operations[1]}{numbers[2]}={results[1]} (left: {', '.join(map(str, sorted([results[1], numbers[3]])))})\n",
                f"{results[1]}{operations[2]}{numbers[3]}={results[2]} (left: {results[2]})\n"
            ]
            expr = f"(({numbers[0]}{operations[0]}{numbers[1]}){operations[1]}{numbers[2]}){operations[2]}{numbers[3]}"
        else:  # (a op1 b) op3 (c op2 d)
            results = [
                eval(f"{numbers[0]}{operations[0]}{numbers[1]}"),
                eval(f"{numbers[2]}{operations[1]}{numbers[3]}"),
                eval(f"({numbers[0]}{operations[0]}{numbers[1]}){operations[2]}({numbers[2]}{operations[1]}{numbers[3]})")
            ]
            solution = [
                f"{numbers[0]}{operations[0]}{numbers[1]}={results[0]} (left: {', '.join(map(str, sorted([results[0], numbers[2], numbers[3]])))})\n",
                f"{numbers[2]}{operations[1]}{numbers[3]}={results[1]} (left: {', '.join(map(str, sorted([results[0], results[1]])))})\n",
                f"{results[0]}{operations[2]}{results[1]}={results[2]} (left: {results[2]})\n"
            ]
            expr = f"({numbers[0]}{operations[0]}{numbers[1]}){operations[2]}({numbers[2]}{operations[1]}{numbers[3]})"
        
        # Only return if result is an integer
        if float(results[2]).is_integer():
            return {
                'numbers': numbers,
                'target': results[2],
                'question': f"Use {', '.join(map(str, sorted(numbers)))} to make {results[2]}.",
                'solution': ''.join(solution),
                'answer': f"The answer is: {remove_unnecessary_parentheses(expr)}={results[2]}.\n"
            }
        return None
    except:
        return None

def generate_random_negative_example(problem):
    """Generate a random incorrect solution for a problem."""
    numbers = problem['numbers']
    target = problem['target']
    
    while True:
        random_ops = random.sample(["+", "-", "*", "/"], 3)
        random_pattern = random.choice([1, 2])
        random_result = process_calculation(numbers, random_ops, random_pattern)
        
        if random_result and abs(random_result['target'] - target) > 0.001:
            return (problem['question'], random_result['solution'] + random_result['answer'], 0.0)
        
def save_questions(train_questions, dev_questions, test_questions):
    """Save questions to files."""
    with open("../questions/train.txt", "w") as f:
        f.write("\n".join(train_questions))
    with open("../questions/dev.txt", "w") as f:
        f.write("\n".join(dev_questions))
    with open("../questions/test.txt", "w") as f:
        f.write("\n".join(test_questions))

def export_data(train_policy_data, train_value_data, dev_policy_data, dev_value_data):
    """Export the generated data using the processor."""
    processor = TrajectoryProcessor()
    output_paths = {
        "train_policy_data_path": "../data/pre_generated/train_policy_data.jsonl",
        "train_value_data_path": "../data/pre_generated/train_value_data.jsonl",
        "dev_policy_data_path": "../data/pre_generated/dev_policy_data.jsonl",
        "dev_value_data_path": "../data/pre_generated/dev_value_data.jsonl",
    }
    processor.export_data((train_policy_data, train_value_data), (dev_policy_data, dev_value_data), output_paths)

# --- Main dataset creation function ---
def create_dataset(range_start, range_end, operations=["+", "-", "*", "/"]):
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
    random.shuffle(problems)
    seen_questions = set()
    unique_problems = []
    
    with tqdm(total=len(problems), desc="Deduplicating problems") as pbar:
        for p in problems:
            if p['question'] not in seen_questions:
                seen_questions.add(p['question'])
                unique_problems.append(p)
            pbar.update(1)
    
    problems = unique_problems
    print(f"After deduplication: {len(problems)} unique problems")
    
    # 3. Split into train, dev, test
    train_problems = problems[40000:]
    dev_problems = problems[20000:40000]
    test_problems = problems[:20000]
    
    print(f"Split into {len(train_problems)} training, {len(dev_problems)} dev, and {len(test_problems)} test problems")
    
    # 4. Extract questions for each split
    train_questions = [p['question'] for p in train_problems]
    dev_questions = [p['question'] for p in dev_problems]
    test_questions = [p['question'] for p in test_problems]
    
    # 5. Save questions to files
    print("Saving questions to files...")
    save_questions(train_questions, dev_questions, test_questions)
    
    # 6. Create policy data (question, solution pairs)
    train_policy_data = [(p['question'], p['solution'] + p['answer']) for p in train_problems]
    dev_policy_data = [(p['question'], p['solution'] + p['answer']) for p in dev_problems]
    
    # 7. Create value data (positive examples with score 1.0)
    train_value_data = [(q, s, 1.0) for q, s in train_policy_data]
    dev_value_data = [(q, s, 1.0) for q, s in dev_policy_data]
    
    # 8. Add negative examples (incorrect solutions with score 0.0)
    print("Generating negative examples for training data...")
    with tqdm(total=len(train_problems), desc="Training negative examples") as pbar:
        for problem in train_problems:
            negative_example = generate_random_negative_example(problem)
            if negative_example:
                train_value_data.append(negative_example)
            pbar.update(1)
    
    print("Generating negative examples for dev data...")
    with tqdm(total=len(dev_problems), desc="Dev negative examples") as pbar:
        for problem in dev_problems:
            negative_example = generate_random_negative_example(problem)
            if negative_example:
                dev_value_data.append(negative_example)
            pbar.update(1)
    
    # 9. Export all data
    print("Exporting data to files...")
    export_data(train_policy_data, train_value_data, dev_policy_data, dev_value_data)
    print("Dataset creation completed successfully!")

if __name__ == "__main__":
    create_dataset(range_start=1, range_end=15, operations=["+", "-", "*", "/"])


