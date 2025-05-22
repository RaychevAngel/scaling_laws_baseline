import requests
import re
import os
from datasets import load_from_disk

def main_0():
    questions=["Use 1, 4, 9, 11 to make 25. | Q\n",
            "Use 2, 3, 7, 10 to make 41. | Q\n"]


    sos_resp = requests.post(
        url="http://127.0.0.1:8058/sos-prediction",
        json={
            "questions": questions,
            "temperature": 1.0
        },
        headers={"Content-Type": "application/json"},
        timeout=60
    )

    sos_result = sos_resp.json()['results']
    for question, result in zip(questions, sos_result):
        print(question + result)

########################################

def evaluate_terminal_state(question: str, result: str) -> float:
    """Evaluate if terminal state solves the arithmetic problem"""
    try:
        # Extract target and numbers from question
        question_text = question.strip()
        target_match = re.search(r'make (-?\d+)', question_text)
        numbers_match = re.search(r'Use ([\d, ]+) to make', question_text)

        target = int(target_match.group(1))
        question_nums = sorted([int(x.strip()) for x in numbers_match.group(1).split(',')])
        
        start_tag = "<START_ANSWER>"

        start_idx = result.find(start_tag) + len(start_tag)
        if start_idx == -1:
            return 0.0
        answer = result[start_idx:]
        equation_match = re.search(r'([\d\s+\-*/()]+)\s*=\s*(-?\d+)', answer)
        if not equation_match:
            return 0.0
        
        left_side = equation_match.group(1).strip()
        right_side = int(equation_match.group(2))
        
        if right_side != target:
            return 0.0
        
        # Extract all numbers used in the expression
        expr_nums = sorted([int(n) for n in re.findall(r'\d+', left_side)])
        
        # Verify solution
        try:
            result = eval(left_side)
            is_close = abs(result - target) < 1e-6
            has_correct_nums = expr_nums == question_nums
            return 1.0 if has_correct_nums and is_close else 0.0
        except:
            return 0.0
    except Exception as e:
        print(f"Error in evaluate_terminal_state: {e}")
        return 0.0

def evaluate_terminal_state_2(question: str, result: str) -> float:
    """Evaluate if terminal state solves the arithmetic problem"""
    try:
        # Extract target and numbers from question
        question_text = question.strip()
        target_match = re.search(r'make (-?\d+)', question_text)
        numbers_match = re.search(r'Use ([\d, ]+) to make', question_text)

        target = int(target_match.group(1))
        question_nums = sorted([int(x.strip()) for x in numbers_match.group(1).split(',')])
        
        end_tag = "<END_THOUGHT>"

        end_idx = result.find(end_tag)
        if end_idx == -1:
            return 0.0
        answers = result[:end_idx]
        answers = answers.split("\n")
        answers = [answer.strip() for answer in answers if answer.strip() and "Left" not in answer]

        output = 0.0
        for answer in answers:
            equation_match = re.search(r'([\d\s+\-*/()]+)\s*=\s*(-?\d+)', answer)
            if not equation_match:
                continue
            
            left_side = equation_match.group(1).strip()
            right_side = int(equation_match.group(2))
            
            if right_side != target:
                continue
        
            # Extract all numbers used in the expression
            expr_nums = sorted([int(n) for n in re.findall(r'\d+', left_side)])
            
            # Verify solution
            result = eval(left_side)
            is_close = abs(result - target) < 1e-6
            has_correct_nums = expr_nums == question_nums
            if has_correct_nums and is_close:
                output = 1.0
                break
        return output
    except Exception as e:
        print(f"Error in evaluate_terminal_state_2: {e}")
        return 0.0



def main_1():
    with open("questions/dev.txt", "r") as f:
        questions = [line.strip() + "\n" for line in f]

    correct = 0
    correct_traces = []
    incorrect = 0
    incorrect_traces = []

    batch_size = 200
    num_batches = len(questions) // batch_size
    print(f"Number of batches: {num_batches}")

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        sos_resp = requests.post(
            url="http://127.0.0.1:8058/sos-prediction",
            json={
                "questions": batch,
                "temperature": 1.0
            },
        )
        sos_results = sos_resp.json()['results']
        for question, result in zip(batch, sos_results):
            label = evaluate_terminal_state(question, result)
            if label == 1.0:
                correct += 1
                correct_traces.append(question + "<START_THOUGHT>\n" + result + "<END_ANSWER>\n")
            else:
                incorrect += 1
                incorrect_traces.append(question + "<START_THOUGHT>\n" + result + "<END_ANSWER>\n")
        print(f"Batch {i//batch_size+1} done")
        print(f"Accuracy: {correct/(correct+incorrect)*100:.2f}%")
        #for i in range(3):
        #    print(correct_traces[i])
        #    print(incorrect_traces[i])

    print(f"Final Accuracy: {correct/(correct+incorrect)*100:.2f}%")

def main_2():
    dataset = load_from_disk("data/sos/iteration_0")
    for i in range(100):
        print(dataset[i]['prompt'])
        print(dataset[i]['completion'])
        print("-"*100)
if __name__ == "__main__":
    main_0()
