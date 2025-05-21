from typing import List, Dict, Tuple, Callable
import random
import asyncio
from utils.mcts_base import MCTSNode, MCTSTree, MCTSForest, RunMCTS
from datasets import Dataset, DatasetDict
import os
from pathlib import Path
import shutil
import time
class MCTSTree_Generate(MCTSTree):
    """MCTS tree implementation for data generation."""
    
    def __init__(self, question: str, max_expansions: int, c_explore: float, request_queue):
        super().__init__(question, max_expansions, c_explore, request_queue)
        self.policy_data = []
        self.value_data = []
        self.sos_data = []

    def _handle_terminal_node(self, node: MCTSNode) -> float:
        """Handle terminal node by evaluating its state"""
        return node.evaluate_terminal_state(self.question)
        
    def _get_search_result(self):
        """Get generation results by collecting data from terminal nodes"""
        solution_leaves = []
        random.shuffle(self.terminal_leaves)
        for leaf in self.terminal_leaves:
            label = leaf.evaluate_terminal_state(self.question)
            if label == 1:
                self.policy_data.append({
                    "prompt": self.question,
                    "completion": leaf.state
                })
                solution_leaves.append(leaf)
            soft_values = []
            current = leaf
            while current:
                if current.labels:
                    soft_values.append(sum(current.labels) / len(current.labels))
                else:
                    soft_values.append(label)
                current = current.parent
            self.value_data.append({
                    "text": self.question + leaf.state,
                    "labels": soft_values[::-1]
                })
        if solution_leaves:
            solution = solution_leaves[0]
            answer = solution.state.removeprefix(solution.parent.state)
            self.sos_data.append({
                "prompt": self.question,
                "completion": self.full_trajectory
            })
            self.sos_data.append({
                "prompt": self.question + self.full_trajectory + "Final Answer:\n",
                "completion": answer
            })
        return self.policy_data, self.value_data, self.sos_data

class MCTSForest_Generate(MCTSForest):
    """Forest of MCTS trees for data generation."""
    
    def __init__(self, questions: List[str],
                 max_expansions: int, c_explore: float, 
                 policy_value_fn: Callable, batch_size: int):
        super().__init__(questions, max_expansions, c_explore, 
                        batch_size, policy_value_fn)
        self.policy_data, self.value_data, self.sos_data = [], [], []

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree for data generation."""
        return MCTSTree_Generate(question=question, max_expansions=self.max_expansions,
                               c_explore=self.c_explore, request_queue=self.request_queue)
        
    def _process_result(self, result):
        """Process policy and value data from tree search"""
        new_policy_data, new_value_data, new_sos_data = result[self.max_expansions[0]] 
        random.shuffle(new_policy_data)
        random.shuffle(new_value_data)
        new_policy_data = new_policy_data[:min(5, len(new_policy_data))]
        new_value_data = new_value_data[:min(5, len(new_value_data))]
        self.policy_data.extend(new_policy_data)
        self.value_data.extend(new_value_data)
        self.sos_data.extend(new_sos_data)
        
    def _print_additional_stats(self):
        """Print additional statistics"""
        print(f"Value examples collected: {len(self.value_data)}")
        print(f"Policy examples collected: {len(self.policy_data)}")
        print(f"SoS examples collected: {len(self.sos_data)}")

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        await super().run_forest()
        return self.policy_data, self.value_data, self.sos_data

class RunMCTS_Generate(RunMCTS):
    """Configuration class for MCTS data generation."""
    
    def __init__(self, config: Dict, policy_value_fn: Callable):
        super().__init__(config, policy_value_fn)
        self.questions = self._load_questions()
        self.forest = self._initialize_forest()

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls from forest."""
        return self.forest.total_api_calls

    def _load_questions(self) -> List[str]:
        """Load questions from configured file."""
        try:
            return self._read_questions(self.config['train_questions_path'])
        except FileNotFoundError as e:
            print(f"Error loading questions: {e}")
            return []

    def _initialize_forest(self) -> MCTSForest_Generate:
        """Initialize MCTS forest for data generation."""
        return MCTSForest_Generate(
            questions=self.questions,
            policy_value_fn=self.policy_value_fn,
            max_expansions=self.config['max_expansions'],
            c_explore=self.config['c_explore'],
            batch_size=self.config['batch_size']
        )

    def export_data(self, data: Tuple[List, List, List]) -> None:
        """
        1. Load existing directory (or empty)
        2. Append new data
        3. Overwrite target dir with retries
        4. On final failure, write to a timestamped sibling dir
        """
        for key, idx in (("policy_data_path", 0), ("value_data_path", 1), ("sos_data_path", 2)):
            p = Path(self.config[key]); p.parent.mkdir(parents=True, exist_ok=True)
            try:
                old = Dataset.load_from_disk(p).to_list()
            except:
                old = []
            comb = old + data[idx]

            for attempt in range(1, 4):
                try:
                    shutil.rmtree(p, ignore_errors=True)
                    Dataset.from_list(comb).save_to_disk(str(p))
                    print(f"[INFO] saved {p} (attempt {attempt})")
                    break
                except Exception as e:
                    print(f"[WARN] save#{attempt} for {p} failed: {e}")
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fb = p.parent / f"{p.name}_{ts}"
                try:
                    Dataset.from_list(comb).save_to_disk(str(fb))
                    print(f"[INFO] fallback saved to {fb}")
                except Exception as e:
                    print(f"[ERROR] fallback save {fb} failed: {e}")

    async def _run_implementation(self):
        """Run the MCTS forest."""
        try:
            monitor_task = asyncio.create_task(self._monitor_collection([self.forest]))
            data = await self.forest.run_forest()
            monitor_task.cancel()
            self.export_data(data)
            return data
        except Exception as e:
            print(f"Error in _run_implementation: {e}")
            raise
