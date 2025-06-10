from typing import List, Dict, Tuple, Callable
import random
import asyncio
from utils.mcts_base import MCTSNode, MCTSTree, MCTSForest, RunMCTS
from datasets import Dataset, DatasetDict
import os
from pathlib import Path
import shutil   
import time
import re

class MCTSTree_Generate(MCTSTree):
    """MCTS tree implementation for data generation."""
    
    def __init__(self, question: str, max_expansions: int, c_explore: float, request_queue):
        super().__init__(question, max_expansions, c_explore, request_queue)
        self.policy_data = []
        self.value_data = []
        self.sos_data = None
        self.target = re.search(r'make (-?\d+)', self.question).group(1)
    def _handle_terminal_node(self, node: MCTSNode) -> float:
        """Handle terminal node by evaluating its state"""
        return node.evaluate_terminal_state(self.question)

    def _extract_full_trajectory(self) -> str:
        """Extract the full trajectory from the terminal nodes"""
        full_trajectory = ""
        for node in self._nodes:
            if node.is_terminal:
                ancestor_list = []
                current = node
                while current:
                    ancestor_list.append(self._idx[current])
                    current = current.parent
                
                for ancestor in ancestor_list:
                    if ancestor != 0:
                        full_trajectory += f"N{ancestor}->"
                    else:
                        full_trajectory += f"Q | " + self._extract_action(node)
            elif node.parent:
                parent_idx = self._idx[node.parent]
                node_idx = self._idx[node]
                action = self._extract_action(node)
                if parent_idx == 0:
                    full_trajectory += f"N{node_idx}->Q | " + action
                else:
                    full_trajectory += f"N{node_idx}->N{parent_idx} | " + action
        return full_trajectory

    def _get_search_result(self):
        """Get generation results by collecting data from terminal nodes"""
        solution_leaf = None
        for leaf in self.terminal_leaves:
            label = leaf.evaluate_terminal_state(self.question)
            if label == 1:
                self.policy_data.append({
                    "prompt": self.question,
                    "completion": leaf.state
                })
                if solution_leaf is None:
                    solution_leaf = leaf
                else:
                    if solution_leaf.value_estimate < leaf.value_estimate:
                        solution_leaf = leaf
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
        if solution_leaf is not None:
            answer = self._extract_action(solution_leaf)
            self.sos_data = {
                "prompt": "Q | " + self.question,
                "completion": "<START_THOUGHT>\n" + self._extract_full_trajectory() + "<END_THOUGHT>\n<START_ANSWER>\n" + answer + "<END_ANSWER>"
            }
            #if self.sos_data:
                #print(self.sos_data['prompt'] + self.sos_data['completion'] + "\n")
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
        if new_sos_data is not None:
            self.sos_data.append(new_sos_data)

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
        self.forest = MCTSForest_Generate(
            questions=self.questions,
            policy_value_fn=self.policy_value_fn,
            max_expansions=self.config['max_expansions'],
            c_explore=self.config['c_explore'],
            batch_size=self.config['batch_size']
        )

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls from forest."""
        return self.forest.total_api_calls

    def export_data(self, data: Tuple[List, List, List]) -> None:
        """Export data to disk."""
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
