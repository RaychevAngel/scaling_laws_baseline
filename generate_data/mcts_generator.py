from typing import List, Dict, Tuple, Callable
import random
import asyncio
from utils.mcts_base import MCTSNode, MCTSTree, MCTSForest, RunMCTS
from datasets import Dataset, DatasetDict
import os

class MCTSTree_Generate(MCTSTree):
    """MCTS tree implementation for data generation."""
    
    def __init__(self, question: str, max_expansions: int, c_explore: float, request_queue):
        super().__init__(question, max_expansions, c_explore, request_queue)
        self.policy_data = []
        self.value_data = []

    def _handle_terminal_node(self, node: MCTSNode) -> float:
        """Handle terminal node by evaluating its state"""
        return node.evaluate_terminal_state(self.question)
        
    def _get_search_result(self):
        """Get generation results by collecting data from terminal nodes"""
        for leaf in self.terminal_leaves:
            label = leaf.evaluate_terminal_state(self.question)
            if label == 1:
                self.policy_data.append({
                    "prompt": self.question,
                    "completion": leaf.state
                })
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
        return self.policy_data, self.value_data

class MCTSForest_Generate(MCTSForest):
    """Forest of MCTS trees for data generation."""
    
    def __init__(self, questions: List[str],
                 max_expansions: int, c_explore: float, 
                 policy_value_fn: Callable, batch_size: int):
        super().__init__(questions, max_expansions, c_explore, 
                        batch_size, policy_value_fn)
        self.policy_data, self.value_data = [], []

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree for data generation."""
        return MCTSTree_Generate(question=question, max_expansions=self.max_expansions,
                               c_explore=self.c_explore, request_queue=self.request_queue)
        
    def _process_result(self, result):
        """Process policy and value data from tree search"""
        new_policy_data, new_value_data = result
        self.policy_data.extend(new_policy_data)
        self.value_data.extend(new_value_data)
        
    def _print_additional_stats(self):
        """Print additional statistics"""
        print(f"Value examples collected: {len(self.value_data)}")
        print(f"Policy examples collected: {len(self.policy_data)}")

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        await super().run_forest()
        return self.policy_data, self.value_data

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
    
    def export_data(self, data: Tuple[List, List]) -> None:
        """Export processed policy and value data to files."""
        policy_data, value_data = data
            
        # Create directories and save datasets
        for path, data_list in [
            (self.config['policy_data_path'], policy_data),
            (self.config['value_data_path'], value_data)
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Dataset.from_list(data_list).save_to_disk(path)

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
