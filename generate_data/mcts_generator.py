from typing import List, Dict, Tuple, Callable
import random
import asyncio
from utils.mcts_base import MCTSNode, MCTSTree, MCTSForest, RunMCTS
from datasets import Dataset, DatasetDict
import os

class MCTSTree_Generate(MCTSTree):
    """MCTS tree implementation for training."""
    
    def __init__(self, question: str, max_expansions: int, c_explore: float, request_queue):
        super().__init__(question, max_expansions, c_explore, request_queue)
        self.policy_training_data = []
        self.value_training_data = []

    def _handle_terminal_node(self, node: MCTSNode) -> float:
        """Handle terminal node by evaluating its state"""
        return node.evaluate_terminal_state(self.question)
        
    def _get_search_result(self):
        """Get generation results by collecting data from terminal nodes"""
        for leaf in self.terminal_leaves:
            label = leaf.evaluate_terminal_state(self.question)
            if label == 1:
                self.policy_training_data.append({
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
            self.value_training_data.append({
                    "text": self.question + leaf.state,
                    "labels": soft_values[::-1]
                })
        return self.policy_training_data, self.value_training_data

class MCTSForest_Generate(MCTSForest):
    """Forest of MCTS trees for training."""
    
    def __init__(self, questions: List[str],
                 max_expansions: int, c_explore: float, 
                 policy_value_fn: Callable, batch_size: int):
        super().__init__(questions, max_expansions, c_explore, 
                        batch_size, policy_value_fn)
        self.policy_training_data, self.value_training_data = [], []

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree for training."""
        return MCTSTree_Generate(question=question, max_expansions=self.max_expansions,
                               c_explore=self.c_explore, request_queue=self.request_queue)
        
    def _process_result(self, result):
        """Process policy and value data from tree search"""
        policy_data, value_data = result
        self.policy_training_data.extend(policy_data)
        self.value_training_data.extend(value_data)
        
    def _print_additional_stats(self):
        """Print additional training statistics"""
        print(f"Examples collected: {len(self.value_training_data)}")

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        await super().run_forest()
        return self.policy_training_data, self.value_training_data

class RunMCTS_Generate(RunMCTS):
    """Configuration class for MCTS training."""
    
    def __init__(self, config: Dict, policy_value_fn: Callable):
        super().__init__(config, policy_value_fn)
        self.questions_train, self.questions_val = self._load_questions()
        self.forest_train, self.forest_val = self._initialize_forest()

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls from training and evaluation forests."""
        return self.forest_train.total_api_calls + self.forest_val.total_api_calls

    def _load_questions(self) -> Tuple[List[str], List[str]]:
        """Load questions from configured file."""
        try:
            return (
                self._read_questions(self.config['train_questions_path']),
                self._read_questions(self.config['dev_questions_path'])
            )
        except FileNotFoundError as e:
            print(f"Error loading questions: {e}")
            return [], []

    def _initialize_forest(self) -> Tuple[MCTSForest_Generate, MCTSForest_Generate]:
        """Initialize MCTS forest for training."""
        # Common parameters for forest initialization
        common_params = {
            'policy_value_fn': self.policy_value_fn,
            'max_expansions': self.config['max_expansions'],
            'c_explore': self.config['c_explore'],
            'batch_size': self.config['batch_size'],
        }
        
        return (
            MCTSForest_Generate(questions=self.questions_train, **common_params),
            MCTSForest_Generate(questions=self.questions_val, **common_params)
        )
    
    def export_training_data(self, train_data: Tuple[List, List], val_data: Tuple[List, List]) -> None:
        """Export processed policy and value training data to files."""
        policy_train, value_train = train_data
        policy_val, value_val = val_data
        
        # Create directories and save datasets
        for path, data_dict in [
            (self.config['policy_data_path'], {"train": policy_train, "dev": policy_val}),
            (self.config['value_data_path'], {"train": value_train, "dev": value_val})
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            DatasetDict({k: Dataset.from_list(v) for k, v in data_dict.items()}).save_to_disk(path)

    async def _run_implementation(self):
        """Run the MCTS forest."""
        try:
            train_monitor_task = asyncio.create_task(self._monitor_collection([self.forest_train]))
            train_data = await self.forest_train.run_forest()
            train_monitor_task.cancel()
            val_monitor_task = asyncio.create_task(self._monitor_collection([self.forest_val]))
            val_data = await self.forest_val.run_forest()
            val_monitor_task.cancel()
            self.export_training_data(train_data, val_data)
            return train_data, val_data
        except Exception as e:
            print(f"Error in _run_implementation: {e}")
            raise
