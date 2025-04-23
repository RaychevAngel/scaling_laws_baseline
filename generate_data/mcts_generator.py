from typing import List, Dict, Tuple, Callable
import random
import asyncio
from utils.mcts_base import MCTSTree, MCTSForest, RunMCTS
from utils.process_data import TrajectoryProcessor

class MCTSTree_Generate(MCTSTree):
    """MCTS tree implementation for training."""
    
    def __init__(self, question: str, max_expansions: int, c_explore: float, request_queue):
        super().__init__(question, max_expansions, c_explore, request_queue)
        self.policy_training_data = []
        self.value_training_data = []

    async def search(self):
        """Perform MCTS search and collect training data."""
        current = self.root
        while self.expansion_count < self.max_expansions and self.non_terminal_leaves:
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                label = current.evaluate_terminal_state(self.question)
                self.backpropagate(current, label)
                current = self.root
            elif not current.is_visited:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            else:
                try:
                    new_states = await self.get_action_values(current)
                    if len(new_states) > 0:
                        self.non_terminal_leaves.remove(current)
                        current.add_children(new_states)
                        for child in current.children:
                            if child.is_terminal:
                                label = child.evaluate_terminal_state(self.question)
                                self.value_training_data.append((self.question, child.state, label))
                                if label:
                                    self.policy_training_data.append((self.question, child.state))
                            else:
                                self.non_terminal_leaves.append(child)
                        self.expansion_count += 1
                except Exception as e:
                    print(f"Expansion error at state '{current.state}': {e}")
                    break # Exit loop on expansion error
            await asyncio.sleep(0)
        return self.policy_training_data, self.value_training_data

class MCTSForest_Generate(MCTSForest):
    """Forest of MCTS trees for training."""
    
    def __init__(self, questions: List[str],
                 max_expansions: int, num_trees: int, c_explore: float, 
                 policy_value_fn: Callable, target_examples: int,
                 batch_size: int):
        super().__init__(questions, max_expansions, num_trees, c_explore, 
                        batch_size, policy_value_fn)
        
        # Initialize data collection
        self.policy_training_data = []
        self.value_training_data = []
        self.target_examples = target_examples

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree for training."""
        return MCTSTree_Generate(
            question=question,
            max_expansions=self.max_expansions,
            c_explore=self.c_explore,
            request_queue=self.request_queue
        )
        
    def _should_stop_collection(self) -> bool:
        """Stop when we've collected enough examples"""
        return len(self.value_training_data) >= self.target_examples
        
    def _process_result(self, result):
        """Process policy and value data from tree search"""
        policy_data, value_data = result
        self.policy_training_data.extend(policy_data)
        self.value_training_data.extend(value_data)

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        await super().run_forest()
        return self.policy_training_data, self.value_training_data

class RunMCTS_Generate(RunMCTS):
    """Configuration class for MCTS training."""
    
    def __init__(self, config: Dict, policy_value_fn: Callable):
        super().__init__(config, policy_value_fn)
        self.trajectory_processor = TrajectoryProcessor()
        
        # Load questions and initialize forests
        self.questions_train, self.questions_val = self._load_questions()
        self.forest_train, self.forest_val = self._initialize_forest()

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls from training and evaluation forests."""
        return self.forest_train.total_api_calls + self.forest_val.total_api_calls

    def _load_questions(self) -> Tuple[List[str], List[str]]:
        """Load questions from configured file."""
        try:
            train_questions = self._read_questions(self.config['train_questions_path'])
            val_questions = self._read_questions(self.config['dev_questions_path'])
            return train_questions, val_questions
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
            'num_trees': 2 * self.config['batch_size'],
        }
        
        forest_train = MCTSForest_Generate(
            questions=self.questions_train,
            target_examples=self.config['target_examples_train'],
            **common_params
        )
        
        forest_val = MCTSForest_Generate(
            questions=self.questions_val,
            target_examples=self.config['target_examples_dev'],
            **common_params
        )
        
        return forest_train, forest_val
    
    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress."""
        super()._print_collection_stats()
        print(f"Train examples collected: {len(self.forest_train.value_training_data)}/{self.config['target_examples_train']}")
        print(f"Dev examples collected: {len(self.forest_val.value_training_data)}/{self.config['target_examples_dev']}")

    def export_training_data(self, train_data: Tuple[List, List], val_data: Tuple[List, List]) -> None:
        """Export processed policy and value training data to files."""
        self.trajectory_processor.export_data(*train_data, *val_data, self.config['policy_data_path'], self.config['value_data_path'])

    async def _run_implementation(self):
        """Run the MCTS forest."""
        train_data = await self.forest_train.run_forest()
        val_data = await self.forest_val.run_forest()
        self.export_training_data(train_data, val_data)
        print("Total API calls: ", self.total_api_calls)
        return train_data, val_data 