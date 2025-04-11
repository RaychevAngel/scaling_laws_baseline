from typing import List, Dict, Tuple, Callable
import asyncio
import math
from utils.mcts_base import MCTSTree, MCTSForest, RunMCTS
import yaml

class MCTSTree_Evaluate(MCTSTree):
    """MCTS tree implementation for evaluation."""
    
    def __init__(self, question: str, max_expansions: int, c_explore: float, request_queue):
        super().__init__(question, max_expansions, c_explore, request_queue)

    def evaluate_tree(self) -> int:
        """Track the favourite trajectory through the tree and record result."""
        current = self.root
        while current.has_children:
            current = current.favourite_child
        return int(current.evaluate_terminal_state(self.question))

    async def search(self):
        """Perform MCTS search for evaluation."""
        current = self.root
        while (self.expansion_count < self.max_expansions and self.non_terminal_leaves):
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            elif not current.is_visited:
                self.backpropagate(current, current.value_estimate)
                current = self.root
            else:
                try:
                    new_states = await self.get_action_values(current)
                    self.non_terminal_leaves.remove(current)
                    current.add_children(new_states)
                    for child in current.children:
                        if not child.is_terminal:
                            self.non_terminal_leaves.append(child)
                    self.expansion_count += 1
                except Exception as e:
                    print(f"Expansion error: {e}")
                    break
            await asyncio.sleep(0)

        return self.evaluate_tree()

class MCTSForest_Evaluate(MCTSForest):
    """Forest of MCTS trees for evaluation."""
    
    def __init__(self, questions: List[str],
                 max_expansions: int, num_trees: int, c_explore: float, 
                 policy_value_fn: Callable, batch_size: int,
                 delta: float, epsilon: float):
        super().__init__(questions, max_expansions, num_trees, c_explore, 
                        policy_value_fn, batch_size)
        
        self.required_examples = math.ceil((1/(2*(epsilon**2))) * math.log(2/delta))
        self.results = []

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a MCTS tree for evaluation."""
        return MCTSTree_Evaluate(
            question=question,
            max_expansions=self.max_expansions,
            c_explore=self.c_explore,
            request_queue=self.request_queue
        )
        
    def _should_stop_collection(self) -> bool:
        """Stop when we've collected enough samples"""
        return len(self.results) >= self.required_examples
        
    def _process_result(self, result):
        """Process evaluation result from tree search"""
        self.results.append(result)

    async def run_forest(self):
        """Run the forest and return accuracy."""
        await super().run_forest()
        return sum(self.results) / len(self.results)

class RunMCTS_Evaluate(RunMCTS):
    """Configuration class for MCTS evaluation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Load questions and initialize forest
        self.questions_test = self._load_questions()
        self.forest_test = self._initialize_forest()

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls from test forest."""
        return self.forest_test.total_api_calls

    def _load_questions(self) -> List[str]:
        """Load questions from configured file."""
        try:
            test_questions = self._read_questions(self.config['test_questions_path'])
            return test_questions
        except FileNotFoundError as e:
            print(f"Error loading questions: {e}")
            return []

    def _initialize_forest(self) -> MCTSForest_Evaluate:
        """Initialize MCTS forest for evaluation."""
        return MCTSForest_Evaluate(
            questions=self.questions_test,
            policy_value_fn=self.policy_value_fn,
            max_expansions=self.config['max_expansions'],
            c_explore=self.config['c_explore'],
            batch_size=self.config['batch_size'],
            num_trees=2 * self.config['batch_size'],
            delta=self.config['delta'],
            epsilon=self.config['epsilon']
        )

    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress."""
        super()._print_collection_stats()
        print(f"Test examples collected: {len(self.forest_test.results)}/{self.forest_test.required_examples}")
        
    def export_evaluation_results(self, accuracy: float) -> None:
        """Export evaluation results and configuration as a YAML file."""
        try:
            self.config['accuracy'] = accuracy
            filepath = f"{self.config['export_data_path']}{self.config['iteration']}.yaml"
            
            with open(filepath, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
            print(f"Results and configuration saved to {filepath}")
        except Exception as e:
            print(f"Error exporting results: {e}")
            import traceback
            traceback.print_exc()

    async def _run_implementation(self):
        """Run evaluation and return results."""
        accuracy = await self.forest_test.run_forest()
        self.export_evaluation_results(accuracy)
        print(f"Overall Accuracy: {accuracy:.4f}")
        return accuracy 