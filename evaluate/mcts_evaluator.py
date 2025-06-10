from typing import List, Dict, Tuple, Callable
import asyncio
import math
from utils.mcts_base import MCTSNode, MCTSTree, MCTSForest, RunMCTS
import yaml
import os
from datetime import datetime

class MCTSTree_Evaluate(MCTSTree):
    """MCTS tree implementation for evaluation."""
    
    def __init__(self, question: str, max_expansions: List[int], c_explore: float, request_queue):
        super().__init__(question, max_expansions, c_explore, request_queue)
        
    def _handle_terminal_node(self, node: MCTSNode) -> float:
        """Handle terminal node by using its value estimate"""
        return node.value_estimate
        
    def _get_search_result(self):
        """Track the favourite trajectory through the tree and record result."""        
        current = self.root
        while len(current.children) > 0:
            current = current.favourite_child
        # Check if the final node is actually terminal before evaluating
        if current.is_terminal:
            return int(current.evaluate_terminal_state(self.question))
        else:
            return 0

class MCTSForest_Evaluate(MCTSForest):
    """Forest of MCTS trees for evaluation."""
    
    def __init__(self, questions: List[str],
                 max_expansions: List[int], c_explore: float, 
                 policy_value_fn: Callable, batch_size: int):
        super().__init__(questions, max_expansions, c_explore, 
                        batch_size, policy_value_fn)
        
        self.results = {max_expansions: [] for max_expansions in self.max_expansions}

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a MCTS tree for evaluation."""
        return MCTSTree_Evaluate(
            question=question,
            max_expansions=self.max_expansions,
            c_explore=self.c_explore,
            request_queue=self.request_queue
        )
        
    def _process_result(self, result):
        """Process evaluation result from tree search"""
        for max_expansions in self.max_expansions:
            self.results[max_expansions].append(result[max_expansions])
        
    def _print_additional_stats(self):
        """Print additional evaluation statistics"""
        for max_expansions in self.max_expansions:
            if self.results[max_expansions]:
                accuracy = sum(self.results[max_expansions])/len(self.results[max_expansions])
                print(f"Current accuracy for {max_expansions} expansions: {accuracy:.4f}")
            else:
                print(f"No results for {max_expansions} expansions")

    async def run_forest(self):
        """Run the forest and return accuracy."""
        await super().run_forest()
        accuracies = {}
        for key in self.results:
            accuracies[key] = sum(self.results[key])/len(self.results[key])
        return accuracies

class RunMCTS_Evaluate(RunMCTS):
    """Configuration class for MCTS evaluation."""
    
    def __init__(self, config: Dict, policy_value_fn: Callable):
        super().__init__(config, policy_value_fn)
        self.forest = MCTSForest_Evaluate(
            questions=self.questions,
            policy_value_fn=self.policy_value_fn,
            max_expansions=self.config['max_expansions'],
            c_explore=self.config['c_explore'],
            batch_size=self.config['batch_size']
        )

    def export_evaluation_results(self, accuracy: float) -> None:
        """Export evaluation results and configuration as a YAML file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.config['export_data_path'], exist_ok=True)
            file_path = os.path.join(self.config['export_data_path'], f"{timestamp}.yaml")
            
            result = {**self.config, 'accuracy': accuracy}
            
            with open(file_path, 'w') as f:
                yaml.dump(result, f, default_flow_style=False)
            print(f"Results saved to {file_path}")
        except Exception as e:
            print(f"Error exporting results: {e}")
            import traceback
            traceback.print_exc()

    async def _run_implementation(self):
        """Run evaluation and return results."""
        monitor_task = asyncio.create_task(self._monitor_collection([self.forest]))
        try:
            accuracies = await self.forest.run_forest()
            self.export_evaluation_results(0.0)
            print("\n" + 20*"=" + "\n")
            for (max_expansions, accuracy) in accuracies.items():
                print(f"Accuracy for {max_expansions} expansions: {accuracy:.4f}")
            print("\n" + 20*"=" + "\n")
            return accuracies
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass 