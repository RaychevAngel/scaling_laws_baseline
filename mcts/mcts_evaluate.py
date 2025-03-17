import asyncio
import traceback
from mcts.mcts import MCTSForest
from typing import Callable, List, Dict, Tuple
import random
import time
from policy_value_fn import PolicyValueModel
from config_evaluate import get_config



class Run_MCTS_Evaluate:
    def __init__(self, config: Dict):
        self.config = config
        self.questions = self._load_questions()
        self.forest = self._initialize_search()
        self.is_running = False
        self.evaluation_task = None
        self.monitor_task = None

    def _load_questions(self) -> List[str]:
        """Load questions from configured file."""
        questions_path = self.config['paths']['questions_path']
        with open(questions_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
        
    def _initialize_model(self) -> PolicyValueModel:
        """Initialize policy-value network model."""
        api_config = self.config['api']
        forest_config = self.config['forest']
        
        return PolicyValueModel(
            openai_api_base=api_config['openai_api_base'],
            openai_api_key=api_config['openai_api_key'],
            value_api_base_url=api_config['value_api_base_url'],
            policy_model=api_config['policy_model'],
            max_workers_policy=forest_config['max_workers_policy']
        )
    
    def _get_initial_values(self, model: PolicyValueModel) -> List[float]:
        """Get initial value estimates for all questions."""
        initial_states = [(q, "") for q in self.questions]
        return model.batch_value_estimate(initial_states)

    def _initialize_search(self) -> MCTSForest:
        """Initialize MCTS forest."""
        policy_value_model = self._initialize_model()
        initial_values = self._get_initial_values(policy_value_model)
        forest_config = self.config['forest']
        
        # Create wrapper function that adds branch_factor and temperature from config
        def policy_value_fn(questions_states: List[Tuple[str, str]]) -> List[List[Tuple[str, float]]]:
            return policy_value_model.get_policy_value([
                (q, s, forest_config['branch_factor'], forest_config['temperature'])
                for q, s in questions_states
            ])
        
        return MCTSForest(
            initial_values=initial_values,
            questions=self.questions,
            max_expansions=forest_config['max_expansions'],
            num_trees=forest_config['num_trees'],
            exploration_constant=forest_config['c_explore'],
            policy_value_fn=policy_value_fn,
            batch_size=forest_config['batch_size'],
            batch_interval=forest_config['batch_interval']
        )
    
    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress."""
        runtime = time.time() - self.forest.start_time
        
        print(f"\n--- Stats after {runtime:.1f} seconds ---")
        print(f"Total API calls: {self.forest.total_api_calls}")
        print(f"API throughput: {self.forest.total_api_calls / runtime if runtime > 0 else 0:.1f} calls/sec")
        print(f"Questions processed: {self.forest.completed_count}/{len(self.questions)}")

    def _check_evaluation_complete(self) -> bool:
        """Check if evaluation targets have been met."""

        return self.forest.completed_count >= len(self.questions)

    async def _monitor_collection(self) -> None:
        """Monitor collection progress and handle periodic tasks."""
        intervals = self.config['intervals']
        last_stats = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_stats >= intervals['stats_interval']:
                self._print_collection_stats()
                last_stats = current_time
            
            if self._check_evaluation_complete():
                self.is_running = False
                break
            
            await asyncio.sleep(1)

    async def start_evaluation(self) -> float:
        """Start the evaluation process and return the average success rate."""
        if self.is_running:
            return 0.0
        
        self.is_running = True
        average_success_rate = 0.0
        
        try:
            # Start the stats monitoring in the background
            self.monitor_task = asyncio.create_task(self._monitor_collection())
            
            # Run the evaluation and wait for it to complete
            average_success_rate = await self.forest.run_forest()
            
            # Once evaluation is done, we don't need the monitor anymore
            self.is_running = False
            
            # Print final results
            print(f"\nEvaluation complete! Average success rate: {average_success_rate:.2%}")
            
        except asyncio.CancelledError:
            print("\nEvaluation was cancelled.")
        except Exception as e:
            print(f"\nError during evaluation: {e}")
        finally:
            self.is_running = False
            
            # Clean up monitor task if it's still running
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
        
        return average_success_rate

    async def stop_evaluation(self) -> None:
        """Stop the evaluation process."""
        if not self.is_running:
            return
        
        print("\nStopping evaluation...")
        self.is_running = False

        for task in [self.monitor_task, self.evaluation_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


async def main(value_size: int, policy_size: int, branch_factor: int, num_expansions: int, temperature: float, c_explore: float):
    """Main entry point for evaluation."""
    config = get_config(value_size, policy_size, branch_factor, num_expansions, temperature, c_explore)
    
    # Evaluation
    evaluator = Run_MCTS_Evaluate(config=config)
    try:
        print("Starting evaluation...")
        average_success_rate = await evaluator.start_evaluation()
        print(f"value_size={value_size}, policy_size={policy_size}, branch_factor={branch_factor}, num_expansions={num_expansions}, temperature={temperature}, c_explore={c_explore}, average_success_rate={average_success_rate:.2%}")
    except KeyboardInterrupt:
        await evaluator.stop_evaluation()
        

if __name__ == "__main__":
    for i in range(1):
        asyncio.run(main(135, 135, 6, 43, 1.0, 0.3))
        asyncio.run(main(135, 135, 7, 37, 1.0, 0.3))
        asyncio.run(main(135, 135, 8, 32, 1.0, 0.3))
        asyncio.run(main(135, 135, 9, 28, 1.0, 0.3))

        asyncio.run(main(135, 135, 8, 64, 1.0, 0.3))
        asyncio.run(main(135, 135, 10, 51, 1.0, 0.3))
        asyncio.run(main(135, 135, 12, 43, 1.0, 0.3))

        asyncio.run(main(135, 135, 12, 85, 1.0, 0.3))
        asyncio.run(main(135, 135, 14, 73, 1.0, 0.3))
        asyncio.run(main(135, 135, 16, 64, 1.0, 0.3))
        asyncio.run(main(135, 135, 18, 57, 1.0, 0.3))

        asyncio.run(main(135, 135, 20, 102, 1.0, 0.3))
    for j in range(2):
        asyncio.run(main(135, 135, 2, 8, 1.0, 0.3))

        asyncio.run(main(135, 135, 2, 16, 1.0, 0.3))
        asyncio.run(main(135, 135, 3, 11, 1.0, 0.3))
        asyncio.run(main(135, 135, 4, 8, 1.0, 0.3))

        asyncio.run(main(135, 135, 3, 21, 1.0, 0.3))
        asyncio.run(main(135, 135, 4, 16, 1.0, 0.3))
        asyncio.run(main(135, 135, 5, 13, 1.0, 0.3))
        asyncio.run(main(135, 135, 6, 11, 1.0, 0.3))

        asyncio.run(main(135, 135, 5, 26, 1.0, 0.3))
        asyncio.run(main(135, 135, 6, 21, 1.0, 0.3))
        asyncio.run(main(135, 135, 7, 18, 1.0, 0.3))
        asyncio.run(main(135, 135, 8, 16, 1.0, 0.3))

        asyncio.run(main(135, 135, 6, 43, 1.0, 0.3))
        asyncio.run(main(135, 135, 7, 37, 1.0, 0.3))
        asyncio.run(main(135, 135, 8, 32, 1.0, 0.3))
        asyncio.run(main(135, 135, 9, 28, 1.0, 0.3))

        asyncio.run(main(135, 135, 8, 64, 1.0, 0.3))
        asyncio.run(main(135, 135, 10, 51, 1.0, 0.3))
        asyncio.run(main(135, 135, 12, 43, 1.0, 0.3))

        asyncio.run(main(135, 135, 12, 85, 1.0, 0.3))
        asyncio.run(main(135, 135, 14, 73, 1.0, 0.3))
        asyncio.run(main(135, 135, 16, 64, 1.0, 0.3))
        asyncio.run(main(135, 135, 18, 57, 1.0, 0.3))
        
        asyncio.run(main(135, 135, 20, 102, 1.0, 0.3))
    
