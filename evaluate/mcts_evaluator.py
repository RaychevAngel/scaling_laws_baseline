from typing import List, Dict, Tuple, Callable
import time
import asyncio
import csv
from utils.mcts_base import MCTSTree, MCTSForest, RunMCTS

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
                 policy_value_fn: Callable, batch_size: int):
        super().__init__(questions, max_expansions, num_trees, c_explore, 
                        policy_value_fn, batch_size)
        
        # Initialize result tracking
        self.results = {}
        self.left_questions = [q for q in questions]
        self.config_lock = asyncio.Lock()
        
        # Create initial trees
        self.trees = self._initialize_trees()

    def _initialize_trees(self) -> List[MCTSTree]:
        """Initialize the forest with trees for each spot."""
        trees = []
        for _ in range(self.num_trees):
            if self.left_questions:
                question = self.left_questions.pop(0)
                trees.append(self._create_tree(question))
        return trees

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree for evaluation."""
        return MCTSTree_Evaluate(
            question=question,
            max_expansions=self.max_expansions,
            c_explore=self.c_explore,
            request_queue=self.request_queue
        )

    async def _select_next_question(self) -> str:
        """Select next question to process."""
        async with self.config_lock:
            if self.left_questions:
                next_question = self.left_questions.pop(0)
                return next_question
            else:
                return None

    async def _handle_spot_error(self, spot_index: int, question: str, error: Exception):
        """Handle errors in tree spot processing."""
        print(f"Spot {spot_index} error: {type(error).__name__}: {str(error)}")
        print(f"Current question: {question}")
        import traceback
        traceback.print_exc()  # Print the stack trace for more detailed error information
        
        try:
            async with self.config_lock:
                if question and question in self.left_questions:
                    self.left_questions.remove(question)
        except Exception as lock_error:
            print(f"Error cleaning up active questions: {type(lock_error).__name__}: {str(lock_error)}")

    async def _run_tree_spot(self, spot_index: int):
        """Manage a single tree spot in the forest."""
        while True:
            current_question = None
            try:
                if len(self.results) >= len(self.questions):
                    break

                # Get current tree and process it
                tree = self.trees[spot_index]
                current_question = tree.question
                result = await tree.search()
                
                self.results[current_question] = result
                
                # Update tree with next question
                next_question = await self._select_next_question()
                if next_question:
                    self.trees[spot_index] = self._create_tree(next_question)
                else:
                    break
                
            except Exception as e:
                await self._handle_spot_error(spot_index, current_question, e)
                await asyncio.sleep(1)

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        await super().run_forest()
        accuracy = sum(self.results.values()) / len(self.results) if self.results else 0.0
        return accuracy

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

    def _read_questions(self, path: str) -> List[str]:
        """Read questions from a file."""
        with open(path, 'r') as f:
            return [line.strip() for line in f]

    def _initialize_forest(self) -> MCTSForest_Evaluate:
        """Initialize MCTS forest for evaluation."""
        policy_value_model = self._initialize_model()
        
        def policy_value_fn(questions_states: List[Tuple[str, str]]) -> List[List[Tuple[str, float]]]:
            return policy_value_model.get_policy_value(
                questions_states,
                self.config['branch_factor'],
                self.config['temperature']
            )
        
        # Parameters for forest initialization
        forest_params = {
            'questions': self.questions_test,
            'policy_value_fn': policy_value_fn,
            'max_expansions': self.config['max_expansions'],
            'c_explore': self.config['c_explore'],
            'batch_size': self.config['batch_size'],
            'num_trees': 2 * self.config['batch_size'],
        }
        
        return MCTSForest_Evaluate(**forest_params)

    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress."""
        runtime = time.time() - self.start_time
        print(f"\n--- Stats after {runtime:.1f} seconds ---")
        print(f"API throughput: {self.total_api_calls / runtime} calls/sec")
        print(f"Test examples collected: {len(self.forest_test.results)}/{len(self.questions_test)}")

    async def _monitor_collection(self) -> None:
        """Monitor collection progress."""
        stats_interval = self.config['stats_interval']
        last_stats = time.time()

        while True:
            current_time = time.time()
            if current_time - last_stats >= stats_interval:
                self._print_collection_stats()
                last_stats = current_time
            await asyncio.sleep(5)

    async def run(self) -> None:
        """Run the MCTS forest for evaluation."""
        monitor_task = asyncio.create_task(self._monitor_collection())

        try:
            accuracy = await self.forest_test.run_forest()
            print("Accuracy: ", accuracy)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass 