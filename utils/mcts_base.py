from typing import Callable, List, Dict, Tuple
from typing_extensions import Self
import re
import time
import asyncio
import requests
import random

class MCTSNode:
    """Node in Monte Carlo Tree Search"""
    def __init__(self, state: str, parent: Self | None, visit_count: int, action_value: float, value_estimate: float):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = visit_count  # N
        self.action_value = action_value  # Q
        self.value_estimate = value_estimate  # V
        self.labels = []
        self.is_terminal = "The answer is:" in self.state

    @property
    def is_visited(self) -> bool: return self.visit_count > 0
    @property
    def has_children(self) -> bool: return len(self.children) > 0

    def add_children(self, state_values: list[tuple[str, float]]):
        """Add child nodes with their value estimates"""
        self.children.extend([MCTSNode(state=state, parent=self, visit_count=0, 
                                      action_value=value, value_estimate=value) 
                             for state, value in state_values])

    def evaluate_terminal_state(self, question: str) -> float:
        """Evaluate if terminal state solves the arithmetic problem"""
        if not self.is_terminal:
            raise ValueError("Evaluation called on non-terminal state")
        try:
            # Extract target and numbers from question
            question_text = question.strip()
            target_match = re.search(r'make (-?\d+)', question_text)
            numbers_match = re.search(r'Use ([\d, ]+) to make', question_text)
            if not target_match or not numbers_match:
                return 0.0
            
            target = int(target_match.group(1))
            question_nums = sorted([int(x.strip()) for x in numbers_match.group(1).split(',')])
            
            # Extract the line containing "The answer is:"
            answer_line = None
            for line in self.state.split('\n'):
                if "The answer is:" in line:
                    answer_line = line.strip()
                    break
            
            if not answer_line:
                return 0.0
                
            last_line = answer_line.removeprefix("The answer is:").removesuffix(".")
            
            equation_match = re.search(r'([\d\s+\-*/()]+)\s*=\s*(-?\d+)', last_line)
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
                return 1.0 if expr_nums == question_nums and abs(result - target) <= 1e-6 else 0.0
            except:
                return 0.0
        except Exception as e:
            print(f"Error in evaluate_terminal_state: {e}")
            return 0.0

    @property
    def favourite_child(self) -> 'MCTSNode':
        """Return child with highest visit count, breaking ties with Q value"""
        if not self.has_children:
            raise ValueError("Node has no children")
        return max(self.children, key=lambda child: (child.visit_count, child.action_value))

class MCTSTree:
    """Base MCTS tree for exploring solutions"""
    def __init__(self, question: str, max_expansions: int, c_explore: float, request_queue):
        self.root = MCTSNode(state="", parent=None, visit_count=0, action_value=0, value_estimate=0)
        self.question = question
        self.expansion_count = 0
        self.max_expansions = max_expansions
        self.c_explore = c_explore
        self.request_queue = request_queue
        self.non_terminal_leaves = [self.root]

    async def get_action_values(self, node: MCTSNode) -> list[tuple[str, float]]:
        """Get action-value pairs from policy-value network"""
        future = asyncio.Future()
        await self.request_queue.put((self.question, node.state, future))
        try:
            return await asyncio.wait_for(future, timeout=60)
        except asyncio.TimeoutError:
            print(f"Error: No response after 60s at {node.state}.")
            return []

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 formula"""
        ucb1 = lambda child: (child.action_value + 
                            self.c_explore * (node.visit_count ** 0.5) / 
                            (child.visit_count + 1))
        return max(node.children, key=ucb1)

    def backpropagate(self, node: MCTSNode, value: float, is_terminal: bool):
        """Update node statistics from leaf to root"""
        while node:
            node.visit_count += 1
            node.action_value += (value - node.action_value) / node.visit_count
            if is_terminal:
                node.labels.append(value)
            node = node.parent

    async def search(self):
        """Base method for MCTS search (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement search method")

class MCTSForest:
    """Forest of MCTS trees for parallel exploration"""
    def __init__(self, questions: List[str], max_expansions: int, num_trees: int, 
                 c_explore: float, batch_size: int,
                 policy_value_fn: Callable[[List[Tuple[str, str]]], List[List[Tuple[str, float]]]]):
        self.questions = questions
        self.max_expansions = max_expansions
        self.num_trees = num_trees
        self.c_explore = c_explore
        self.policy_value_fn = policy_value_fn
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        self.start_time = time.time()
        self.total_api_calls = 0
        self.trees = [self._create_tree(random.choice(questions)) for _ in range(num_trees)]

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _create_tree method")

    async def _batch_processor(self):
        """Process policy-value network requests in batches"""
        batch, futures = [], []
        print("Starting batch processor")
        
        while True:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.1)
                batch.append((request[0], request[1]))
                futures.append(request[2])
                if len(batch) >= self.batch_size or (batch and self.request_queue.empty()):
                    self.total_api_calls += len(batch)
                    current_batch, current_futures = batch, futures
                    batch, futures = [], []  # Reset for next batch
                    asyncio.create_task(self._process_network_requests(current_batch, current_futures))
            except asyncio.TimeoutError:
                print("Timeout error")
                pass
            await asyncio.sleep(0.001)

    async def _process_network_requests(self, batch: list, futures: list):
        """Process batch of requests through policy-value network"""
        try:
            results = self.policy_value_fn(batch)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)        
        except Exception as e:
            print(f"Network request error: {e}")
            self._handle_batch_error(futures, e)

    def _handle_batch_error(self, futures: list, error: Exception):
        """Handle errors in batch processing"""
        for future in futures:
            if not future.done():
                future.set_exception(error)

    async def _run_tree_spot(self, spot_index: int):
        """Run a single tree spot in the forest"""
        while True:
            current_question = None
            try:
                if self._should_stop_collection():
                    break

                tree = self.trees[spot_index]
                current_question = tree.question
                result = await tree.search()
                self._process_result(result)
                
                # Replace with a new tree on a different question
                next_question = random.choice(self.questions)
                self.trees[spot_index] = self._create_tree(next_question)
            except Exception as e:
                print(f"Tree error at spot {spot_index}: {str(e)}")
                print(f"Current question: {current_question}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)
    
    def _should_stop_collection(self) -> bool:
        """Determine if collection should stop (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _should_stop_collection method")
    
    def _process_result(self, result):
        """Process result from tree search (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _process_result method")

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing"""
        batch_processor = asyncio.create_task(self._batch_processor())
        tree_spots = [self._run_tree_spot(i) for i in range(len(self.trees))]
        await asyncio.gather(*tree_spots)
        batch_processor.cancel()
        try:
            await batch_processor
        except asyncio.CancelledError:
            pass

class RunMCTS:
    """Base configuration class for MCTS"""
    def __init__(self, config: Dict, policy_value_fn: Callable[[List[Tuple[str, str]]], List[List[Tuple[str, float]]]]):
        self.config = config
        self.start_time = time.time()
        self.policy_value_fn = policy_value_fn
        
    def _read_questions(self, path: str) -> List[str]:
        """Read questions from a file"""
        with open(path, 'r') as f:
            return [line.strip() + '\n' for line in f]
            
    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress"""
        runtime = time.time() - self.start_time
        print(f"\n--- Stats after {runtime:.1f} seconds ---")
        print(f"API throughput: {self.total_api_calls / runtime} calls/sec")
        print(f"Total API calls: {self.total_api_calls}")
            
    async def _monitor_collection(self) -> None:
        """Monitor collection progress"""
        stats_interval = self.config['stats_interval']
        last_stats = time.time()

        while True:
            current_time = time.time()
            if current_time - last_stats >= stats_interval:
                self._print_collection_stats()
                last_stats = current_time
            await asyncio.sleep(5)
            
    async def _run_implementation(self):
        """Implementation-specific run logic (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _run_implementation method")
            
    async def run(self) -> None:
        """Run the MCTS process with monitoring"""
        monitor_task = asyncio.create_task(self._monitor_collection())
        try:
            return await self._run_implementation()
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
