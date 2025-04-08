from typing import Self, Callable, List, Dict
import re
import time
import asyncio
from utils.request_api import PolicyValueModel

class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    
    def __init__(self, state: str, parent: Self | None, visit_count: int, action_value: float, value_estimate: float):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = visit_count  # N
        self.action_value = action_value  # Q
        self.value_estimate = value_estimate  # V
        self.is_terminal = self.state.endswith('.') or self.state.endswith('.\n')

    @property
    def is_visited(self) -> bool:
        return self.visit_count > 0

    @property
    def has_children(self) -> bool:
        return len(self.children) > 0

    def add_children(self, state_values: list[tuple[str, float]]):
        """Add child nodes with their value estimates."""
        self.children.extend([MCTSNode(state=state, parent=self, visit_count=0, 
                                      action_value=value, value_estimate=value) 
                             for state, value in state_values])

    def evaluate_terminal_state(self, question: str) -> float:
        """Evaluate if terminal state solves the arithmetic game with a specified target."""
        if not self.is_terminal:
            raise ValueError("Evaluation called on non-terminal state")
        try:
            # Extract target and numbers from question
            question_text = question.strip()
            target_match = re.search(r'make (\d+)', question_text)
            numbers_match = re.search(r'Use ([\d, ]+) to make', question_text)
            if not target_match or not numbers_match:
                return 0.0
            
            target = int(target_match.group(1))
            question_nums = sorted([int(x.strip()) for x in numbers_match.group(1).split(',')])
            
            # Check solution
            last_line = self.state.split('\n')[-1]
            # Extract the equation part
            equation_match = re.search(r'([\d\s+\-*/()]+)\s*=\s*(\d+)', last_line)
            if not equation_match:
                return 0.0
            
            left_side = equation_match.group(1).strip()
            right_side = int(equation_match.group(2))
            
            if right_side != target:
                return 0.0
            
            # Extract all numbers used in the expression
            expr_nums = sorted([int(n) for n in re.findall(r'\d+', left_side)])
            
            # Check if the expression evaluates to the target and uses the right numbers
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
        """Return child with highest visit count, breaking ties with Q value."""
        if not self.has_children:
            raise ValueError("Node has no children")
            
        return max(self.children, key=lambda child: (child.visit_count, child.action_value))

class MCTSTree:
    """Base class for MCTS tree for exploring solutions to a question."""
    
    def __init__(self, question: str, max_expansions: int, 
                 c_explore: float, request_queue):
        self.root = MCTSNode(state="", parent=None, visit_count=0, action_value=0, value_estimate=0)
        self.question = question
        self.expansion_count = 0
        self.max_expansions = max_expansions
        self.c_explore = c_explore
        self.request_queue = request_queue
        self.non_terminal_leaves = [self.root]

    async def get_action_values(self, node: MCTSNode) -> list[tuple[str, float]]:
        """Get action-value pairs from policy-value network."""
        future = asyncio.Future()
        await self.request_queue.put((self.question, node.state, future))
        try:
            return await asyncio.wait_for(future, timeout=60)
        except asyncio.TimeoutError:
            print(f"Error: No response after 60s at {node.state}.")
            return []

    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 formula."""
        ucb1 = lambda child: (child.action_value + 
                            self.c_explore * (node.visit_count ** 0.5) / 
                            (child.visit_count + 1))
        return max(node.children, key=ucb1)

    def backpropagate(self, node: MCTSNode, value: float):
        """Update node statistics from leaf to root."""
        while node:
            node.visit_count += 1
            node.action_value += (value - node.action_value) / node.visit_count
            node = node.parent

    async def search(self):
        """Base method for MCTS search. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement search method")

class MCTSForest:
    """Base class for a forest of MCTS trees for parallel exploration of multiple questions."""
    
    def __init__(self, questions: List[str],
                 max_expansions: int, num_trees: int, c_explore: float, 
                 policy_value_fn: Callable, batch_size: int):
        # Initialize forest parameters
        self.questions = questions
        self.max_expansions = max_expansions
        self.num_trees = num_trees
        self.c_explore = c_explore
        
        # Set network functions
        self.policy_value_fn = policy_value_fn
        
        # Set up batch processing
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        
        # Track runtime
        self.start_time = time.time()
        self.total_api_calls = 0

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_tree method")

    async def _batch_processor(self):
        """Process policy-value network requests in batches."""
        batch, futures = [], []
        print("Starting batch processor")
        
        while True:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
                batch.append((request[0], request[1]))
                futures.append(request[2])

                if len(batch) >= self.batch_size or (batch and self.request_queue.empty()):
                    self.total_api_calls += len(batch)
                    
                    current_batch, current_futures = batch, futures
                    batch, futures = [], []  # Reset for next batch
                    asyncio.create_task(self._process_network_requests(current_batch, current_futures))
                
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.001)

    async def _process_network_requests(self, batch: list, futures: list):
        """Process batch of requests through policy-value network."""
        try:
            results = self.policy_value_fn(batch)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)        
        except Exception as e:
            print(f"Network request error: {e}")
            self._handle_batch_error(futures, e)

    def _handle_batch_error(self, futures: list, error: Exception):
        """Handle errors in batch processing."""
        for future in futures:
            if not future.done():
                future.set_exception(error)

    async def _select_next_question(self) -> str:
        """Select next question to process. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _select_next_question method")

    async def _run_tree_spot(self, spot_index: int):
        """Base method for managing a single tree spot in the forest. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_tree_spot method")

    async def run_forest(self):
        """Run the MCTS forest with parallel tree processing."""
        batch_processor = asyncio.create_task(self._batch_processor())
        tree_spots = [self._run_tree_spot(i) for i in range(len(self.trees))]
        await asyncio.gather(*tree_spots)
        batch_processor.cancel()
        try:
            await batch_processor
        except asyncio.CancelledError:
            pass

class RunMCTS:
    """Base configuration class for MCTS."""
    def __init__(self, config: Dict):
        self.config = config
        self.start_time = time.time()
        
    def _initialize_model(self) -> PolicyValueModel:
        """Initialize policy-value network model."""
        return PolicyValueModel(
            openai_api_base=self.config['openai_api_base'],
            value_api_base_url=self.config['value_api_base'],
            policy_model=self.config['policy_model']
        )
