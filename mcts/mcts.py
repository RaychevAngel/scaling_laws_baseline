from typing import Self, Callable, List
import random
import re
import time
import asyncio

class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    
    def __init__(self, state: str, parent: Self | None, visit_count: int | 0, action_value: float, value_estimate: float):
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
        """Evaluate if terminal state solves the 24 game."""
        if not self.is_terminal:
            raise ValueError("Evaluation called on non-terminal state")
        try:
            question_nums = sorted([int(x) for x in question.split()])
            last_line = ''.join(c for c in self.state.split('\n')[-1] 
                              if c in '0123456789+-*/()=')
            parts = last_line.split('=')
            if len(parts) != 2 or parts[1] != '24':
                return 0.0
            
            expr_nums = sorted([int(n) for n in re.findall(r'\d+', parts[0])])
            return 1.0 if expr_nums == question_nums and abs(eval(parts[0]) - 24) <= 1e-6 else 0.0
        except:
            return 0.0

    @property
    def favourite_child(self) -> 'MCTSNode':
        """Return child with highest visit count, breaking ties with Q value."""
        if not self.has_children:
            raise ValueError("Node has no children")
            
        return max(self.children, key=lambda child: (child.visit_count, child.action_value))

class MCTSTree:
    """Single MCTS tree for exploring solutions to a question."""
    
    def __init__(self, root_value: float, question: str, max_expansions: int, 
                 exploration_constant: float, request_queue, is_training: bool,
                 process_policy_trajectory: Callable | None,
                 process_value_trajectory: Callable | None):
        
        self.is_training = is_training

        self.root = MCTSNode(state="", action_value=root_value, value_estimate=root_value)
        self.question = question
        self.expansion_count = 0
        self.max_expansions = max_expansions
        self.exploration_constant = exploration_constant
        self.request_queue = request_queue
        self.non_terminal_leaves = [self.root]

        if self.is_training:
            self.policy_training_data = []
            self.value_training_data = []
            self.process_policy_trajectory = process_policy_trajectory
            self.process_value_trajectory = process_value_trajectory

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
                            self.exploration_constant * (node.visit_count ** 0.5) / 
                            (child.visit_count + 1))
        return max(node.children, key=ucb1)

    def backpropagate(self, node: MCTSNode, value: float):
        """Update node statistics from leaf to root."""
        while node:
            node.visit_count += 1
            node.action_value += (value - node.action_value) / node.visit_count
            node = node.parent

    def evaluate_tree(self) -> int:
        """Track the favourite trajectory through the tree and record result."""
        current = self.root
        while current.has_children:
            current = current.favourite_child
        return int(current.evaluate_terminal_state(self.question))
    
    def process_trajectories(self, question: str, policy_data: list, value_data: list):
        """Process and store trajectories as training data."""
        unique_value_data = list(set(value_data))
        if policy_data:
            policy_data = random.choices(policy_data, k=len(unique_value_data))
        
        processed_policy_data = self.process_policy_trajectory(question, policy_data)
        processed_value_data = self.process_value_trajectory(question, unique_value_data)
        return processed_policy_data, processed_value_data

    async def search(self):
        """Perform MCTS search and collect training data."""
        current = self.root
        while (self.expansion_count < self.max_expansions and self.non_terminal_leaves):
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                if self.is_training:
                    value = current.evaluate_terminal_state(self.question)
                    self.value_training_data.append((current.state, value))
                    if value:    
                        self.policy_training_data.append(current.state)
                self.backpropagate(current, value)
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

        if self.is_training:
            return self.process_trajectories(self.question, self.policy_training_data, self.value_training_data)
        else:
            return self.evaluate_tree()

class MCTSForest:
    """Forest of MCTS trees for parallel exploration of multiple questions."""
    
    def __init__(self, initial_values: list[float], questions: list[str],
                 max_expansions: int, num_trees: int, exploration_constant: float, 
                 policy_value_fn: Callable, target_examples: int | None,
                 process_policy_trajectory: Callable | None,
                 process_value_trajectory: Callable | None,
                 batch_size: int, is_training: bool):
        
        self.is_training = is_training

        # Initialize forest parameters
        self.initial_values = initial_values
        self.questions = questions
        self.max_expansions = max_expansions
        self.num_trees = num_trees
        self.exploration_constant = exploration_constant
        
        # Set network functions
        self.policy_value_fn = policy_value_fn
        self.process_policy_trajectory = process_policy_trajectory
        self.process_value_trajectory = process_value_trajectory
        
        # Initialize data collection
        if self.is_training:
            self.policy_training_data = []
            self.value_training_data = []
            self.policy_data_counts = {q: 0 for q in questions}
            self.value_data_counts = {q: 0 for q in questions}
            self.target_examples = target_examples
        else:
            self.results = {}
            self.completed_count = 0 
            self.left_questions = [q for q in questions]
            self.config_lock = asyncio.Lock()

        # Set up batch processing
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        
        # Track API usage and runtime
        self.total_api_calls = 0
        self.start_time = time.time()
        
        # Create initial trees
        self.trees = self._initialize_trees()

    def _initialize_trees(self) -> List[MCTSTree]:
        """Initialize the forest with trees for each spot."""
        trees = []
        for i in range(self.num_trees):
            question_idx = i % len(self.questions)
            question = self.questions[question_idx]
            trees.append(self._create_tree(question_idx, question))
        return trees

    def _create_tree(self, question_idx: int, question: str) -> MCTSTree:
        """Create a new MCTS tree."""
        return MCTSTree(
            root_value=self.initial_values[question_idx],
            question=question,
            max_expansions=self.max_expansions,
            exploration_constant=self.exploration_constant,
            request_queue=self.request_queue,
            is_training=self.is_training,
            process_policy_trajectory=self.process_policy_trajectory,
            process_value_trajectory=self.process_value_trajectory
        )

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

    async def _run_tree_spot(self, spot_index: int):
        """Manage a single tree spot in the forest."""
        while True:
            current_question = None
            try:
                if self.is_training:
                    if len(self.policy_training_data) >= self.target_examples:
                        break
                else:
                    if len(self.results) >= len(self.questions):
                        break

                # Get current tree and process it
                tree = self.trees[spot_index]
                current_question = tree.question
                result = await tree.search()
                
                if self.is_training:
                    policy_data, value_data = result
                    self.policy_training_data.extend(policy_data)
                    self.value_training_data.extend(value_data)
                    self.policy_data_counts[current_question] += len(policy_data)
                    self.value_data_counts[current_question] += len(value_data)
                else:
                    self.results[current_question] = result
                    self.completed_count += 1
                
                # Update tree with next question
                next_question = await self._select_next_question(current_question)
                self._update_tree_spot(spot_index, next_question)
                
            except Exception as e:
                await self._handle_spot_error(spot_index, current_question, e)
                await asyncio.sleep(1)

    async def _select_next_question(self, current_question: str) -> str:
        """Select next question to process based on data counts."""
        if self.is_training:
            return random.choice(self.questions)
        else:
            async with self.config_lock:
                if self.left_questions:
                    next_question = self.left_questions.pop(0)
                    return next_question
                else:
                    return None

    def _update_tree_spot(self, spot_index: int, question: str):
        """Update tree spot with new question."""
        question_idx = list(self.policy_data_counts.keys()).index(question)
        self.trees[spot_index] = self._create_tree(question_idx, question)

    async def _handle_spot_error(self, spot_index: int, question: str, error: Exception):
        """Handle errors in tree spot processing."""
        print(f"Spot {spot_index} error: {type(error).__name__}: {str(error)}")
        print(f"Current question: {question}")
        print(f"Stack trace:", exc_info=True)
        
        try:
            async with self.config_lock:
                if question and question in self.left_questions:
                    self.left_questions.remove(question)
        except Exception as lock_error:
            print(f"Error cleaning up active questions: {type(lock_error).__name__}: {str(lock_error)}")

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

        if self.is_training:
            return self.policy_training_data, self.value_training_data
        else:
            return sum(self.results.values()) / len(self.results)

