from typing import Self, Callable, List, Dict, Tuple
import random
import re
import time
import asyncio
from utils.process_data import TrajectoryProcessor
from utils.request_api import PolicyValueModel
from config_mcts import get_config
import csv

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
    """Single MCTS tree for exploring solutions to a question."""
    
    def __init__(self, question: str, max_expansions: int, 
                 c_explore: float, request_queue, is_training: bool):
        
        self.is_training = is_training

        self.root = MCTSNode(state="", parent=None, visit_count=0, action_value=0, value_estimate=0)
        self.question = question
        self.expansion_count = 0
        self.max_expansions = max_expansions
        self.c_explore = c_explore
        self.request_queue = request_queue
        self.non_terminal_leaves = [self.root]

        if self.is_training:
            self.policy_training_data = []
            self.value_training_data = []

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

    def evaluate_tree(self) -> int:
        """Track the favourite trajectory through the tree and record result."""
        current = self.root
        while current.has_children:
            current = current.favourite_child
        return int(current.evaluate_terminal_state(self.question))
    
    def deduplicate_trajectories(self, policy_data: list, value_data: list):
        """Deduplicate trajectories and store them as training data."""
        value_data = list(set(value_data))
        if policy_data:
            policy_data = random.choices(policy_data, k=len(value_data))
        return policy_data, value_data

    async def search(self):
        """Perform MCTS search and collect training data."""
        current = self.root
        while (self.expansion_count < self.max_expansions and self.non_terminal_leaves):
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                if self.is_training:
                    label = current.evaluate_terminal_state(self.question)
                    self.value_training_data.append((self.question, current.state, label))
                    if label:    
                        self.policy_training_data.append((self.question, current.state))
                    self.backpropagate(current, label)
                else:
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

        if self.is_training:
            return self.deduplicate_trajectories(self.policy_training_data, self.value_training_data)
        else:
            return self.evaluate_tree()




class MCTSForest:
    """Forest of MCTS trees for parallel exploration of multiple questions."""
    
    def __init__(self, questions: List[str],
                 max_expansions: int, num_trees: int, c_explore: float, 
                 policy_value_fn: Callable, target_examples: int | None,
                 batch_size: int, is_training: bool):
        
        self.is_training = is_training
        

        # Initialize forest parameters
        self.questions = questions
        self.max_expansions = max_expansions
        self.num_trees = num_trees
        self.c_explore = c_explore
        
        # Set network functions
        self.policy_value_fn = policy_value_fn
        
        # Initialize data collection
        if self.is_training:
            self.policy_training_data = []
            self.value_training_data = []
            self.target_examples = target_examples
        else:
            self.results = {}
            self.left_questions = [q for q in questions]
            self.config_lock = asyncio.Lock()

        # Set up batch processing
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        
        # Track runtime
        self.start_time = time.time()
        self.total_api_calls = 0

        # Create initial trees
        self.trees = self._initialize_trees()

    def _initialize_trees(self) -> List[MCTSTree]:
        """Initialize the forest with trees for each spot."""
        trees = []
        for _ in range(self.num_trees):
            if self.is_training:
                question = random.choice(self.questions)
            else:
                if self.left_questions:
                    question = self.left_questions.pop(0)
                else:
                    break

            trees.append(self._create_tree(question))
        return trees

    def _create_tree(self, question: str) -> MCTSTree:
        """Create a new MCTS tree."""
        return MCTSTree( # Access initial value directly
            question=question,
            max_expansions=self.max_expansions,
            c_explore=self.c_explore,
            request_queue=self.request_queue,
            is_training=self.is_training
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
                    if len(self.value_training_data) >= self.target_examples:
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
                else:
                    self.results[current_question] = result
                
                # Update tree with next question
                next_question = await self._select_next_question()
                self._update_tree_spot(spot_index, next_question)
                
            except Exception as e:
                await self._handle_spot_error(spot_index, current_question, e)
                await asyncio.sleep(1)

    async def _select_next_question(self) -> str:
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
        self.trees[spot_index] = self._create_tree(question)

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
            accuracy = sum(self.results.values()) / len(self.results) if self.results else 0.0
            return accuracy




class Run_MCTS_Forest:
    """Run MCTS forest for training or evaluation."""
    def __init__(self, config: Dict):
        """Initialize MCTS forest."""
        self.start_time = time.time()
        self.config = config
        self.is_training = self.config['is_training']
        # Load questions and initialize forests
        self.questions_train, self.questions_val, self.questions_test = self._load_questions()
        self.forest_train, self.forest_val, self.forest_test = self._initialize_forest()

        if self.is_training:
            self.trajectory_processor = TrajectoryProcessor()

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls from training and evaluation forests."""
        if self.is_training:
            return self.forest_train.total_api_calls + self.forest_val.total_api_calls
        else:
            return self.forest_test.total_api_calls

    def _load_questions(self) -> Tuple[List[str], List[str], List[str]]:
        """Load questions from configured file based on mode."""
        try:
            if self.is_training:
                train_questions = self._read_questions(self.config['input_data_paths']['train_questions_path'])
                val_questions = self._read_questions(self.config['input_data_paths']['val_questions_path'])
                return train_questions, val_questions, []
            else:
                test_questions = self._read_questions(self.config['input_data_paths']['test_questions_path'])
                return [], [], test_questions
        except FileNotFoundError as e:
            print(f"Error loading questions: {e}")
            return [], [], []

    def _read_questions(self, path: str) -> List[str]:
        """Read questions from a file."""
        with open(path, 'r') as f:
            return [line.strip() for line in f]

    def _initialize_model(self) -> PolicyValueModel:
        """Initialize policy-value network model."""
        return PolicyValueModel(
            openai_api_base=self.config['openai_api_base'],
            openai_api_key=self.config['openai_api_key'],
            value_api_base_url=self.config['value_api_base'],
            policy_model=self.config['policy_model'],
            max_workers=self.config['max_workers']
        )

    def _initialize_forest(self) -> Tuple[MCTSForest, MCTSForest, MCTSForest]:
        """Initialize MCTS forest."""
        policy_value_model = self._initialize_model()
        
        def policy_value_fn(questions_states: List[Tuple[str, str]]) -> List[List[Tuple[str, float]]]:
            return policy_value_model.get_policy_value(
                questions_states,
                self.config['branch_factor'],
                self.config['temperature']
            )
        
        # Common parameters for forest initialization
        common_params = {
            'max_expansions': self.config['max_expansions'],
            'num_trees': self.config['num_trees'],
            'c_explore': self.config['c_explore'],
            'policy_value_fn': policy_value_fn,
            'batch_size': self.config['batch_size'],
            'is_training': self.is_training
        }
        
        if self.is_training:
            forest_train = MCTSForest(
                questions=self.questions_train,
                target_examples=self.config['target_examples_train'],
                **common_params
            )
            forest_val = MCTSForest(
                questions=self.questions_val,
                target_examples=self.config['target_examples_val'],
                **common_params
            )
            return forest_train, forest_val, None
        else:
            forest_test = MCTSForest(
                questions=self.questions_test,
                **common_params
            )
            return None, None, forest_test

    def export_training_data(self, train_data: Tuple[List, List], val_data: Tuple[List, List]) -> None:
        """Export processed policy and value training data to files.
        
        This function is used only if is_training is True.
        """
        self.trajectory_processor.export_data(train_data, val_data, self.config['output_data_paths'])

    
    def export_evaluation_data(self, accuracy: float) -> None:
        """Export evaluation data to a CSV file."""
        try:
            # Open the CSV file in append mode
            with open(self.config['output_data_paths']['evaluation_stats'], 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Write the header if the file is empty
                if f.tell() == 0:
                    writer.writerow([
                        "accuracy", "iteration", "policy_size", "value_size", 
                        "branch_factor", "max_expansions", "temperature", "c_explore"
                    ])
                
                # Write the data row
                writer.writerow([
                    accuracy, self.config['iteration'], self.config['policy_size'], self.config['value_size'], 
                    self.config['branch_factor'], self.config['max_expansions'], 
                    self.config['temperature'], self.config['c_explore']
                ])
                
        except Exception as e:
            print(f"Error exporting evaluation data: {e}")

    def _print_collection_stats(self) -> None:
        """Print current data collection and processing progress."""
        runtime = time.time() - self.start_time
        print(f"\n--- Stats after {runtime:.1f} seconds ---")
        print(f"API throughput: {self.total_api_calls / runtime} calls/sec")
        if self.is_training:
            print(f"Total API calls: {self.total_api_calls}")
            print(f"Train examples collected: {len(self.forest_train.value_training_data)}/{self.config['target_examples_train']}")
            print(f"Val examples collected: {len(self.forest_val.value_training_data)}/{self.config['target_examples_val']}")
        else:
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
        """Run the MCTS forest."""
        monitor_task = asyncio.create_task(self._monitor_collection())

        try:
            if self.is_training:
                train_data = await self.forest_train.run_forest()
                val_data = await self.forest_val.run_forest()
                self.export_training_data(train_data, val_data)
                print("Total API calls: ", self.total_api_calls)
            else:
                accuracy = await self.forest_test.run_forest()
                self.export_evaluation_data(accuracy)
                print("Accuracy: ", accuracy)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

def main(policy_size: int, value_size: int, branch_factor: int, max_expansions: int, iteration: int):
    config = get_config(policy_size=policy_size, value_size=value_size, branch_factor=branch_factor, max_expansions=max_expansions, iteration=iteration)
    run_mcts_forest = Run_MCTS_Forest(config)

    asyncio.run(run_mcts_forest.run())

if __name__ == "__main__":
    main(policy_size=135, value_size=135, branch_factor=3, max_expansions=20, iteration=2)