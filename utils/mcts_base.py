from typing import Callable, List, Dict, Tuple
from typing_extensions import Self
import os
import time
import re
import asyncio
import requests
import random
from graphviz import Digraph
from IPython.display import Image, display


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
        self.terminal_leaves = []

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
            
    def _handle_terminal_node(self, node: MCTSNode) -> float:
        """Handle terminal node - return label value (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _handle_terminal_node method")
        
    def _handle_expansion(self, node: MCTSNode, new_states: list[tuple[str, float]]):
        """Handle node expansion (can be overridden by subclasses for specialized behavior)"""
        if new_states:
            self.non_terminal_leaves.remove(node)
            node.add_children(new_states)
            for child in node.children:
                if child.is_terminal:
                    self.terminal_leaves.append(child)
                else:
                    self.non_terminal_leaves.append(child)
            self.expansion_count += 1
            
    def _get_search_result(self):
        """Get result of search (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _get_search_result method")

    async def search(self):
        """Common MCTS search implementation"""
        current = self.root
        while self.expansion_count < self.max_expansions and self.non_terminal_leaves:
            if current.has_children:
                current = self.select_child(current)
            elif current.is_terminal:
                label = self._handle_terminal_node(current)
                self.backpropagate(current, label, True)
                current = self.root
            elif not current.is_visited:
                self.backpropagate(current, current.value_estimate, False)
                current = self.root
            else:
                try:
                    new_states = await self.get_action_values(current)
                    self._handle_expansion(current, new_states)
                except Exception as e:
                    print(f"Expansion error at state '{current.state}': {e}")
                    break
            await asyncio.sleep(0)
        
        # Visualization is disabled by default
        self.visualize_tree(enable=False)
        return self._get_search_result()


    def visualize_tree(self, enable=False):
        if not enable:
            return
            
        try:
            os.makedirs('visualizations', exist_ok=True)
            q, nodes, values = [self.root], [], []
            while q:
                node = q.pop(0)  # Use FIFO queue to process breadth-first
                nodes.append(node)
                values.append(node.action_value)
                q.extend(node.children)
            def _value_to_hex(v):
                """blue (low) --> red (high) gradient."""
                # action_value is already normalized to a suitable range
                r, g, b = (int(c*255) for c in (v, 0.2, 1-v))
                return f"#{r:02x}{g:02x}{b:02x}"

            dot = Digraph("tree", graph_attr={"rankdir": "TB"})  # Top→Bottom
            for idx, node in enumerate(nodes):
                nid = f"n{idx}"
                node._dot_id = nid                         # stash for edges
                label = f"{self.question}\n {node.state}{node.action_value:.3f}\n{node.visit_count}"
                dot.node(nid, label=label,
                        style="filled",
                        fillcolor=_value_to_hex(node.action_value))
        
            for node in nodes:
                for child in node.children:
                    dot.edge(node._dot_id, child._dot_id)

            timestamp = int(time.time())
            filename = f"visualizations/tree_{timestamp}.png"
            dot.render(filename.replace('.png', ''), format="png", cleanup=True)
        except Exception as e:
            print(f"Error visualizing tree: {e}")
            import traceback
            traceback.print_exc()

class MCTSForest:
    """Forest of MCTS trees for parallel exploration"""
    def __init__(self, questions: List[str], max_expansions: int, 
                 c_explore: float, batch_size: int,
                 policy_value_fn: Callable[[List[Tuple[str, str]]], List[List[Tuple[str, float]]]]):
        self.questions = questions
        self.left_questions = list(questions)
        self.ended_questions_count = 0
        self.questions_lock = asyncio.Lock()

        self.trees = [None] * (2 * batch_size)
        self.max_expansions = max_expansions
        self.c_explore = c_explore
        self.policy_value_fn = policy_value_fn
        self.request_queue = asyncio.Queue()
        self.batch_size = batch_size
        
        self.start_time = time.time()
        self.total_api_calls = 0
        
                
    def print_stats(self):
        """Print current statistics about forest progress"""
        runtime = time.time() - self.start_time
        print(f"\n--- Stats after {runtime:.1f} seconds ---")
        print(f"API throughput: {self.total_api_calls / runtime} calls/sec")
        print(f"Questions progress: {self.ended_questions_count}/{len(self.questions)}")
        self._print_additional_stats()
        
    def _print_additional_stats(self):
        """Hook for subclasses to print additional statistics"""
        pass

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
            try:
                async with self.questions_lock:
                    if not self.left_questions:
                        break
                    question = self.left_questions.pop(0)
                    self.trees[spot_index] = self._create_tree(question)
                result = await self.trees[spot_index].search()
                self._process_result(result)
                self.ended_questions_count += 1
            except Exception as e:
                print(f"Tree error at spot {spot_index}: {str(e)}")
                if self.trees[spot_index]:
                    print(f"Current question: {self.trees[spot_index].question}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)
    
    def _process_result(self, result):
        """Process result from tree search (to be implemented by subclasses)"""
        pass

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
            
    async def _monitor_collection(self, forests):
        """Monitor collection progress"""
        stats_interval = self.config['stats_interval']
        last_stats = time.time()

        while True:
            current_time = time.time()
            if current_time - last_stats >= stats_interval:
                for forest in forests:
                    forest.print_stats()
                last_stats = current_time
            await asyncio.sleep(5)
            
    async def _run_implementation(self):
        """Implementation-specific run logic (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _run_implementation method")
            
    async def run(self) -> None:
        """Run the MCTS process with monitoring"""
        # This dummy implementation will be overridden by subclasses
        monitor_task = asyncio.create_task(self._monitor_collection([]))
        try:
            return await self._run_implementation()
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
