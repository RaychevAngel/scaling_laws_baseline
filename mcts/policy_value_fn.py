from openai import OpenAI
import requests
from typing import List, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class PolicyValueModel:
    """Handles policy generation and value estimation for RL-based math problem solving."""
    
    def __init__(
        self, 
        openai_api_base: str,
        openai_api_key: str = "sk-placeholder",
        value_api_base_url: str = None,
        policy_model: str = "mstojkov/policy-135-iter0",
        max_workers: int = 4
    ):
        self.policy_network = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
        self.policy_model = policy_model
        self.value_network_url = value_api_base_url
        self.max_workers = max_workers

    def get_policy_value(self, questions_and_states: List[Tuple[str, str]], branch_factor: int, temperature: float):
        """Sample actions and estimate their values."""
        if not questions_and_states:
            return []
        
        next_states = self.query_policy_network(questions_and_states, branch_factor, temperature)
        
        # Prepare value estimation inputs
        value_inputs = [(q[0], s) for i, q in enumerate(questions_and_states) for s in next_states[i]]
        positions = [(i, j) for i, states in enumerate(next_states) for j in range(len(states))]
        
        values = self.query_value_network(value_inputs)
        
        # Organize results
        result = [[] for _ in questions_and_states]
        for (i, j), value in zip(positions, values):
            result[i].append((next_states[i][j], value))
        
        return result

    def _get_suffix(self, state: str, content: str) -> str:
        """Determine appropriate suffix based on state."""
        lines = len(state.split("\n"))
        if lines < 4:
            return "\n"
        if lines >= 4 and not state.rstrip().endswith(".") and not content.endswith("."):
            return "."
        return ""

    def query_value_network(self, questions_and_states: List[Tuple[str, str]]):
        """Estimate values for multiple states."""
        if not questions_and_states or not self.value_network_url:
            return [0.5] * len(questions_and_states)

        try:
            response = requests.post(
                url=self.value_network_url,
                json={"messages": [{"role": "user", "content": f"{q}\n{s}"} for q, s in questions_and_states]},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            if response.status_code == 200:
                return [float(v) for v in response.json()['value']]
            return [0.5] * len(questions_and_states)
        except Exception as e:
            print(f"Value estimation error: {type(e).__name__}: {str(e)}")
            return [0.5] * len(questions_and_states)

    def query_policy_network(self, questions_and_states: List[Tuple[str, str]], branch_factor: int, temperature: float):
        """Query policy network for next actions and process responses in parallel."""
        if not questions_and_states:
            return [[] for _ in questions_and_states]

        next_states = [[] for _ in questions_and_states]
        
        # Helper function to process a single question-state pair
        def process_single_query(idx_and_question_state):
            idx, (question, state) = idx_and_question_state
            try:
                response = self.policy_network.chat.completions.create(
                    model=self.policy_model,
                    messages=[{"role": "user", "content": f"{question}\n{state}"}],
                    temperature=temperature,
                    n=branch_factor,
                    max_completion_tokens=20,
                    stop=["\n"]
                )
                
                states_for_this_query = []
                for choice in response.choices:
                    if hasattr(choice, 'message'):
                        content = choice.message.content.strip()
                        next_state = state + content + self._get_suffix(state, content)
                        states_for_this_query.append(next_state)
                
                return idx, states_for_this_query
                        
            except Exception as e:
                print(f"Policy network error for message {idx}: {e}")
                return idx, []
        
        # Use ThreadPoolExecutor to process requests in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(questions_and_states))) as executor:
            # Submit all queries to the executor
            future_to_idx = {
                executor.submit(process_single_query, (idx, qs)): idx 
                for idx, qs in enumerate(questions_and_states)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, result_states = future.result()
                next_states[idx] = result_states
        
        return next_states



if __name__ == "__main__":
    model = PolicyValueModel(
        openai_api_base="http://101.98.36.147:42202/v1",
        value_api_base_url=None,
        max_workers=4  # Adjust this based on your API rate limits and performance needs
    )
    
    question1 = "Use 3 7 11 12 to make 10"
    state1 = "11-12=-1 (left: 3, 7, -1)\n"
    question2 = "Use 5 7 11 3 to make 1"
    state2 = "11-3=8 (left: 5, 7, 8)\n"
    results = model.get_policy_value([(question1, state1), (question2, state2)], 3, 1.0)
    
    for result_list in results:
        for next_state, value in result_list:
            print(f"{next_state}{value}")