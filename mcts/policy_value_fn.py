from openai import OpenAI
import requests
from typing import List, Tuple

class PolicyValueModel:
    """Handles policy generation and value estimation for RL-based math problem solving."""
    
    def __init__(
        self, 
        openai_api_base: str,
        openai_api_key: str = "sk-placeholder",
        value_api_base_url: str = None,
        policy_model: str = "lakomey/sft-135-iter1-10-b32"
    ):
        self.policy_network = OpenAI(base_url=openai_api_base, api_key=openai_api_key)
        self.policy_model = policy_model
        self.value_network_url = value_api_base_url

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
        """Query policy network for next actions and process responses."""
        if not questions_and_states:
            return [[] for _ in questions_and_states]

        try:
            response = self.policy_network.chat.completions.create(
                model=self.policy_model,
                messages=[{"role": "user", "content": f"{q}\n{s}"} for q, s in questions_and_states],
                temperature=temperature,
                n=branch_factor,
                max_completion_tokens=20,
                stop=["\n"]
            )
            
            # Process responses using the index field
            next_states = [[] for _ in questions_and_states]
            for choice in response.choices:
                if hasattr(choice, 'message'):
                    idx = choice.index
                    content = choice.message.content.strip()
                    state = questions_and_states[idx][1] + content + self._get_suffix(questions_and_states[idx][1], content)
                    next_states[idx].append(state)
            return next_states

        except Exception as e:
            print(f"Policy network error: {e}")
            return [[] for _ in questions_and_states]



if __name__ == "__main__":
    model = PolicyValueModel(
        openai_api_base="http://81.166.173.12:10569/v1",
        value_api_base_url="http://142.113.129.186:44723/predict"
    )
    
    question = "3 7 11 12"
    state = "11-12=-1 (left: 3, 7, -1)\n"
    results = model.get_policy_value([(question, state)], 40, 1.0)
    
    for result_list in results:
        for next_state, value in result_list:
            print(f"{next_state}{value}")