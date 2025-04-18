import requests
from typing import List, Tuple, Dict

class PolicyValueFunction:
    """Class that handles policy and value predictions from separate servers."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def __call__(self, qs: List[Tuple[str, str]]) -> List[List[Tuple[str, float]]]:
        """Get policy samples and value estimates from the servers."""
        if not qs: 
            return []
            
        try:
            for i, (q,s) in enumerate(qs):
                if len(s.split("\n")) >= 4:
                    qs[i] = (q, s + "The answer is:")

            policy_resp = requests.post(
                url=f"http://{self.config['host']}:{self.config['policy_port']}{self.config['policy_endpoint']}",
                json={
                    "questions_and_states": qs, 
                    "branch_factor": self.config['branch_factor'],
                    "temperature": self.config['temperature']
                },
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if policy_resp.status_code != 200:
                print(f"Policy API error: {policy_resp.status_code} - {policy_resp.text}")
                return [[] for _ in qs]
                
            policy_results = policy_resp.json()['results']
            # Process policy results to get next states
            next_states = []
            for (question, state), actions in zip(qs, policy_results):
                clean_state = self.clean_text(state)
                states_for_this_question = []
                # Iterate through each action proposed for the current state
                for action in actions:
                    if clean_state.endswith("The answer is:"):
                        # If state already ends with the prompt, just append the action
                        new_state = clean_state + action
                    else:
                        # Otherwise, construct the new state more carefully
                        clean_action = self.clean_text(action)
                        if state != "" and action != "":
                            new_state = clean_state + "\n" + clean_action
                        elif state != "":
                            new_state = clean_state
                        elif action != "":
                            new_state = clean_action
                        else:
                            new_state = ""
                        new_state = new_state + self.get_suffix(new_state) # Apply suffix logic
                    states_for_this_question.append(new_state) 
                
                # Append all generated states for this question to the main list
                next_states.append(states_for_this_question)
            
            # Get value predictions for all next states
            value_inputs = [(q[0], s) for i, q in enumerate(qs) for s in next_states[i]]
            value_resp = requests.post(
                url=f"http://{self.config['host']}:{self.config['value_port']}{self.config['value_endpoint']}",
                json={"questions_and_states": value_inputs},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if value_resp.status_code != 200:
                print(f"Value API error: {value_resp.status_code} - {value_resp.text}")
                return [[] for _ in qs]
                
            values = value_resp.json()['results']
            
            # Organize results
            result = [[] for _ in qs]
            positions = [(i, j) for i, states in enumerate(next_states) for j in range(len(states))]
            for (i, j), value in zip(positions, values):
                result[i].append((next_states[i][j], value))
                
            return result
            
        except Exception as e:
            print(f"Request error: {type(e).__name__}: {str(e)}")
            return [[] for _ in qs] 

    def clean_text(self, text: str) -> str:
        """Clean text by removing double newlines and trimming newlines from start and end."""
        text = text.replace("\n\n", "\n").strip("\n")
        return "" if text == "\n" else text

    def get_suffix(self, state: str) -> str:
        """Get the appropriate suffix for the state."""
        if state == "":
            return ""
        elif len(state.split("\n")) < 4:
            return "\n"
        elif not state.rstrip().endswith("."):
            return "."
        else:
            return ""