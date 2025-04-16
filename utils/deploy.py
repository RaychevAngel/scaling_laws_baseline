import argparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List, Tuple
import threading
from transformers import AutoTokenizer

class PolicyValueRequest(BaseModel):
    questions_and_states: List[Tuple[str, str]]
    branch_factor: int
    temperature: float

class PolicyValueServer:
    def __init__(self, policy_model: str, value_model: str, host: str, port: int, endpoint: str):
        self.policy_model = policy_model
        self.value_model = value_model
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.value_token_id = None  # Will be set during startup
        self.app = FastAPI(title="Policy-Value Predictor API")
        self.setup_app()
        self.server_thread = None
        
    def setup_app(self):
        app = self.app
        
        @app.on_event("startup")
        async def startup():
            # Initialize models when server starts
            # Note: vllm 0.8.4 does not support tensor_parallel_devices for specific GPU assignment
            app.state.policy_llm = LLM(model=self.policy_model)
            app.state.value_llm = LLM(model=self.value_model)
            
            # Set the value token ID for "1"
            tokenizer = AutoTokenizer.from_pretrained(self.value_model)
            self.value_token_id = tokenizer.encode("1", add_special_tokens=False)[0]
            print(f"Value token ID for '1': {self.value_token_id}")
            # Cannot reliably print GPU assignment with this vLLM version


        @app.post(self.endpoint)
        async def predict_policy_value(request: PolicyValueRequest):
            if not app.state.policy_llm or not app.state.value_llm:
                raise HTTPException(status_code=500, detail="Models not initialized")
            
            # Get policy predictions
            texts = [f"{q}\n{s}" for q, s in request.questions_and_states]
            policy_response = self._predict_policy(
                app.state.policy_llm, 
                texts, 
                request.temperature, 
                request.branch_factor
            )
            
            # Process policy results
            next_states = []
            for (question, state), actions in zip(request.questions_and_states, policy_response):
                states_for_this_question = [
                    state + action + self._get_suffix(state, action)
                    for action in actions
                ]
                next_states.append(states_for_this_question)
            
            # Get value predictions
            value_inputs = [(q[0], s) for i, q in enumerate(request.questions_and_states) for s in next_states[i]]
            value_texts = [f"{q}\n{s}" for q, s in value_inputs]
            values = self._predict_value(app.state.value_llm, value_texts)
            
            # Organize results
            result = [[] for _ in request.questions_and_states]
            positions = [(i, j) for i, states in enumerate(next_states) for j in range(len(states))]
            for (i, j), value in zip(positions, values):
                result[i].append((next_states[i][j], value))
            
            return {"results": result}
    
    def _predict_policy(self, llm, texts, temperature, branch_factor):
        """Generate policy (action) predictions using beam search"""
        # Configure sampling parameters for policy prediction
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=20,  
            stop=["\n"],
            use_beam_search=True,
            best_of=branch_factor,  
            length_penalty=0.0 
        )
        
        # Generate completions
        outputs = llm.generate(texts, sampling_params)
        
        # Process outputs
        results = []
        for output in outputs:
            # Extract unique actions
            beams = set(o.text.strip() for o in output.outputs)
            results.append(list(beams))
            
        return results
    
    def _predict_value(self, llm, texts):
        """Predict value scores based on first token logprobs"""
        # Configure sampling parameters for value prediction
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1
        )
        
        # Generate completions
        outputs = llm.generate(texts, sampling_params)
        
        # Process outputs
        results = []
        for output in outputs:
            value = torch.exp(torch.tensor(output.outputs[0].logprobs[0][self.value_token_id])).item()
            results.append(value)
            
        return results
    
    def _get_suffix(self, state: str, action: str) -> str:
        """Determine appropriate suffix for concatenating state and action"""
        lines = len(state.split("\n"))
        if lines < 4:
            return "\n"
        if lines >= 4 and not state.rstrip().endswith(".") and not action.endswith("."):
            return "."
        return ""
    
    def start(self):
        """Start the server in a background thread"""
        self.server_thread = threading.Thread(
            target=lambda: uvicorn.run(self.app, host=self.host, port=self.port)
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"Server started: http://{self.host}:{self.port}")
        
    def stop(self):
        """Server terminates when main program exits"""
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value_model", required=True, help="Model for value prediction")
    parser.add_argument("--policy_model", required=True, help="Model for policy prediction")
    parser.add_argument("--host", required=True, help="Host address to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--endpoint", required=True, help="API endpoint path")
    args = parser.parse_args()
    
    server = PolicyValueServer(
        args.policy_model, 
        args.value_model, 
        args.host, 
        args.port, 
        args.endpoint
    )
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Example: curl -X POST \"http://{args.host}:{args.port}{args.endpoint}\" -H \"Content-Type: application/json\" -d '{{\"questions_and_states\": [[\"What is 2+2?\", \"Let\\\'s solve:\"]], \"branch_factor\": 2}}'")
    uvicorn.run(server.app, host=args.host, port=args.port) 