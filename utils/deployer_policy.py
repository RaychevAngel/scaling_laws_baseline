import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List, Tuple
import threading

class PolicyRequest(BaseModel):
    questions_and_states: List[Tuple[str, str]]
    branch_factor: int
    temperature: float

class PolicyServer:
    def __init__(self, policy_model: str, host: str, port: int, endpoint: str, revision: str=None):
        self.policy_model = policy_model
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.app = FastAPI(title="Policy Predictor API")
        self.setup_app()
        self.server_thread = None
        self.revision = revision
    def setup_app(self):
        app = self.app
        
        @app.on_event("startup")
        async def startup():
            print(f"Loading policy model...")
            if self.revision:
                print(f"Using revision: {self.revision}")
            
            try:
                app.state.policy_llm = LLM(
                    model=self.policy_model,
                    tensor_parallel_size=1,
                    disable_log_stats=True,
                    revision=self.revision,
                    gpu_memory_utilization=0.23
                )
                print("Policy model loaded.")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

        @app.post(self.endpoint)
        async def predict_policy(request: PolicyRequest):
            if not app.state.policy_llm:
                raise HTTPException(status_code=500, detail="Model not initialized")
            
            # Get policy predictions
            texts = [q+s for q, s in request.questions_and_states]
            policy_response = self._predict_policy(
                app.state.policy_llm, 
                texts, 
                request.temperature, 
                request.branch_factor
            )
            
            return {"results": policy_response}
    
    def _predict_policy(self, llm, texts, temperature, branch_factor):
        """Generate policy (action) predictions using beam search"""
        # Configure sampling parameters for policy prediction
        sampling_params = SamplingParams(
            n=branch_factor,
            temperature=temperature,
            max_tokens=20,  
            stop=["\n"],
            ignore_eos=False  # Prevent the model from stopping at EOS token
        )
        # Generate completions
        outputs = llm.generate(texts, sampling_params, use_tqdm=False)
        # Process outputs
        results = []
        for output in outputs:
            # Extract unique actions
            beams = set(o.text.strip() for o in output.outputs)
            results.append(list(beams))
            
        return results
    
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
    parser.add_argument("--policy_model", required=True, help="Model for policy prediction")
    parser.add_argument("--host", required=True, help="Host address to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--endpoint", required=True, help="API endpoint path")
    args = parser.parse_args()
    
    server = PolicyServer(
        args.policy_model,
        args.host, 
        args.port, 
        args.endpoint
    )
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")