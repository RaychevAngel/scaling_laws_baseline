import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List, Tuple
import threading

class SosRequest(BaseModel):
    questions: List[str]
    temperature: float

class SosServer:
    def __init__(self, sos_model: str, revision: str, host: str, port: int, endpoint: str, gpu_memory_utilization: float, max_tokens: int):
        self.sos_model = sos_model
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.app = FastAPI(title="Sos Predictor API")
        self.setup_app()
        self.server_thread = None
        self.revision = revision
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens if max_tokens is not None else 1000
    def setup_app(self):
        app = self.app
        
        @app.on_event("startup")
        async def startup():
            print(f"Loading sos model...")
            if self.revision:
                print(f"Using revision: {self.revision}")
            
            try:
                app.state.sos_llm = LLM(
                    model=self.sos_model,
                    tensor_parallel_size=1,
                    disable_log_stats=True,
                    revision=self.revision,
                    gpu_memory_utilization=self.gpu_memory_utilization
                )
                print("Sos model loaded.")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

        @app.post(self.endpoint)
        async def predict_sos(request: SosRequest):
            if not app.state.sos_llm:
                raise HTTPException(status_code=500, detail="Model not initialized")
            
            # Get policy predictions
            texts = [q + "<START_THOUGHT>\nN1->Q | " for q in request.questions]
            completions, tokens_usage = self._predict_sos(
                app.state.sos_llm, 
                texts, 
                request.temperature
            )
            
            return {
                "completions": completions,
                "tokens_usage": tokens_usage
            }
    
    def _predict_sos(self, llm, texts, temperature):
        """Generate sos predictions using beam search"""
        # Configure sampling parameters for sos prediction
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=self.max_tokens,  
            stop=["<END_ANSWER>"],
            ignore_eos=False
        )
        generations = llm.generate(texts, sampling_params, use_tqdm=False)
        completions = []
        tokens_usage = []
        for generation in generations:
            completions.append(generation.outputs[0].text)
            tokens_usage.append(len(generation.outputs[0].token_ids))
        return completions, tokens_usage
    
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
    parser.add_argument("--sos_model", required=True, help="Model for sos prediction")
    parser.add_argument("--host", required=True, help="Host address to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--endpoint", required=True, help="API endpoint path")
    args = parser.parse_args()
    
    server = SosServer(
        args.sos_model,
        args.host, 
        args.port, 
        args.endpoint
    )
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")