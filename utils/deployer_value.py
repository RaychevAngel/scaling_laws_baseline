import argparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os
os.environ["VLLM_USE_V1"] = "0" 
from vllm import LLM, SamplingParams

from typing import List, Tuple
import threading
from transformers import AutoTokenizer
import math
import traceback  # Added for better error tracking

class ValueRequest(BaseModel):
    questions_and_states: List[Tuple[str, str]]

class ValueServer:
    def __init__(self, value_model: str, host: str, port: int, endpoint: str, revision: str=None):
        self.value_model = value_model
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.value_token_id = None  # Will be set during startup
        self.app = FastAPI(title="Value Predictor API")
        self.setup_app()
        self.server_thread = None
        self.revision = revision
        
    def setup_app(self):
        app = self.app
        
        @app.on_event("startup")
        async def startup():
            print("Loading value model...")
            app.state.value_llm = LLM(
                model=self.value_model,
                tensor_parallel_size=1,
                disable_log_stats=True,
                revision=self.revision,
                gpu_memory_utilization=0.95
                )
            print("Value model loaded.")
            
            # Set the value token ID for "1"
            self.tokenizer = AutoTokenizer.from_pretrained(self.value_model)
            self.value_token_id = self.tokenizer.encode("1", add_special_tokens=False)[0]
            print(f"Value token ID for '1': {self.value_token_id}")

        @app.post(self.endpoint)
        async def predict_value(request: ValueRequest):
            try:
                if not app.state.value_llm:
                    print("ERROR: Model not initialized")
                    raise HTTPException(status_code=500, detail="Model not initialized")
                
                # Get value predictions
                texts = [q+s for q, s in request.questions_and_states]
                
                values = self._predict_value(app.state.value_llm, texts)
                
                return {"results": values}
            except Exception as e:
                print(f"ERROR in predict_value: {str(e)}")
                print(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    def _predict_value(self, llm, texts: List[str]):
        captured_vals: List[float] = []

        def _grab_value(_, logits: torch.Tensor):
            # compute probabilities and convert to Python floats right away
            if logits.dim() == 1:
                p = torch.sigmoid(logits[self.value_token_id]).item()
                captured_vals.append(p)
            else:
                ps = torch.sigmoid(logits[:, self.value_token_id]).tolist()
                captured_vals.extend(ps)
            return logits  # return unmodified so generation continues

        params = SamplingParams(
            max_tokens        = 1,
            temperature       = 0.0,  # greedy
            top_k             = 1,
            logits_processors = [_grab_value],
        )

        llm.generate(texts, params, use_tqdm=False)

        if len(captured_vals) < len(texts):
            raise RuntimeError(f"Expected {len(texts)} values but got {len(captured_vals)}")

        return captured_vals[0:len(texts)]


    
    def start(self):
        """Start the server in a background thread"""
        self.server_thread = threading.Thread(
            target=lambda: uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"Server started: http://{self.host}:{self.port}")
        
    def stop(self):
        """Server terminates when main program exits (no explicit action needed)"""
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value_model", required=True, help="Model for value prediction")
    parser.add_argument("--host", required=True, help="Host address to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--endpoint", required=True, help="API endpoint path")
    args = parser.parse_args()
    
    temp_server_instance = ValueServer(
        args.value_model, 
        args.host, 
        args.port, 
        args.endpoint
    )

    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(temp_server_instance.app, host=args.host, port=args.port, log_level="warning")
