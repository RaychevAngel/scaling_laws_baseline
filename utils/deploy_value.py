import argparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List, Tuple
import threading
from transformers import AutoTokenizer
import math
import traceback  # Added for better error tracking

class ValueRequest(BaseModel):
    questions_and_states: List[Tuple[str, str]]

class ValueServer:
    def __init__(self, value_model: str, host: str, port: int, endpoint: str):
        self.value_model = value_model
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.value_token_id = None  # Will be set during startup
        self.app = FastAPI(title="Value Predictor API")
        self.setup_app()
        self.server_thread = None

    def setup_app(self):
        app = self.app

        @app.on_event("startup")
        async def startup():
            print("Loading value model...")
            app.state.value_llm = LLM(
                model=self.value_model,
                tensor_parallel_size=1,
                disable_log_stats=True,

                # new: larger auto-batch caps – higher sm occupancy
                gpu_memory_utilization = 0.9,
                max_num_batched_tokens = 2048,
                max_num_seqs = 512,
            )
            print("Value model loaded.")

            # Set the value token ID for "1"
            # Load tokenizer (should be safe on any device now)
            tokenizer = AutoTokenizer.from_pretrained(self.value_model)
            self.value_token_id = tokenizer.encode("1", add_special_tokens=False)[0]
            print(f"Value token ID for '1': {self.value_token_id}")

        @app.post(self.endpoint)
        async def predict_value(request: ValueRequest):
            try:
                if not app.state.value_llm:
                    print("ERROR: Model not initialized")
                    raise HTTPException(status_code=500, detail="Model not initialized")

                # Get value predictions
                texts = [f"{q}\n{s}" for q, s in request.questions_and_states]

                values = self._predict_value(app.state.value_llm, texts)

                return {"results": values}
            except Exception as e:
                print(f"ERROR in predict_value: {str(e)}")
                print(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    def _predict_value(self, llm, texts):
        """Predict value scores based on first token logprobs"""
        try:
            # Configure sampling parameters
            sampling_params = SamplingParams(max_tokens=1, logprobs=20)
            outputs = llm.generate(texts, sampling_params, use_tqdm=False)

            results = []
            for i, output in enumerate(outputs):
                try:
                    log_dict = output.outputs[0].logprobs[0]
                    if self.value_token_id in log_dict:
                        logprob = log_dict[self.value_token_id].logprob
                        value = math.exp(logprob)
                    else:
                        value = 0.0
                    results.append(value)
                except Exception as e:
                    print(f"ERROR processing output {i+1}: {str(e)}")
                    print(f"Output structure: {output}")
                    raise

            return results
        except Exception as e:
            print(f"ERROR in _predict_value: {str(e)}")
            print(traceback.format_exc())
            raise

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
    parser.add_argument("--host", required=True, help="Host address to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--endpoint", required=True, help="API endpoint path")
    args = parser.parse_args()

    server = ValueServer(
        args.value_model,
        args.host,
        args.port,
        args.endpoint
    )
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")
