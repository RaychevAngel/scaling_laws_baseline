from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams
import torch

class RequestData(BaseModel):
    messages: list

app = FastAPI()

# Initialize the model with proper Hugging Face model loading
model = LLM(
    model="/scaling_laws_baseline/models/iter_1/135_135/value/batch_900/full_model.pt",
    trust_remote_code=True,
    dtype="float16",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    model_format="pt"  # Specify PyTorch format
)

@app.post("/predict")
async def predict_value(data: RequestData):
    if not data.messages:
        raise HTTPException(status_code=400, detail="Conversation list is empty")
    
    state = [msg.get("content", "").strip() for msg in data.messages]
    if not state:
        raise HTTPException(status_code=400, detail="No state found in conversation")
    
    try:
        # Use vLLM's generate with custom sampling params
        sampling_params = SamplingParams(
            temperature=0.0,  # For deterministic output
            max_tokens=1,     # Since we're just getting a value
            stop=None
        )
        
        outputs = model.generate(state, sampling_params)
        values = [output.outputs[0].text for output in outputs]
        
        return {"value": values}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)