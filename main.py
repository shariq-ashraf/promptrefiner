from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load model and tokenizer
model_name = "alibaba-pai/Qwen2-1.5B-Instruct-Refine"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Define request model
class UserPrompt(BaseModel):
    prompt: str

@app.post("/refine-prompt")
async def refine_prompt(user_prompt: UserPrompt):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    system_prompt = (
        "You are a professional prompt refiner. Your task is to take a user's prompt and improve it by correcting "
        "grammar, spelling, and sentence structure. Enhance fluency, clarity, and natural tone without changing "
        "the original intent. Add slight descriptive detail only if it improves understanding. Do not over-extend, "
        "repeat, or remove any important information. Return only the refined prompt, nothing else."
    )

    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt.prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    refined_prompt = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return {"refined_prompt": refined_prompt}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
