# inference/loader.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer