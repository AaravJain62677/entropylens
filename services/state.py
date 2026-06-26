# services/state.py

from inference.loader import load_model_and_tokenizer

class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.active_model_name: str | None = None

    def load(self, model_name: str, model_path: str, device: str = "auto"):
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, device)
        self.active_model_name = model_name

    def is_loaded(self) -> bool:
        return self.model is not None

# singleton used across CLI + API
model_state = ModelState()