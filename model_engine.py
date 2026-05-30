"""
model_engine.py — Encapsulates OpenVINO model loading and inference.

Wraps the OVModelForCausalLM + AutoTokenizer lifecycle into a clean,
reusable class that the Flask app consumes.
"""

import os
import warnings
import threading
from dotenv import load_dotenv

load_dotenv()

# Suppress the Mistral/Qwen regex tokenization warning
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")


class ModelEngine:
    """Manages model loading and text generation for a single OpenVINO model."""

    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "./qwen-0.5b-ov")
        self.device = os.getenv("DEVICE", "GPU")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.max_history = int(os.getenv("MAX_HISTORY", "20"))

        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def model_name(self):
        """Returns a human-readable model name derived from the path."""
        return os.path.basename(self.model_path.rstrip("/"))

    def load(self):
        """Load the model and tokenizer. Call once at startup."""
        if self._loaded:
            return

        # Import here so the app can start even if openvino isn't installed
        # (useful for front-end-only development/testing)
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        print(f"[ModelEngine] Loading {self.model_path} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = OVModelForCausalLM.from_pretrained(
            self.model_path, device=self.device, compile=True
        )
        self._loaded = True
        print(f"[ModelEngine] Model loaded successfully.")

    def is_loaded(self):
        """Check if the model is ready for inference."""
        return self._loaded

    def get_config(self):
        """Return current configuration as a dict (safe for JSON)."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "max_history": self.max_history,
            "loaded": self._loaded,
        }

    def generate(self, history: list[dict]) -> str:
        """
        Generate a response given a conversation history.

        Args:
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            The assistant's response text.
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Trim history to the configured maximum
        trimmed = history[-self.max_history :]

        with self._lock:
            text = self.tokenizer.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")

            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

        return response

    def generate_stream(self, history: list[dict]):
        """
        Stream-generate a response token by token.

        Yields chunks of text as they are generated.

        Args:
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Yields:
            str: Chunks of generated text.
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        from transformers import TextIteratorStreamer

        trimmed = history[-self.max_history :]

        with self._lock:
            text = self.tokenizer.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "streamer": streamer,
            }

            # Run generation in a separate thread so we can yield from the streamer
            thread = threading.Thread(
                target=self.model.generate, kwargs=generation_kwargs
            )
            thread.start()

            for chunk in streamer:
                if chunk:
                    yield chunk

            thread.join()
