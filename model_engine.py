"""
model_engine.py — Encapsulates OpenVINO model loading and inference.

Wraps the OVModelForCausalLM + AutoTokenizer lifecycle into a clean,
reusable class that the Flask app consumes.
"""

import os
import gc
import warnings
import threading
from dotenv import load_dotenv

load_dotenv()

# Suppress the Mistral/Qwen regex tokenization warning
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

# GPU OOM error patterns to detect and handle gracefully
_GPU_OOM_PATTERNS = [
    "exceed_allocatable_mem_size",
    "Exceeded max size of memory object allocation",
    "out of memory",
]


def _is_gpu_oom_error(error: Exception) -> bool:
    """Check if an exception is a GPU out-of-memory error."""
    msg = str(error).lower()
    return any(pattern.lower() in msg for pattern in _GPU_OOM_PATTERNS)


class ModelEngine:
    """Manages model loading and text generation for a single OpenVINO model."""

    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "./qwen-0.5b-ov")
        self.device = os.getenv("DEVICE", "AUTO")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.max_history = int(os.getenv("MAX_HISTORY", "20"))
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "1024"))
        self.generation_timeout = int(os.getenv("GENERATION_TIMEOUT", "120"))

        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._loaded = False
        self._active_device = None  # Track which device is actually in use

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Try loading with requested device, fall back to CPU on failure
        device_order = self._get_device_fallback_order()

        for device in device_order:
            try:
                print(f"[ModelEngine] Loading {self.model_path} onto {device}...")
                self.model = OVModelForCausalLM.from_pretrained(
                    self.model_path, device=device, compile=True
                )
                self._active_device = device
                self._loaded = True
                print(f"[ModelEngine] Model loaded successfully on {device}.")
                return
            except Exception as e:
                if _is_gpu_oom_error(e) and device != "CPU":
                    print(f"[ModelEngine] GPU memory error on {device}: {e}")
                    print(f"[ModelEngine] Falling back to next device...")
                    # Clean up failed load
                    self.model = None
                    gc.collect()
                    continue
                elif device != "CPU" and "GPU" in device.upper():
                    print(f"[ModelEngine] Failed to load on {device}: {e}")
                    print(f"[ModelEngine] Falling back to next device...")
                    self.model = None
                    gc.collect()
                    continue
                else:
                    raise

        raise RuntimeError(
            f"Failed to load model on any device. Tried: {device_order}"
        )

    def _get_device_fallback_order(self) -> list[str]:
        """Determine device fallback order based on configured DEVICE."""
        device = self.device.upper()
        if device == "AUTO":
            return ["GPU", "CPU"]
        elif device == "GPU":
            return ["GPU", "CPU"]
        elif device == "CPU":
            return ["CPU"]
        else:
            return [device, "CPU"]

    def is_loaded(self):
        """Check if the model is ready for inference."""
        return self._loaded

    def get_config(self):
        """Return current configuration as a dict (safe for JSON)."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self._active_device or self.device,
            "max_new_tokens": self.max_new_tokens,
            "max_history": self.max_history,
            "max_input_tokens": self.max_input_tokens,
            "loaded": self._loaded,
        }

    def _trim_history_to_fit(self, history: list[dict]) -> list[dict]:
        """
        Trim conversation history so the tokenized input fits within
        max_input_tokens. Removes the oldest messages first (but always
        keeps the most recent user message).

        Args:
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            Trimmed history that fits within the token budget.
        """
        # Start with the configured max_history limit
        trimmed = history[-self.max_history:]

        while len(trimmed) > 0:
            text = self.tokenizer.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=True
            )
            token_count = len(self.tokenizer.encode(text))

            if token_count <= self.max_input_tokens:
                return trimmed

            # If only 1 message left and it's still too long, we have to keep it
            # but warn — the model will handle truncation
            if len(trimmed) <= 1:
                print(
                    f"[ModelEngine] WARNING: Single message has {token_count} tokens "
                    f"(limit: {self.max_input_tokens}). Proceeding anyway."
                )
                return trimmed

            # Remove the oldest message
            print(
                f"[ModelEngine] Input too long ({token_count} tokens > "
                f"{self.max_input_tokens}). Trimming oldest message..."
            )
            trimmed = trimmed[1:]

        return trimmed

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

        # Trim history to fit within token budget
        trimmed = self._trim_history_to_fit(history)

        with self._lock:
            text = self.tokenizer.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")

            try:
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )
            except RuntimeError as e:
                if _is_gpu_oom_error(e):
                    raise RuntimeError(
                        "GPU ran out of memory. Try a shorter message, start a new "
                        "conversation, or switch to CPU in the .env configuration."
                    ) from e
                raise

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
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

        # Trim history to fit within token budget
        trimmed = self._trim_history_to_fit(history)

        # Hold the lock only during tokenization and thread start,
        # NOT during the entire streaming iteration
        self._lock.acquire()
        try:
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

            generation_error = [None]  # Mutable container to capture thread errors

            def _generate_with_error_handling():
                try:
                    self.model.generate(**generation_kwargs)
                except Exception as e:
                    generation_error[0] = e
                    print(f"[ModelEngine] Generation error: {e}")

            # Run generation in a separate thread so we can yield from the streamer
            thread = threading.Thread(
                target=_generate_with_error_handling,
                daemon=True,
            )
            thread.start()
        finally:
            # Release the lock immediately after starting generation
            # This allows other requests to be queued rather than blocked
            self._lock.release()

        # Stream tokens outside the lock
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        except Exception as e:
            if _is_gpu_oom_error(e):
                yield "\n\n⚠️ GPU ran out of memory. Please start a new conversation or switch to CPU."
            else:
                raise

        # Wait for the generation thread to finish with a timeout
        thread.join(timeout=self.generation_timeout)

        if thread.is_alive():
            print(
                f"[ModelEngine] WARNING: Generation thread did not finish within "
                f"{self.generation_timeout}s timeout."
            )

        # Check if the generation thread encountered an error
        if generation_error[0] is not None:
            err = generation_error[0]
            if _is_gpu_oom_error(err):
                yield "\n\n⚠️ GPU ran out of memory. Please start a new conversation or switch to CPU."
            else:
                raise RuntimeError(f"Generation failed: {err}") from err

        # Encourage garbage collection after generation to free GPU memory
        gc.collect()
