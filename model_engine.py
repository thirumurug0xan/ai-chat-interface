"""
model_engine.py — Encapsulates OpenVINO model loading and inference.

Wraps the OVModelForCausalLM + AutoTokenizer lifecycle into a clean,
reusable class that the Flask app consumes.
"""

import os
import gc
import time
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

    # Map user-facing device names to OpenVINO device strings
    _DEVICE_MAP = {
        "AUTO": "AUTO",
        "GPU": "GPU",
        "CPU": "CPU",
        "XPU": "HETERO:GPU,CPU",
    }

    # All user-facing device options
    AVAILABLE_DEVICES = ["AUTO", "GPU", "CPU", "XPU"]

    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "./qwen-0.5b-ov")
        self.model_path = os.path.expanduser(model_path.strip()) if model_path else "./qwen-0.5b-ov"
        self.device = os.getenv("DEVICE", "AUTO")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.max_history = int(os.getenv("MAX_HISTORY", "20"))
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "1024"))
        self.generation_timeout = int(os.getenv("GENERATION_TIMEOUT", "120"))

        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._loaded = False
        self._active_device = None      # Track which device is actually in use
        self._requested_device = self.device  # Track what the user requested
        self._switching = False         # True while a device switch is in progress

    @property
    def model_name(self):
        """Returns a human-readable model name derived from the path."""
        return os.path.basename(self.model_path.rstrip("/"))

    def load(self):
        """Load the model and tokenizer. Call once at startup."""
        if self._loaded:
            return

        # Check if the path is explicitly intended to be a local path
        # but the directory does not exist. This avoids confusing Hugging Face Hub
        # validation errors when a local path is incorrect.
        is_local = (
            self.model_path.startswith(".") or
            self.model_path.startswith("/") or
            self.model_path.startswith("~") or
            "\\" in self.model_path or
            ".." in self.model_path or
            os.path.isdir(self.model_path)
        )
        if is_local and not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Local model directory does not exist: '{self.model_path}'. "
                f"Please check your path configuration."
            )

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
                elif device != "CPU" and ("GPU" in device.upper() or "HETERO" in device.upper()):
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

    def _unload(self):
        """Unload the current model and free resources."""
        print("[ModelEngine] Unloading model...")
        self.model = None
        self._loaded = False
        self._active_device = None
        gc.collect()
        print("[ModelEngine] Model unloaded.")

    def switch_model(self, new_model_path: str) -> dict:
        """
        Switch to a different model at runtime.

        Unloads the current model, updates the model path,
        and reloads. Falls back to the previous model on failure.

        Args:
            new_model_path: Path to the new model directory.

        Returns:
            Dict with keys: success, model_name, model_path, message.
        """
        new_model_path = os.path.abspath(new_model_path.strip())
        if not os.path.isdir(new_model_path):
            return {
                "success": False,
                "model_name": self.model_name,
                "model_path": self.model_path,
                "message": f"Model directory does not exist: {new_model_path}",
            }

        if self._switching:
            return {
                "success": False,
                "model_name": self.model_name,
                "model_path": self.model_path,
                "message": "A device or model switch is already in progress. Please wait.",
            }

        previous_path = self.model_path
        previous_loaded = self._loaded

        self._switching = True
        try:
            with self._lock:
                self._unload()
                self.model_path = new_model_path
                self.tokenizer = None  # Force tokenizer reload
                
                try:
                    self.load()
                    return {
                        "success": True,
                        "model_name": self.model_name,
                        "model_path": self.model_path,
                        "message": f"Successfully loaded model: {self.model_name}",
                    }
                except Exception as e:
                    print(f"[ModelEngine] Failed to load new model from {new_model_path}: {e}")
                    # Try to roll back to the previous path
                    self.model_path = previous_path
                    self.tokenizer = None
                    try:
                        self.load()
                        return {
                            "success": False,
                            "model_name": self.model_name,
                            "model_path": self.model_path,
                            "message": f"Failed to load model from {new_model_path}: {str(e)}. Successfully reverted to previous model: {self.model_name}.",
                        }
                    except Exception as e2:
                        print(f"[ModelEngine] CRITICAL: Failed to reload previous model from {previous_path}: {e2}")
                        return {
                            "success": False,
                            "model_name": "None",
                            "model_path": "None",
                            "message": f"Failed to load model from {new_model_path} and failed to revert to previous model. Engine is offline.",
                        }
        finally:
            self._switching = False

    def switch_device(self, new_device: str) -> dict:
        """
        Switch the model to a different device at runtime.

        Unloads the current model, updates the device setting,
        and reloads. Falls back to the previous device on failure.

        Args:
            new_device: One of 'AUTO', 'GPU', 'CPU', 'CPU+GPU'.

        Returns:
            Dict with keys: success, active_device, requested_device, message.
        """
        new_device = new_device.upper().strip()
        if new_device not in self._DEVICE_MAP:
            return {
                "success": False,
                "active_device": self._active_device,
                "requested_device": new_device,
                "message": f"Unknown device: {new_device}. Valid options: {', '.join(self.AVAILABLE_DEVICES)}",
            }

        if self._switching:
            return {
                "success": False,
                "active_device": self._active_device,
                "requested_device": new_device,
                "message": "A device switch is already in progress. Please wait.",
            }

        previous_device = self.device
        previous_active = self._active_device

        self._switching = True
        try:
            with self._lock:
                # Unload current model
                self._unload()

                # Set the new device
                self.device = new_device
                self._requested_device = new_device

                # Attempt to load on the new device
                try:
                    self.load()
                    active = self._active_device

                    # Determine user-friendly label
                    resolved_label = self._get_friendly_device_name(active)
                    requested_label = new_device

                    if new_device == "AUTO":
                        msg = f"Switched to AUTO mode (resolved to {resolved_label})"
                    elif active != self._DEVICE_MAP.get(new_device, new_device):
                        # Loaded on a different device than requested (fallback)
                        msg = f"Failed to switch to {requested_label}, defaulting to {resolved_label}"
                    else:
                        msg = f"Switched to {resolved_label}"

                    return {
                        "success": True,
                        "active_device": active,
                        "requested_device": new_device,
                        "active_device_friendly": resolved_label,
                        "message": msg,
                    }

                except Exception as e:
                    print(f"[ModelEngine] Failed to switch to {new_device}: {e}")

                    # Try to fall back to previous device
                    self.device = previous_device
                    self._requested_device = previous_device

                    try:
                        self.load()
                        fallback_label = self._get_friendly_device_name(self._active_device)
                        return {
                            "success": False,
                            "active_device": self._active_device,
                            "requested_device": new_device,
                            "active_device_friendly": fallback_label,
                            "message": f"Failed to switch to {new_device}, defaulting to {fallback_label}",
                        }
                    except Exception as e2:
                        print(f"[ModelEngine] CRITICAL: Failed to reload on previous device {previous_device}: {e2}")
                        return {
                            "success": False,
                            "active_device": None,
                            "requested_device": new_device,
                            "active_device_friendly": "None",
                            "message": f"Failed to switch to {new_device} and failed to restore {previous_device}. Model is offline.",
                        }
        finally:
            self._switching = False

    def _get_friendly_device_name(self, ov_device: str) -> str:
        """Convert an OpenVINO device string to a user-friendly name."""
        if ov_device is None:
            return "None"
        upper = ov_device.upper()
        if "HETERO" in upper:
            return "XPU"
        return upper

    def _get_device_fallback_order(self) -> list[str]:
        """Determine device fallback order based on configured DEVICE."""
        device = self.device.upper().strip()
        if device == "AUTO":
            return ["GPU", "CPU"]
        elif device == "GPU":
            return ["GPU", "CPU"]
        elif device == "CPU":
            return ["CPU"]
        elif device == "XPU":
            return ["HETERO:GPU,CPU", "GPU", "CPU"]
        else:
            # Could be a raw OpenVINO string like HETERO:GPU,CPU
            ov_device = self._DEVICE_MAP.get(device, device)
            return [ov_device, "CPU"]

    def is_loaded(self):
        """Check if the model is ready for inference."""
        return self._loaded

    def get_physical_devices(self) -> list[str]:
        """Detect what hardware devices are actually present and supported by OpenVINO."""
        # AUTO and CPU are always supported
        devices = ["AUTO", "CPU"]
        try:
            from openvino import Core
            core = Core()
            ov_devices = core.available_devices
            ov_devices_upper = [d.upper() for d in ov_devices]
            if "GPU" in ov_devices_upper:
                devices.append("GPU")
                devices.append("XPU")  # XPU maps to HETERO:GPU,CPU
            if "NPU" in ov_devices_upper:
                devices.append("NPU")
        except Exception as e:
            print(f"[ModelEngine] Warning: Could not dynamically detect devices: {e}")
            active = (self._active_device or self.device or "").upper()
            if "GPU" in active or "HETERO" in active:
                devices.extend(["GPU", "XPU"])
        return devices

    def get_config(self):
        """Return current configuration as a dict (safe for JSON)."""
        active = self._active_device or self.device
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": active,
            "device_friendly": self._get_friendly_device_name(active),
            "requested_device": self._requested_device,
            "available_devices": self.get_physical_devices(),
            "switching": self._switching,
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

    def count_tokens(self, history: list[dict]) -> int:
        """
        Count the number of tokens that the given message history
        would consume after applying the chat template.

        Args:
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            The token count.
        """
        if not self._loaded:
            return 0
        try:
            text = self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            return len(self.tokenizer.encode(text))
        except Exception:
            return 0

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

    def generate_stream(self, history: list[dict], max_new_tokens: int = None):
        """
        Stream-generate a response token by token.

        Yields chunks of text as they are generated.

        Args:
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.
            max_new_tokens: Override max new tokens for this generation.
                            Falls back to self.max_new_tokens if None.

        Yields:
            str: Chunks of generated text.
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        from transformers import TextIteratorStreamer

        # Determine effective max_new_tokens
        effective_max = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

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
                "max_new_tokens": effective_max,
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
        token_count = 0
        start_time = None
        try:
            for chunk in streamer:
                if chunk:
                    if start_time is None:
                        start_time = time.perf_counter()
                    # Count tokens in this chunk
                    token_count += len(self.tokenizer.encode(chunk, add_special_tokens=False))
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

        # Compute and yield generation metadata
        elapsed = time.perf_counter() - start_time if start_time else 0
        tps = round(token_count / elapsed, 1) if elapsed > 0 else 0
        yield {
            "__meta__": {
                "tokens": token_count,
                "elapsed_sec": round(elapsed, 2),
                "tokens_per_sec": tps,
            }
        }

        # Encourage garbage collection after generation to free GPU memory
        gc.collect()
