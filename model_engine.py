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

# Suppress the Mistral/Qwen regex tokenization warning and length warning
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer.*")

# GPU OOM error patterns to detect and handle gracefully
_GPU_OOM_PATTERNS = [
    "exceed_available_mem_size",
    "Exceeded max size of memory allocation",
    "out of memory",
]

_FATAL_GPU_CRASH_PATTERNS = [
    "exceed_allocatable_mem_size",
    "exceeded max size of memory object allocation",
]


def _is_gpu_oom_error(error: Exception) -> bool:
    """Check if an exception is a GPU out-of-memory error."""
    msg = str(error).lower()
    return any(pattern.lower() in msg for pattern in _GPU_OOM_PATTERNS)


def _is_fatal_gpu_crash_error(error: Exception) -> bool:
    """Check if an exception is a fatal GPU allocation/crash error requiring server restart."""
    msg = str(error).lower()
    return any(pattern.lower() in msg for pattern in _FATAL_GPU_CRASH_PATTERNS)


def _is_use_cache_error(error: Exception) -> bool:
    """Check if an exception is a use_cache incompatibility error."""
    msg = str(error)
    return "use_cache" in msg and "use_cache=False" in msg


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

        # Force low-VRAM mode and max token limit configurations
        self.force_low_vram = os.getenv("FORCE_LOW_VRAM_MODE", "False").lower() in ("true", "1", "yes")
        self.max_token_limit = int(os.getenv("MAX_TOKEN_LIMIT", "4096"))

        # Load KV-cache configuration from environment
        use_cache_env = os.getenv("USE_CACHE", "True").lower() in ("true", "1", "yes")
        self._use_cache = use_cache_env
        self.model_file = os.getenv("MODEL_FILE_NAME", "").strip() or None

        # Tokenizer options
        self.trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "True").lower() in ("true", "1", "yes")
        self.fix_mistral_regex = os.getenv("FIX_MISTRAL_REGEX", "True").lower() in ("true", "1", "yes")

        # Load custom ov_config
        self.ov_config = {}
        ov_config_str = os.getenv("OV_CONFIG", "").strip()
        if ov_config_str:
            try:
                import json
                self.ov_config = json.loads(ov_config_str)
            except Exception as e:
                print(f"[ModelEngine] Warning: Failed to parse OV_CONFIG JSON: {e}")
        else:
            # Fall back to default structure: PERFORMANCE_HINT
            perf_hint = os.getenv("OV_PERFORMANCE_HINT", "LATENCY").strip()
            if perf_hint:
                self.ov_config["PERFORMANCE_HINT"] = perf_hint

        # Apply KV cache precision from env if not already specified in ov_config
        if "KV_CACHE_PRECISION" not in self.ov_config:
            kv_prec = os.getenv("OV_KV_CACHE_PRECISION", "u8").strip()
            if kv_prec and kv_prec.upper() != "DEFAULT":
                self.ov_config["KV_CACHE_PRECISION"] = kv_prec

        if self.force_low_vram:
            print("[ModelEngine] FORCE_LOW_VRAM_MODE is enabled. Applying memory optimizations to ov_config.")
            self.ov_config.update({
                "KV_CACHE_PRECISION": "u8",
                "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                "PERFORMANCE_HINT": "LATENCY",
                "COMPUTE_HINT": "ECONOMY",
                "GPU_DISABLE_REORDER_CACHING": "YES"
            })

        self.kv_cache_precision = self.ov_config.get("KV_CACHE_PRECISION", "default").strip().lower()
        self._last_load_warnings = []   # Warnings from the last load operation

    @property
    def model_name(self):
        """Returns a human-readable model name derived from the path."""
        return os.path.basename(self.model_path.rstrip("/"))

    def load(self):
        """Load the model and tokenizer. Call once at startup."""
        if self._loaded:
            return

        self._last_load_warnings = []  # Reset warnings for this load attempt

        load_path = self.model_path

        # Check if the path is explicitly intended to be a local path
        # but the directory does not exist. This avoids confusing Hugging Face Hub
        # validation errors when a local path is incorrect.
        is_local = (
            load_path.startswith(".") or
            load_path.startswith("/") or
            load_path.startswith("~") or
            "\\" in load_path or
            ".." in load_path or
            os.path.isdir(load_path)
        )
        if is_local:
            if not os.path.isdir(load_path):
                raise FileNotFoundError(
                    f"Local model directory does not exist: '{load_path}'. "
                    f"Please check your path configuration."
                )
            # Must contain a config.json or an .xml file to be valid.
            has_local_xml = any(
                f.endswith(".xml") 
                for f in os.listdir(load_path) 
                if os.path.isfile(os.path.join(load_path, f))
            )
            has_config = os.path.isfile(os.path.join(load_path, "config.json"))
            if not has_local_xml and not has_config:
                raise FileNotFoundError(
                    f"Invalid model directory: '{load_path}'. "
                    f"A valid Hugging Face / Optimum model directory must contain a 'config.json' file or an '.xml' file."
                )

        # Import here so the app can start even if openvino isn't installed
        # (useful for front-end-only development/testing)
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        # Load tokenizer with parameters
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    load_path,
                    trust_remote_code=self.trust_remote_code,
                    fix_mistral_regex=self.fix_mistral_regex
                )
            except TypeError as e:
                if "fix_mistral_regex" in str(e):
                    print("[ModelEngine] fix_mistral_regex parameter not supported by AutoTokenizer. Retrying without it...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        load_path,
                        trust_remote_code=self.trust_remote_code
                    )
                else:
                    raise

        # Try loading with requested device, fall back to CPU on failure
        device_order = self._get_device_fallback_order()

        for device in device_order:
            # Prepare configuration for the specific device
            active_ov_config = self.ov_config.copy() if self.ov_config else {}
            
            # Apply dynamic KV cache precision configuration
            if self.kv_cache_precision and self.kv_cache_precision.lower() != "default":
                active_ov_config["KV_CACHE_PRECISION"] = self.kv_cache_precision.lower()
            else:
                active_ov_config.pop("KV_CACHE_PRECISION", None)
            
            # Check for GPU-specific configurations
            gpu_large_allocs = os.getenv("OV_GPU_ENABLE_LARGE_ALLOCATIONS", "True").lower() in ("true", "1", "yes")
            if "GPU" in device.upper() or "AUTO" in device.upper() or "HETERO" in device.upper():
                if gpu_large_allocs and "GPU_ENABLE_LARGE_ALLOCATIONS" not in active_ov_config:
                    active_ov_config["GPU_ENABLE_LARGE_ALLOCATIONS"] = "YES"
            else:
                # Strip GPU-only properties on CPU device to prevent driver/plugin exceptions
                active_ov_config.pop("GPU_ENABLE_LARGE_ALLOCATIONS", None)
                active_ov_config.pop("GPU_DISABLE_REORDER_CACHING", None)

            while True:
                try:
                    print(f"[ModelEngine] Loading {load_path} onto {device} with use_cache={self._use_cache} and ov_config={active_ov_config}...")
                    model_kwargs = {
                        "device": device,
                        "compile": True,
                        "use_cache": self._use_cache,
                    }
                    if active_ov_config:
                        model_kwargs["ov_config"] = active_ov_config
                    if self.model_file:
                        model_kwargs["file_name"] = self.model_file

                    self.model = OVModelForCausalLM.from_pretrained(
                        load_path, **model_kwargs
                    )
                    self._active_device = device
                    self._loaded = True
                    print(f"[ModelEngine] Model loaded successfully on {device}.")
                    return
                except Exception as e:
                    # Check if exception is due to an unsupported property/option
                    error_msg = str(e)
                    unsupported_property = None
                    
                    # Check if any configured key name is present in the error message
                    for key in list(active_ov_config.keys()):
                        if key in error_msg:
                            unsupported_property = key
                            break
                            
                    # Match words that look like property names (UPPERCASE with underscores)
                    if not unsupported_property:
                        import re
                        prop_matches = re.findall(r'\b[A-Z_]{3,}\b', error_msg)
                        for match in prop_matches:
                            if match in active_ov_config:
                                unsupported_property = match
                                break
                                
                    if unsupported_property:
                        print(f"[ModelEngine] Detected unsupported property: {unsupported_property}. Removing from ov_config and retrying...")
                        active_ov_config.pop(unsupported_property)
                        self.ov_config.pop(unsupported_property, None)
                        if unsupported_property == "KV_CACHE_PRECISION":
                            self.kv_cache_precision = "default"
                        continue  # Retry loading on the same device with the cleaned configuration
                        
                    # Handle use_cache incompatibility: retry with use_cache=False
                    if _is_use_cache_error(e) and self._use_cache:
                        print(f"[ModelEngine] use_cache=True not supported for this model. Retrying with use_cache=False...")
                        self._use_cache = False
                        self._last_load_warnings.append(
                            "This model was exported without KV-cache support. "
                            "Loaded with use_cache=False — generation may be slower."
                        )
                        self.model = None
                        gc.collect()
                        continue
                        
                    # If we got here, it's a device load failure (OOM or other)
                    if (_is_gpu_oom_error(e) or _is_fatal_gpu_crash_error(e)) and device != "CPU":
                        print(f"[ModelEngine] GPU memory error on {device}: {e}")
                        print(f"[ModelEngine] Falling back to next device...")
                        self.model = None
                        gc.collect()
                        break
                    elif device != "CPU" and ("GPU" in device.upper() or "HETERO" in device.upper()):
                        print(f"[ModelEngine] Failed to load on {device}: {e}")
                        print(f"[ModelEngine] Falling back to next device...")
                        self.model = None
                        gc.collect()
                        break
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

    def reload(self) -> dict:
        """Unload and reload the model on the current device to apply compile-time options."""
        if not self._loaded:
            return {"success": False, "message": "No model loaded to reload."}

        with self._lock:
            try:
                self._unload()
                self.load()
                return {
                    "success": True,
                    "model_name": self.model_name,
                    "model_path": self.model_path,
                    "message": f"Successfully reloaded model with KV cache precision: {self.kv_cache_precision}",
                    "warnings": list(self._last_load_warnings),
                }
            except Exception as e:
                print(f"[ModelEngine] Failed to reload model: {e}")
                # Try to load again to recover
                try:
                    self.load()
                    return {
                        "success": False,
                        "model_name": self.model_name,
                        "model_path": self.model_path,
                        "message": f"Reload failed: {str(e)}. Recovered with current settings.",
                        "warnings": list(self._last_load_warnings),
                    }
                except Exception as e2:
                    return {
                        "success": False,
                        "model_name": "None",
                        "model_path": "None",
                        "message": f"Reload failed: {str(e)} and recovery failed. Model is offline.",
                        "warnings": [],
                    }

    def switch_model(
        self,
        new_model_path: str,
        use_cache: bool = None,
        model_file: str = None,
        trust_remote_code: bool = None,
        fix_mistral_regex: bool = None,
        ov_config: dict = None,
        ov_performance_hint: str = None,
        ov_cache_dir: str = None,
    ) -> dict:
        """
        Switch to a different model at runtime.

        Unloads the current model, updates the model path and configuration,
        and reloads. Falls back to the previous model and settings on failure.

        Args:
            new_model_path: Path to the new model directory.
            use_cache: Optional override for KV-cache generation.
            model_file: Optional override for model XML file name.
            trust_remote_code: Optional override for trust_remote_code tokenizer parameter.
            fix_mistral_regex: Optional override for fix_mistral_regex tokenizer parameter.
            ov_config: Optional full custom OpenVINO configuration mapping.
            ov_performance_hint: Optional OpenVINO performance hint override.
            ov_cache_dir: Optional OpenVINO compiler cache directory override.

        Returns:
            Dict with keys: success, model_name, model_path, message, warnings.
        """
        new_model_path = os.path.abspath(new_model_path.strip())
        if not os.path.isdir(new_model_path):
            return {
                "success": False,
                "model_name": self.model_name,
                "model_path": self.model_path,
                "message": f"Model directory does not exist: {new_model_path}",
                "warnings": [],
            }

        if self._switching:
            return {
                "success": False,
                "model_name": self.model_name,
                "model_path": self.model_path,
                "message": "A device or model switch is already in progress. Please wait.",
                "warnings": [],
            }

        previous_path = self.model_path
        previous_loaded = self._loaded
        previous_use_cache = self._use_cache
        previous_model_file = self.model_file
        previous_trust_remote = self.trust_remote_code
        previous_fix_mistral = self.fix_mistral_regex
        previous_ov_config = self.ov_config

        self._switching = True
        try:
            with self._lock:
                self._unload()
                self.model_path = new_model_path
                self.tokenizer = None  # Force tokenizer reload
                
                # Apply new options if specified, otherwise fall back to environment defaults

                if use_cache is not None:
                    self._use_cache = use_cache
                else:
                    self._use_cache = os.getenv("USE_CACHE", "True").lower() in ("true", "1", "yes")

                if model_file is not None:
                    self.model_file = model_file.strip() or None
                else:
                    self.model_file = os.getenv("MODEL_FILE_NAME", "").strip() or None

                if trust_remote_code is not None:
                    self.trust_remote_code = trust_remote_code
                else:
                    self.trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "True").lower() in ("true", "1", "yes")

                if fix_mistral_regex is not None:
                    self.fix_mistral_regex = fix_mistral_regex
                else:
                    self.fix_mistral_regex = os.getenv("FIX_MISTRAL_REGEX", "True").lower() in ("true", "1", "yes")

                if ov_config is not None:
                    self.ov_config = ov_config
                elif ov_performance_hint is not None:
                    self.ov_config = {}
                    if ov_performance_hint:
                        self.ov_config["PERFORMANCE_HINT"] = ov_performance_hint
                else:
                    self.ov_config = {}
                    ov_config_str = os.getenv("OV_CONFIG", "").strip()
                    if ov_config_str:
                        try:
                            import json
                            self.ov_config = json.loads(ov_config_str)
                        except Exception as e:
                            print(f"[ModelEngine] Warning: Failed to parse OV_CONFIG JSON: {e}")
                    else:
                        perf_hint = os.getenv("OV_PERFORMANCE_HINT", "LATENCY").strip()
                        if perf_hint:
                            self.ov_config["PERFORMANCE_HINT"] = perf_hint

                # Apply KV cache precision from env if not already specified in ov_config
                if "KV_CACHE_PRECISION" not in self.ov_config:
                    kv_prec = os.getenv("OV_KV_CACHE_PRECISION", "u8").strip()
                    if kv_prec and kv_prec.upper() != "DEFAULT":
                        self.ov_config["KV_CACHE_PRECISION"] = kv_prec

                # Reload and apply Force low-VRAM settings
                self.force_low_vram = os.getenv("FORCE_LOW_VRAM_MODE", "False").lower() in ("true", "1", "yes")
                self.max_token_limit = int(os.getenv("MAX_TOKEN_LIMIT", "4096"))

                if self.force_low_vram:
                    print("[ModelEngine] FORCE_LOW_VRAM_MODE is enabled. Applying memory optimizations to ov_config.")
                    self.ov_config.update({
                        "KV_CACHE_PRECISION": "u8",
                        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                        "PERFORMANCE_HINT": "LATENCY",
                        "COMPUTE_HINT": "ECONOMY",
                        "GPU_DISABLE_REORDER_CACHING": "YES"
                    })
                
                self.kv_cache_precision = self.ov_config.get("KV_CACHE_PRECISION", "default").strip().lower()
                
                try:
                    self.load()
                    return {
                        "success": True,
                        "model_name": self.model_name,
                        "model_path": self.model_path,
                        "message": f"Successfully loaded model: {self.model_name}",
                        "warnings": list(self._last_load_warnings),
                    }
                except Exception as e:
                    print(f"[ModelEngine] Failed to load new model from {new_model_path}: {e}")
                    # Try to roll back to the previous path and configuration
                    self.model_path = previous_path
                    self.tokenizer = None
                    self._use_cache = previous_use_cache
                    self.model_file = previous_model_file
                    self.trust_remote_code = previous_trust_remote
                    self.fix_mistral_regex = previous_fix_mistral
                    self.ov_config = previous_ov_config
                    try:
                        self.load()
                        return {
                            "success": False,
                            "model_name": self.model_name,
                            "model_path": self.model_path,
                            "message": f"Failed to load model from {new_model_path}: {str(e)}. Successfully reverted to previous model: {self.model_name}.",
                            "warnings": list(self._last_load_warnings),
                        }
                    except Exception as e2:
                        print(f"[ModelEngine] CRITICAL: Failed to reload previous model from {previous_path}: {e2}")
                        return {
                            "success": False,
                            "model_name": "None",
                            "model_path": "None",
                            "message": f"Failed to load model from {new_model_path} and failed to revert to previous model. Engine is offline.",
                            "warnings": [],
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

    def get_model_max_sequence_length(self) -> int:
        """Retrieve the maximum sequence length supported by the model config or tokenizer."""
        # Try model config first
        if self.model and hasattr(self.model, "config"):
            config = self.model.config
            for attr in ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"):
                val = getattr(config, attr, None)
                if isinstance(val, int) and 0 < val < 1000000:
                    return val
        
        # Try tokenizer next
        if self.tokenizer and hasattr(self.tokenizer, "model_max_length"):
            val = self.tokenizer.model_max_length
            if isinstance(val, int) and 0 < val < 1000000:
                return val
                
        return 4096  # Default fallback if unknown

    @property
    def effective_max_input_tokens(self) -> int:
        """Return the effective maximum input tokens capped by model's physical limit."""
        if not self._loaded:
            return self.max_input_tokens
        return min(self.max_input_tokens, self.get_model_max_sequence_length())

    def get_config(self):
        """Return current configuration as a dict (safe for JSON)."""
        active = self._active_device or self.device
        model_limit = self.get_model_max_sequence_length() if self._loaded else 4096
        
        # Extract architectural properties for memory warning calculations
        num_layers = 32
        num_kv_heads = 8
        head_dim = 128
        kv_precision = 2  # Default FP16 (2 bytes)
        
        if self.model and hasattr(self.model, "config"):
            config = self.model.config
            
            # Layers
            for attr in ("num_hidden_layers", "n_layers", "num_layers"):
                val = getattr(config, attr, None)
                if isinstance(val, int) and val > 0:
                    num_layers = val
                    break
            
            # KV Heads
            num_kv_heads = getattr(config, "num_key_value_heads", None)
            if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
                num_kv_heads = getattr(config, "num_attention_heads", 8)
                
            # Head Dimension
            hidden_size = getattr(config, "hidden_size", None)
            num_attention_heads = getattr(config, "num_attention_heads", None)
            if isinstance(hidden_size, int) and isinstance(num_attention_heads, int) and num_attention_heads > 0:
                head_dim = hidden_size // num_attention_heads
            else:
                head_dim = getattr(config, "head_dim", 128)
                
        # Check if KV cache precision is set to u8 or similar
        if self.kv_cache_precision:
            prec = self.kv_cache_precision.lower()
            if "u8" in prec or "int8" in prec or "i8" in prec:
                kv_precision = 1
            elif "f32" in prec or "fp32" in prec:
                kv_precision = 4
            else:
                kv_precision = 2
        elif self.ov_config and isinstance(self.ov_config, dict):
            prec = self.ov_config.get("KV_CACHE_PRECISION", "").lower()
            if "u8" in prec or "int8" in prec or "i8" in prec:
                kv_precision = 1
            elif "f32" in prec or "fp32" in prec:
                kv_precision = 4
            else:
                kv_precision = 2
                
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
            "effective_max_input_tokens": self.effective_max_input_tokens,
            "model_max_sequence_length": model_limit,
            "model_architecture": {
                "num_layers": num_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "kv_precision_bytes": kv_precision,
            },
            "loaded": self._loaded,
            "use_cache": self._use_cache,
            "model_file": self.model_file,
            "trust_remote_code": self.trust_remote_code,
            "fix_mistral_regex": self.fix_mistral_regex,
            "kv_cache_precision": self.kv_cache_precision,
            "ov_config": self.ov_config,
        }

    def _trim_history_to_fit(self, history: list[dict]) -> list[dict]:
        """
        Trim conversation history so the tokenized input fits within
        the effective max_input_tokens. Removes the oldest messages first (but always
        keeps the most recent user message).

        Args:
            history: List of {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            Trimmed history that fits within the token budget.
        """
        effective_max = self.effective_max_input_tokens
        # Start with the configured max_history limit, but scale it up if the context window is large
        history_limit = max(self.max_history, 1000) if effective_max > 4096 else self.max_history
        trimmed = history[-history_limit:]

        while len(trimmed) > 0:
            text = self.tokenizer.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=True
            )
            token_count = len(self.tokenizer.encode(text, verbose=False))

            if token_count <= effective_max:
                return trimmed

            # If only 1 message left and it's still too long, we have to keep it
            # but warn — the model will handle truncation
            if len(trimmed) <= 1:
                print(
                    f"[ModelEngine] WARNING: Single message has {token_count} tokens "
                    f"(limit: {effective_max}). Proceeding anyway."
                )
                return trimmed

            # Remove the oldest message
            print(
                f"[ModelEngine] Input too long ({token_count} tokens > "
                f"{effective_max}). Trimming oldest message..."
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

    def _enforce_strict_truncation(self, inputs):
        """Enforce strict truncation guardrail by keeping only the newest max_token_limit tokens."""
        seq_len = inputs["input_ids"].shape[-1]
        if seq_len > self.max_token_limit:
            print(
                f"[ModelEngine] WARNING: Input token length ({seq_len}) exceeds "
                f"MAX_TOKEN_LIMIT ({self.max_token_limit}). Forcing hard truncation to keep only the newest "
                f"tokens."
            )
            # Truncate input_ids and other keys along sequence dimension (last dimension)
            inputs["input_ids"] = inputs["input_ids"][:, -self.max_token_limit:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_token_limit:]
            if "position_ids" in inputs:
                inputs["position_ids"] = inputs["position_ids"][:, -self.max_token_limit:]
            for key in list(inputs.keys()):
                if key not in ["input_ids", "attention_mask", "position_ids"]:
                    if hasattr(inputs[key], "shape") and len(inputs[key].shape) >= 2 and inputs[key].shape[-1] == seq_len:
                        inputs[key] = inputs[key][..., -self.max_token_limit:]
        return inputs

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
            inputs = self._enforce_strict_truncation(inputs)

            try:
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )
            except RuntimeError as e:
                if _is_fatal_gpu_crash_error(e):
                    raise RuntimeError(
                        "The model crashed due to exceeding the maximum allocatable memory size on the GPU. Please restart the server."
                    ) from e
                if _is_gpu_oom_error(e):
                    raise RuntimeError(
                        "GPU ran out of memory. Try reducing the 'Max Input Tokens' (context window) "
                        "in settings, starting a new conversation, or switching the device to CPU."
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

        # Hold the lock during the entire generation and streaming process
        # to ensure that no two threads execute inference/generation concurrently.
        self._lock.acquire()
        try:
            text = self.tokenizer.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = self._enforce_strict_truncation(inputs)

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

            # Stream tokens
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
                if _is_fatal_gpu_crash_error(e):
                    yield "\n\n🚨 ERROR: The model crashed due to exceeding the maximum allocatable memory size on the GPU. Please restart the server."
                elif _is_gpu_oom_error(e):
                    yield "\n\n⚠️ GPU ran out of memory. Try reducing the 'Max Input Tokens' (context window) in settings, starting a new conversation, or switching the device to CPU."
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
                if _is_fatal_gpu_crash_error(err):
                    yield "\n\n🚨 ERROR: The model crashed due to exceeding the maximum allocatable memory size on the GPU. Please restart the server."
                elif _is_gpu_oom_error(err):
                    yield "\n\n⚠️ GPU ran out of memory. Try reducing the 'Max Input Tokens' (context window) in settings, starting a new conversation, or switching the device to CPU."
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
        finally:
            if "thread" in locals() and thread.is_alive():
                thread.join()
            self._lock.release()

        # Encourage garbage collection after generation to free GPU memory
        gc.collect()


class MultiModelManager:
    """Manages multiple loaded ModelEngine instances and provides active model switching."""

    def __init__(self):
        self._engines = {}  # model_path -> ModelEngine
        self._active_path = None  # path of the currently active model
        self._mgr_lock = threading.Lock()

    @property
    def active_engine(self) -> ModelEngine:
        """Get the currently active ModelEngine, or None."""
        if self._active_path and self._active_path in self._engines:
            return self._engines[self._active_path]
        return None

    def get_loaded_models(self) -> list[dict]:
        """Return info about all loaded models."""
        models = []
        for path, eng in self._engines.items():
            models.append({
                "model_name": eng.model_name,
                "model_path": eng.model_path,
                "is_active": (path == self._active_path),
                "device": eng._active_device or eng.device,
                "device_friendly": eng._get_friendly_device_name(eng._active_device or eng.device),
                "loaded": eng.is_loaded(),
            })
        return models

    def load_model(self, model_path: str, device: str = None, make_active: bool = True, **kwargs) -> dict:
        """Load a new model and optionally make it active.

        If the model is already loaded, just activate it.
        kwargs are passed to ModelEngine configuration (use_cache, model_file, etc.)
        """
        model_path = os.path.abspath(model_path.strip())

        # If already loaded, just activate
        if model_path in self._engines and self._engines[model_path].is_loaded():
            if make_active:
                self._active_path = model_path
            eng = self._engines[model_path]
            return {
                "success": True,
                "model_name": eng.model_name,
                "model_path": model_path,
                "message": f"Model already loaded: {eng.model_name}. Activated.",
                "warnings": [],
                "loaded_models": self.get_loaded_models(),
            }

        if not os.path.isdir(model_path):
            return {
                "success": False,
                "model_name": "",
                "model_path": model_path,
                "message": f"Model directory does not exist: {model_path}",
                "warnings": [],
                "loaded_models": self.get_loaded_models(),
            }

        # Create a new engine
        eng = ModelEngine()
        eng.model_path = model_path
        if device:
            eng.device = device
            eng._requested_device = device

        # Apply optional kwargs
        if 'use_cache' in kwargs and kwargs['use_cache'] is not None:
            eng._use_cache = kwargs['use_cache']
        if 'model_file' in kwargs and kwargs['model_file'] is not None:
            eng.model_file = kwargs['model_file'] or None
        if 'trust_remote_code' in kwargs and kwargs['trust_remote_code'] is not None:
            eng.trust_remote_code = kwargs['trust_remote_code']
        if 'fix_mistral_regex' in kwargs and kwargs['fix_mistral_regex'] is not None:
            eng.fix_mistral_regex = kwargs['fix_mistral_regex']
        if 'ov_performance_hint' in kwargs and kwargs['ov_performance_hint']:
            eng.ov_config["PERFORMANCE_HINT"] = kwargs['ov_performance_hint']

        try:
            eng.load()
            self._engines[model_path] = eng
            if make_active:
                self._active_path = model_path
            return {
                "success": True,
                "model_name": eng.model_name,
                "model_path": model_path,
                "message": f"Successfully loaded model: {eng.model_name}",
                "warnings": list(eng._last_load_warnings),
                "loaded_models": self.get_loaded_models(),
            }
        except Exception as e:
            # Check if this is a GPU memory/OOM error. If so, and we have other models loaded,
            # try to unload all other models to free memory and retry loading.
            if (_is_gpu_oom_error(e) or _is_fatal_gpu_crash_error(e)) and len(self._engines) > 0:
                print(f"[MultiModelManager] GPU OOM during load. Unloading other models to free memory...")
                # Create a list of paths to unload (all except the current one)
                other_paths = [p for p in list(self._engines.keys()) if p != model_path]
                for path in other_paths:
                    try:
                        self.unload_model(path)
                    except Exception as unload_err:
                        print(f"[MultiModelManager] Failed to unload {path}: {unload_err}")
                
                # Force garbage collection
                gc.collect()
                time.sleep(1.0)  # Give GPU driver a moment to reclaim memory
                
                # Retry loading
                try:
                    print(f"[MultiModelManager] Retrying load of {eng.model_name} after freeing memory...")
                    eng.load()
                    self._engines[model_path] = eng
                    if make_active:
                        self._active_path = model_path
                    return {
                        "success": True,
                        "model_name": eng.model_name,
                        "model_path": model_path,
                        "message": f"Successfully loaded model: {eng.model_name} (unloaded other models to free GPU memory)",
                        "warnings": list(eng._last_load_warnings),
                        "loaded_models": self.get_loaded_models(),
                    }
                except Exception as retry_e:
                    e = retry_e  # Update exception for the fallback message below

            msg = str(e)
            if _is_fatal_gpu_crash_error(e):
                msg = (
                    "The model crashed due to exceeding the maximum allocatable memory size on the GPU. Please restart the server."
                )
            elif _is_gpu_oom_error(e):
                msg = (
                    "GPU ran out of memory while loading the model. "
                    "Try reducing the context window/history size, or switch the device to CPU."
                )
            return {
                "success": False,
                "model_name": "",
                "model_path": model_path,
                "message": f"Failed to load model: {msg}",
                "warnings": [],
                "loaded_models": self.get_loaded_models(),
            }

    def activate_model(self, model_path: str) -> dict:
        """Switch the active model to an already-loaded model."""
        model_path = os.path.abspath(model_path.strip())
        if model_path not in self._engines:
            return {
                "success": False,
                "message": f"Model not loaded: {model_path}",
                "loaded_models": self.get_loaded_models(),
            }
        eng = self._engines[model_path]
        if not eng.is_loaded():
            return {
                "success": False,
                "message": f"Model is not in loaded state: {eng.model_name}",
                "loaded_models": self.get_loaded_models(),
            }
        self._active_path = model_path
        return {
            "success": True,
            "model_name": eng.model_name,
            "model_path": model_path,
            "message": f"Switched to: {eng.model_name}",
            "loaded_models": self.get_loaded_models(),
        }

    def unload_model(self, model_path: str) -> dict:
        """Unload a specific model and free its resources."""
        model_path = os.path.abspath(model_path.strip())
        if model_path not in self._engines:
            return {
                "success": False,
                "message": f"Model not found: {model_path}",
                "loaded_models": self.get_loaded_models(),
            }

        eng = self._engines.pop(model_path)
        eng._unload()

        # If we unloaded the active model, pick another if available
        if self._active_path == model_path:
            if self._engines:
                self._active_path = next(iter(self._engines))
            else:
                self._active_path = None

        return {
            "success": True,
            "message": f"Unloaded model: {eng.model_name}",
            "loaded_models": self.get_loaded_models(),
        }

    # Delegate common operations to the active engine
    def is_loaded(self):
        eng = self.active_engine
        return eng.is_loaded() if eng else False

    def get_config(self):
        eng = self.active_engine
        if eng:
            config = eng.get_config()
            config["loaded_models"] = self.get_loaded_models()
            return config
        return {
            "model_name": "No model loaded",
            "model_path": "",
            "device": "—",
            "device_friendly": "—",
            "requested_device": os.getenv("DEVICE", "AUTO").upper(),
            "kv_cache_precision": os.getenv("OV_KV_CACHE_PRECISION", "u8").strip().lower(),
            "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "512")),
            "max_input_tokens": int(os.getenv("MAX_INPUT_TOKENS", "1024")),
            "loaded": False,
            "loaded_models": [],
        }

    @property
    def model_name(self):
        eng = self.active_engine
        return eng.model_name if eng else "No model loaded"

    @property
    def model_path(self):
        eng = self.active_engine
        return eng.model_path if eng else ""

    @property
    def _active_device(self):
        eng = self.active_engine
        return eng._active_device if eng else None

    @property
    def _lock(self):
        eng = self.active_engine
        return eng._lock if eng else self._mgr_lock

    @property
    def _switching(self):
        eng = self.active_engine
        return eng._switching if eng else False

    def count_tokens(self, history):
        eng = self.active_engine
        return eng.count_tokens(history) if eng else 0

    def generate(self, history):
        eng = self.active_engine
        if not eng:
            raise RuntimeError("No model is loaded.")
        return eng.generate(history)

    def generate_stream(self, history, max_new_tokens=None):
        eng = self.active_engine
        if not eng:
            raise RuntimeError("No model is loaded.")
        return eng.generate_stream(history, max_new_tokens=max_new_tokens)

    def switch_device(self, new_device):
        eng = self.active_engine
        if not eng:
            return {"success": False, "message": "No model loaded"}
        return eng.switch_device(new_device)

    @property
    def max_new_tokens(self):
        eng = self.active_engine
        return eng.max_new_tokens if eng else 512

    @max_new_tokens.setter
    def max_new_tokens(self, val):
        eng = self.active_engine
        if eng:
            eng.max_new_tokens = val

    @property
    def max_input_tokens(self):
        eng = self.active_engine
        return eng.max_input_tokens if eng else 1024

    @max_input_tokens.setter
    def max_input_tokens(self, val):
        eng = self.active_engine
        if eng:
            eng.max_input_tokens = val

    @property
    def effective_max_input_tokens(self):
        eng = self.active_engine
        return eng.effective_max_input_tokens if eng else 1024

    @property
    def _requested_device(self):
        eng = self.active_engine
        return eng._requested_device if eng else "AUTO"

    @_requested_device.setter
    def _requested_device(self, val):
        eng = self.active_engine
        if eng:
            eng._requested_device = val

    @property
    def kv_cache_precision(self):
        eng = self.active_engine
        return eng.kv_cache_precision if eng else "default"

    @kv_cache_precision.setter
    def kv_cache_precision(self, val):
        eng = self.active_engine
        if eng:
            eng.kv_cache_precision = val

    def reload(self) -> dict:
        """Reload the active model engine."""
        eng = self.active_engine
        if not eng:
            return {"success": False, "message": "No active model engine to reload."}
        return eng.reload()
