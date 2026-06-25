# OpenVINO Low-VRAM Mode & Context Window Optimization Guide

This guide explains how to optimize the model engine for long context windows (e.g., 32k tokens) and low-VRAM environments (e.g., Intel integrated GPUs or cards with $\le$ 8GB VRAM).

---

## 1. Why Do Out-of-Memory (OOM) Errors Occur?

When running Large Language Models on a GPU, memory consumption comes from three main sources:
1. **Model Weights**: A 3.8B model quantized to INT8 takes a static ~3.8 GB of VRAM.
2. **Activation Memory**: Temporary allocations used during feedforward operations (proportional to batch size and prompt length).
3. **Key-Value (KV) Cache**: Stored context history that allows the model to remember previous tokens. The KV cache grows **linearly** with the number of tokens in your conversation history.

In long context scenarios (like 16k or 32k tokens), the KV cache can easily exceed the size of the model weights, exhausting GPU memory and throwing allocation errors:
`Caught exception: [GPU] Exceeded max size of memory allocation`

---

## 2. Low-VRAM Optimization Features

Enabling `FORCE_LOW_VRAM_MODE=True` in your `.env` file injects specific properties into the OpenVINO device compiler config to minimize VRAM requirements:

*   **`"KV_CACHE_PRECISION": "u8"`**
    Compresses the stored Key and Value tensors from FP16 (2 bytes per element) to 8-bit unsigned integers (1 byte per element). This **cuts the KV cache memory footprint in half**.
*   **`"DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"`**
    Groups weight values dynamically to optimize activation quantization, reducing peak runtime workspace memory.
*   **`"PERFORMANCE_HINT": "LATENCY"`**
    Tells the compiler to optimize compile/execution graphs to minimize token-generation latency.
*   **`"GPU_DISABLE_REORDER_CACHING": "YES"`**
    Prevents the Intel graphics driver from allocating excessive cache memory during compilation, protecting systems from driver-level crashes on loading.

### Dynamic Parameter Recovery
Different OpenVINO versions and devices support different configurations. To prevent the engine from crashing on startup if a specific key is unsupported (e.g., `COMPUTE_HINT`), the engine uses **automatic config-filtering**:
If a property fails compilation, the loader detects the unsupported key from the error message, logs a warning, pops the key, and successfully retries compilation on the same device.

---

## 3. Sizing Your KV Cache: The Math

The memory required for the KV Cache is computed using this formula:

$$\text{KV Cache Size (Bytes)} = 2 \times \text{Layers} \times \text{KV Heads} \times \text{Head Dimension} \times \text{Tokens} \times \text{Precision (Bytes)}$$

*The initial factor of $2$ accounts for storing both Keys and Values.*

### Example Case: Phi-4-Mini-Instruct (3.8B Model)
*   **Layers ($L$)**: 32
*   **KV Heads ($H_{kv}$)**: 8 (uses Grouped Query Attention)
*   **Head Dimension ($D$)**: 96
*   **Context Length ($S$)**: 32,768 tokens (32k)

#### Scenario A: Low-VRAM Mode Disabled (FP16 Cache)
*   **Precision ($P$)**: 2 bytes
*   **Math**: $2 \times 32 \times 8 \times 96 \times 32,768 \times 2 = 3,221,225,472 \text{ bytes} \approx \mathbf{3.0\text{ GB}}$
*   **Total VRAM needed**: $3.8\text{ GB (Weights)} + 3.0\text{ GB (Cache)} + 1.2\text{ GB (Overheads)} = \mathbf{8.0\text{ GB}}$  
    *Verdict: ❌ Highly unstable or OOMs on an 8GB GPU.*

#### Scenario B: Low-VRAM Mode Enabled (`u8` Cache)
*   **Precision ($P$)**: 1 byte
*   **Math**: $2 \times 32 \times 8 \times 96 \times 32,768 \times 1 = 1,610,612,736 \text{ bytes} \approx \mathbf{1.5\text{ GB}}$
*   **Total VRAM needed**: $3.8\text{ GB (Weights)} + 1.5\text{ GB (Cache)} + 0.9\text{ GB (Overheads)} = \mathbf{6.2\text{ GB}}$  
    *Verdict:  Stable and fits comfortably on an 8GB GPU with 1.8GB to spare.*

---

## 4. Performance & Token-Speed Trade-offs

Enabling Low-VRAM Mode alters generation speeds depending on context size:

1.  **Short Contexts (<1,000 tokens)**:
    *   **Effect**: A minor **5% to 10% decrease** in tokens/sec.
    *   **Why**: The GPU must decompress the 8-bit integers back to FP16 to calculate attention, adding minor compute overhead.
2.  **Long Contexts (10k to 32k tokens)**:
    *   **Effect**: **No noticeable speed loss** (and can sometimes be slightly faster).
    *   **Why**: At high token counts, memory bandwidth (reading the huge KV cache from memory) becomes the primary bottleneck rather than raw calculation. Reading a compressed 8-bit cache cuts bandwidth requests in half, mitigating the decompression overhead.

---

## 5. Configuration & Enforcing Guardrails

To enable these optimizations, modify your `.env` file with the following settings:

```bash
# Force Low-VRAM optimizations (8-bit KV caching, driver memory limits)
FORCE_LOW_VRAM_MODE=True

# Enlarge hardware memory allocation limits (highly recommended for >1024 contexts)
OV_GPU_ENABLE_LARGE_ALLOCATIONS=True

# Set the maximum input tokens you want to allow in your history window
MAX_INPUT_TOKENS=32768

# Set the hard truncation limit (protects your GPU from exceeding memory budget)
MAX_TOKEN_LIMIT=32768
```

### The Strict Guardrail
If a single user prompt or history exceeds `MAX_TOKEN_LIMIT`, the engine applies a **hard truncation guardrail**. Instead of attempting to run the long prompt and crashing the GPU, the engine keeps only the newest `MAX_TOKEN_LIMIT` tokens and outputs a warning:
`[ModelEngine] WARNING: Input token length (35100) exceeds MAX_TOKEN_LIMIT (32768). Forcing hard truncation to keep only the newest tokens.`
This guarantees the service remains online and stable under heavy context loads.

---

## 6. Managing Repetitive Token Generation & Precision Loss

### Why Repeating Loops or Wrong Tokens Occur
Using `u8` (8-bit) KV cache compression reduces the memory footprint by 50%, but introduces minor **quantization noise** (rounding errors). This can sometimes lead to text generation degradation:
1. **Repetitive Loops**: The model gets stuck repeating the same phrases or sentences over and over.
2. **Loss of Focus**: Rounding errors compound over long chat histories, making the attention calculation less precise.
3. **Attention Sink Corruption**: Important prompt tokens (such as the initial system prompt token) receive distorted attention values, skewing the overall generation calibration.

### Mitigation Strategies

#### A. Adjust Generation Parameters
If the model gets stuck repeating text, adjust the sampling parameters in your inference requests:
*   **Repetition Penalty**: Set `repetition_penalty` to `1.1` or `1.15`. This dynamically decreases the probability of generating tokens that have already appeared.
*   **Temperature**: Slightly increase the temperature (e.g., from `0.7` to `0.85`) to introduce randomness, helping the model break out of repetitive loops.
*   **Top-P (Nucleus Sampling)**: Set `top_p` to `0.9` or `0.95` to filter out low-probability tail tokens.

#### B. Context-Dependent Execution
*   **For Short Conversations (< 2,000 tokens)**: Run with `FORCE_LOW_VRAM_MODE=False` to use full `FP16` precision and guarantee maximum accuracy.
*   **For Long Conversations (> 8,000 tokens)**: Run with `FORCE_LOW_VRAM_MODE=True` to conserve memory and stay within your GPU's physical hardware limits.

