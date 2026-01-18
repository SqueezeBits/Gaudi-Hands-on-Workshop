# Gaudi Hands-on Workshop

This repository contains four hands-on examples that demonstrate common workflows
on Intel® Gaudi®: diffusion image generation, LLM fine-tuning, profiling, and
vLLM quantization.

## Examples at a Glance

| Example | Folder | What you will do |
| --- | --- | --- |
| Diffusion | `Diffusion/` | Run Qwen-Image diffusion on Intel® Gaudi® and compare latency. |
| LLM Fine-tuning | `LLM-Fine-tuning/` | Fine-tune a small LLM with LoRA and GraLoRA. |
| Profiler | `Profiler/` | Profile attention and HPU traces to analyze performance. |
| vLLM Quantization | `vLLM-Quantization/` | Calibrate and quantize a vLLM model for inference. |

## Prerequisites

- Intel® Gaudi® environment with the required drivers and runtimes installed.
- Python environment with JupyterLab.
- Internet access for model and dataset downloads (if not already cached).

You can launch JupyterLab with the helper script:

```bash
bash run_jupyter_lab.sh
```

## How to Run

1. Start JupyterLab using the script above.
2. Open the notebooks in each folder in the suggested order.
3. Run cells top-to-bottom. Some notebooks generate artifacts or require configs
   from their local `configs/` folders.

## Example 1: Diffusion (Qwen-Image)

**Goal:** Generate images with Qwen-Image on Gaudi and inspect latency behavior.

Key files:

- `Diffusion/Gaudi_QwenImage_Workshop.ipynb` - Main hands-on notebook.
- `Diffusion/gaudi_qwen_patch.py` - Patches/utility helpers for Gaudi runs.
- `Diffusion/gaudi_transformer_qwenimage.py` - Gaudi transformer integration.
- `Diffusion/run_qwen_latency.py` - Script to measure latency.

Suggested flow:

1. Run the notebook to set up the environment and execute generation.
2. Use the latency script to compare runs and adjust settings.

## Example 2: LLM Fine-tuning (LoRA and GraLoRA)

**Goal:** Fine-tune a model with parameter-efficient techniques.

Key files:

- `LLM-Fine-tuning/1_LoRA_finetuning.ipynb` - LoRA walk-through.
- `LLM-Fine-tuning/2_GraLoRA_finetuning.ipynb` - GraLoRA walk-through.
- `LLM-Fine-tuning/run_lora_fine_tuning.py` - Scripted LoRA training.
- `LLM-Fine-tuning/joseon_persona_dataset.csv` - Sample dataset.

Suggested flow:

1. Start with LoRA notebook to understand the baseline setup.
2. Move to GraLoRA and compare results.
3. Use the training script for repeatable runs.

## Example 3: Profiler (Attention + HPU Trace)

**Goal:** Learn profiling tools and interpret performance signals.

Key files:

- `Profiler/profiling_hpu_trace.ipynb` - Profile basic operations such as matrix multiplication.
- `Profiler/profiling_attn_implementation.ipynb` - Profile attention.


Suggested flow:

1. Run HPU trace to inspect kernel timelines and bottlenecks.
2. Run attention profiling to compare implementations.

## Example 4: vLLM Quantization

**Goal:** Calibrate and quantize a model for efficient inference on Gaudi.

Key files:

- `vLLM-Quantization/1_vLLM_Inference.ipynb` - Baseline inference.
- `vLLM-Quantization/2_Calibration.ipynb` - Calibration steps.
- `vLLM-Quantization/3_Quantization.ipynb` - Quantization workflow.

Suggested flow:

1. Run baseline inference to establish metrics.
2. Calibrate with representative data.
3. Quantize and re-run inference to compare quality and speed.

## Repository Structure

```
Gaudi-Hands-on-Workshop/
├─ Diffusion/
├─ LLM-Fine-tuning/
├─ Profiler/
├─ vLLM-Quantization/
└─ run_jupyter_lab.sh
```

## Notes

- Notebooks may download models on first run; this can take time.
- If you use custom datasets or models, update paths in the notebooks or scripts.
