import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# CRITICAL: Set PT_HPU_MAX_COMPOUND_OP_SIZE BEFORE importing habana/torch
# diffusers sets this to 1 in pipe.to('hpu'), which causes OOM
os.environ["PT_HPU_MAX_COMPOUND_OP_SIZE"] = "9223372036854775807"

import torch
import transformers
import habana_frameworks.torch.core as htcore
from diffusers import DiffusionPipeline
from habana_frameworks.torch.hpu.graphs import wrap_in_hpu_graph

from gaudi_qwen_patch import patch_qwenimage_for_hpu


# Patch diffusers to prevent it from setting PT_HPU_MAX_COMPOUND_OP_SIZE=1
# This happens in DiffusionPipeline.to() and causes OOM
def _patch_diffusers_hpu_env():
    """Prevent diffusers from overriding PT_HPU_MAX_COMPOUND_OP_SIZE to 1."""
    import diffusers.pipelines.pipeline_utils as pu
    
    _original_to = pu.DiffusionPipeline.to
    
    def _patched_to(self, *args, **kwargs):
        # Disable the automatic HPU env var setting
        kwargs.setdefault("hpu_migration", False)
        result = _original_to(self, *args, **kwargs)
        # Restore our preferred value after .to() call
        os.environ["PT_HPU_MAX_COMPOUND_OP_SIZE"] = "9223372036854775807"
        return result
    
    pu.DiffusionPipeline.to = _patched_to

_patch_diffusers_hpu_env()


# Patch for transformers compatibility (from_model_config issue)
def _patched_from_model_config(cls, model_config, **kwargs):
    """
    Patched version of GenerationConfig.from_model_config that handles dict decoder_config.
    This fixes compatibility between diffusers and newer transformers versions.
    """
    from transformers import GenerationConfig
    
    config_dict = model_config.to_diff_dict() if hasattr(model_config, "to_diff_dict") else {}
    
    # Handle decoder_config if present
    if hasattr(model_config, "decoder") and model_config.decoder is not None:
        decoder_config = model_config.decoder
        if isinstance(decoder_config, dict):
            decoder_config_dict = decoder_config
        elif hasattr(decoder_config, "to_dict"):
            decoder_config_dict = decoder_config.to_dict()
        else:
            decoder_config_dict = {}
        config_dict.update(decoder_config_dict)
    
    # Filter to only include valid GenerationConfig parameters
    generation_config_keys = set(GenerationConfig().to_dict().keys())
    filtered_config = {k: v for k, v in config_dict.items() if k in generation_config_keys}
    
    return GenerationConfig(**filtered_config, **kwargs)

# Apply the patch
transformers.generation.configuration_utils.GenerationConfig.from_model_config = classmethod(_patched_from_model_config)


def _percentile(values: Iterable[float], pct: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen-Image latency runner (Gaudi/HPU)")
    p.add_argument("--model-id", default="Qwen/Qwen-Image")
    p.add_argument("--device", default="hpu")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--mode", default="eager", choices=("eager", "compile", "hpu_graphs"))

    p.add_argument(
        "--prompt",
        default=(
            'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee $2 per cup," '
            'with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman. '
            "Ultra HD, 4K, cinematic composition"
        ),
    )
    p.add_argument("--positive-magic", default=", Ultra HD, 4K, cinematic composition.")
    p.add_argument("--negative-prompt", default=" ")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)

    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--true-cfg-scale", type=float, default=4.0)
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="example.png")

    p.add_argument("--patch-rope", action="store_true", help="Apply HPU RoPE patch")
    p.add_argument("--patch-fused-attn", action="store_true", help="Apply HPU fused attention patch")
    return p


def _dtype_from_arg(arg: str) -> torch.dtype:
    if arg == "bf16":
        return torch.bfloat16
    if arg == "fp16":
        return torch.float16
    if arg == "fp32":
        return torch.float32
    raise ValueError(f"Unexpected dtype: {arg}")


def _ensure_lazy_mode(mode: str):
    lazy = os.environ.get("PT_HPU_LAZY_MODE", "")
    if mode == "compile" and lazy == "1":
        raise RuntimeError("torch.compile requires PT_HPU_LAZY_MODE=0 (eager).")
    if mode == "hpu_graphs" and lazy != "1":
        raise RuntimeError("hpu_graphs requires PT_HPU_LAZY_MODE=1 (lazy).")


def _compile_pipeline_modules(pipe: Any, mode: str) -> None:
    """
    Apply torch.compile or wrap_in_hpu_graph to the pipeline modules, following the pattern used in
    `optimum-habana/examples/stable-diffusion/text_to_image_generation.py`.

    - Prefer `pipe.stages` if present (list of module attribute names)
    - For HPU graphs, if `_repeated_blocks` is defined, compile/wrap those repeated blocks individually
      to avoid locking the entire module to warmup-defined shapes.
    """

    if mode == "eager":
        return

    def compile_module(module: torch.nn.Module) -> torch.nn.Module:
        if mode == "compile":
            return torch.compile(module, backend="hpu_backend", options={"keep_input_mutations": True})
        if mode == "hpu_graphs":
            return wrap_in_hpu_graph(module)
        return module

    stages = getattr(pipe, "stages", None)
    if not stages:
        # Prefer the same structure used in optimum-habana examples: treat the main compute
        # module as a stage, so we can optionally wrap repeated blocks instead of the entire module.
        if hasattr(pipe, "transformer") and isinstance(pipe.transformer, torch.nn.Module):
            stages = ["transformer"]
            try:
                setattr(pipe, "stages", stages)
            except Exception:
                pass
        else:
            raise AttributeError("Pipeline has no `stages` list and no `transformer` module to compile/wrap.")

    for target_module in stages:
        target_module_obj = getattr(pipe, target_module)
        if not isinstance(target_module_obj, torch.nn.Module):
            continue

        repeated = getattr(pipe, "_repeated_blocks", None) or getattr(target_module_obj, "_repeated_blocks", None)

        if mode == "hpu_graphs" and repeated:
            # Compile repeated blocks dynamically
            for name, module in target_module_obj.named_modules():
                if module.__class__.__name__ in repeated:
                    if "." in name:
                        parent_name = ".".join(name.split(".")[:-1])
                        child_name = name.split(".")[-1]
                        parent_module = target_module_obj.get_submodule(parent_name)
                        setattr(parent_module, child_name, compile_module(module))
                    else:
                        setattr(target_module_obj, name, compile_module(module))
        else:
            # Fallback: compile/wrap the entire module (may lock graph cache to warmup-defined shapes)
            setattr(pipe, target_module, compile_module(target_module_obj))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    # Create outputs directory if it doesn't exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Prepend outputs/ to output path if not already there
    if args.output and not args.output.startswith("outputs/"):
        args.output = str(outputs_dir / args.output)

    torch_dtype = _dtype_from_arg(args.dtype)
    device = args.device
    _ensure_lazy_mode(args.mode)

    # Enable bf16 reduction for SDP (critical for HPU memory efficiency)
    if hasattr(torch._C, "_set_math_sdp_allow_fp16_bf16_reduction"):
        torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)
        print("[INFO] Enabled _set_math_sdp_allow_fp16_bf16_reduction(True)")

    # Helpful to print minimal run metadata for notebooks/log parsing
    run_meta: Dict[str, Any] = {
        "model_id": args.model_id,
        "device": device,
        "dtype": args.dtype,
        "mode": args.mode,
        "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE", ""),
        "patch_rope": bool(args.patch_rope),
        "patch_fused_attn": bool(args.patch_fused_attn),
        "steps": args.steps,
        "size": f"{args.width}x{args.height}",
        "num_runs": args.num_runs,
        "skip": args.skip,
        "true_cfg_scale": args.true_cfg_scale,
    }
    print(f"[run_qwen_latency] {run_meta}")

    # PT_HPU_MAX_COMPOUND_OP_SIZE is set at module level (before imports) to prevent OOM
    print(f"[INFO] PT_HPU_MAX_COMPOUND_OP_SIZE = {os.environ.get('PT_HPU_MAX_COMPOUND_OP_SIZE', 'not set')}")

    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)

    if args.patch_rope or args.patch_fused_attn:
        pipe = patch_qwenimage_for_hpu(pipe, patch_rope=args.patch_rope, patch_fused_attn=args.patch_fused_attn)

    pipe = pipe.to(device)

    # Set stages and _repeated_blocks for efficient HPU graphs wrapping
    # (same as GaudiQwenImagePipeline in optimum-habana)
    if not hasattr(pipe, "stages") or not pipe.stages:
        pipe.stages = ["transformer"]
    if not hasattr(pipe, "_repeated_blocks") or not pipe._repeated_blocks:
        pipe._repeated_blocks = ["QwenImageTransformerBlock"]

    _compile_pipeline_modules(pipe, args.mode)

    latencies = []
    last_image = None

    for i in range(args.num_runs):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        torch.hpu.synchronize()
        start = time.time()
        last_image = pipe(
            prompt=args.prompt + args.positive_magic,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator,
        ).images[0]
        htcore.mark_step()
        torch.hpu.synchronize()
        lat = time.time() - start
        latencies.append(lat)
        print(f"[iter] {i+1}/{args.num_runs} latency_sec={lat:.4f}")

    effective = latencies[max(0, args.skip) :]
    if effective:
        mean = statistics.fmean(effective)
        p50 = _percentile(effective, 50)
        p90 = _percentile(effective, 90)
        print(
            "[summary] "
            + " ".join(
                [
                    f"mean_sec={mean:.4f}",
                    f"p50_sec={p50:.4f}",
                    f"p90_sec={p90:.4f}",
                    f"runs={len(effective)}",
                ]
            )
        )
    else:
        print("[summary] not_enough_runs_after_skip=1")

    if last_image is not None and args.output:
        last_image.save(args.output)
        print(f"[output] saved_image={args.output}")

    # Save latency results to JSON file alongside the image
    if args.output and effective:
        output_path = Path(args.output)
        json_path = output_path.with_suffix(".json")
        latency_data = {
            "stage": output_path.stem,
            "image_path": str(output_path),
            "run_meta": run_meta,
            "all_latencies": latencies,
            "effective_latencies": effective,
            "statistics": {
                "mean_sec": mean,
                "p50_sec": p50,
                "p90_sec": p90,
                "num_runs": len(effective),
            },
        }
        with open(json_path, "w") as f:
            json.dump(latency_data, f, indent=2)
        print(f"[output] saved_latency_json={json_path}")


if __name__ == "__main__":
    main()

