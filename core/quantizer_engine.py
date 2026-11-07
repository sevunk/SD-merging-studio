"""Universal Quantizer (INT8 ConvRot), Unquantizer v5.0 (BF16), & Krea 2 GGUF GPU Engine.

Preserves 100% of the exact logic from:
- Cell 14: Krea 2 to GGUF T4 GPU Converter v3.0
- Cell 15: Universal Unquantizer v5.0 (MOSS & Krea 2 INT8 ConvRot / FP8 -> BF16 + rope.bands restoration)
- Cell 16 & 17: Universal INT8 ConvRot Native Quantizer (MOSS 757-tensor & Krea 2 12.56GB Tensor.Art layout)
"""

from __future__ import annotations
import gc
import os
import struct
from typing import Any, Dict
import numpy as np

from architectures import ARCH_REGISTRY, auto_detect_architecture
from architectures.moss_image import MossImageSpec
from architectures.krea2 import Krea2Spec
from .convrot_math import (
    CONVROT_GROUP_SIZE,
    dequantize_int8_convrot_to_fp32,
    quantize_fp32_to_int8_convrot_krea2,
    quantize_fp32_to_int8_convrot_moss,
)
from .safetensors_io import (
    build_int8_convrot_plan,
    encode_aligned_safetensors_header,
    find_scale_key_for_int8,
    read_safetensors_header_fast,
    write_tensor_binary,
)

try:
    import torch
    from safetensors import safe_open
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def run_universal_int8_convrot_quantizer(
    input_model_path: str,
    output_model_path: str,
    model_arch: str = "Auto-Detect",
    prefix_mode: str = "Auto",
    inject_moss_attention_config: bool = True,
    restore_moss_rope_bands: bool = True,
) -> Dict[str, Any]:
    """Executes the exact Universal INT8 ConvRot Quantizer from Cell 16 & Cell 17."""
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model not found: {input_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_header = read_safetensors_header_fast(input_model_path)
    all_keys = sorted([k for k in in_header.keys() if k != "__metadata__"])

    if model_arch == "Auto-Detect":
        spec = auto_detect_architecture(all_keys, os.path.basename(input_model_path))
        active_arch = spec.arch_id
    else:
        active_arch = model_arch

    key_shapes = {k: in_header[k]["shape"] for k in all_keys}
    tasks, header_index = build_int8_convrot_plan(
        key_shapes=key_shapes,
        active_arch=active_arch,
        prefix_mode=prefix_mode,
        inject_moss_attention_config=inject_moss_attention_config,
        restore_moss_rope_bands=restore_moss_rope_bands,
    )

    header_bytes = encode_aligned_safetensors_header(header_index)
    comfy_quant_bytes = (
        MossImageSpec.COMFY_QUANT_PAYLOAD
        if active_arch == "MOSS_Image"
        else Krea2Spec.COMFY_QUANT_PAYLOAD
    )

    with open(output_model_path, "wb") as f_out:
        f_out.write(header_bytes)
        with safe_open(input_model_path, framework="pt", device="cpu") as f_in:
            for idx, task in enumerate(tasks):
                if task["type"] == "quantize":
                    w = f_in.get_tensor(task["input_key"])
                    if active_arch == "MOSS_Image":
                        q, scale = quantize_fp32_to_int8_convrot_moss(w, device)
                        f_out.write(comfy_quant_bytes)
                        f_out.write(q.cpu().numpy().tobytes())
                        f_out.write(scale.cpu().numpy().tobytes())
                        del q, scale
                    else:
                        q, scale = quantize_fp32_to_int8_convrot_krea2(w, device)
                        f_out.write(q.cpu().numpy().tobytes())
                        f_out.write(scale.cpu().numpy().tobytes())
                        f_out.write(comfy_quant_bytes)
                        del q, scale
                    del w
                elif task["type"] == "attn_config":
                    f_out.write(MossImageSpec.ATTN_CONFIG_PAYLOAD)
                elif task["type"] == "synthetic_rope_bands":
                    exponent = torch.arange(16, device="cpu", dtype=torch.float32) / 16.0
                    rope_bands = (1.0 / (10000.0 ** exponent)).contiguous()
                    f_out.write(rope_bands.numpy().tobytes())
                else:
                    tensor = f_in.get_tensor(task["input_key"])
                    out_dt = torch.float32 if task["dtype_name"] == "F32" else torch.bfloat16
                    write_tensor_binary(f_out, tensor, out_dt)
                    del tensor

                if idx % 8 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()

        f_out.flush()

    return {
        "output_path": output_model_path,
        "architecture": active_arch,
        "total_tensors": len([k for k in header_index.keys() if k != "__metadata__"]),
        "size_gb": round(os.path.getsize(output_model_path) / (1024 ** 3), 4),
    }


def run_universal_unquantizer_v5(
    input_model_path: str,
    output_model_path: str,
    model_arch: str = "Auto-Detect",
    prefix_mode: str = "Auto",
    restore_moss_rope_bands: bool = True,
) -> Dict[str, Any]:
    """Executes the Universal Unquantizer v5.0 (Cell 15) using pure PyTorch Hadamard Un-rotation
    (zero external comfy-kitchen cloning required!).
    """
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"Input model not found: {input_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_header = read_safetensors_header_fast(input_model_path)
    all_keys = sorted([k for k in in_header.keys() if k != "__metadata__"])
    all_keys_set = set(all_keys)

    if model_arch == "Auto-Detect":
        spec = auto_detect_architecture(all_keys, os.path.basename(input_model_path))
        active_arch = spec.arch_id
    else:
        active_arch = model_arch

    def format_out_key(key: str) -> str:
        clean_k = MossImageSpec.strip_prefix(key)
        if prefix_mode == "Auto":
            return clean_k if active_arch == "MOSS_Image" else key
        elif prefix_mode == "Strip_Prefix":
            return clean_k
        elif prefix_mode == "Add_Prefix":
            return f"model.diffusion_model.{clean_k}"
        return key

    raw_tasks = []
    has_rope_bands = any(MossImageSpec.strip_prefix(k) == "rope.bands" for k in all_keys)

    for key in all_keys:
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale") or "comfy_attention.config" in key:
            continue

        clean_k = MossImageSpec.strip_prefix(key)
        out_key = format_out_key(key)
        shape = list(in_header[key]["shape"])
        num_el = int(np.prod(shape)) if shape else 1

        if active_arch == "MOSS_Image":
            dtype_str, bpe = MossImageSpec.get_layer_dtype_spec(clean_k)
        elif active_arch == "Pure_BF16":
            dtype_str, bpe = "BF16", 2
        else:
            dtype_str, bpe = Krea2Spec.get_layer_dtype_spec(clean_k, mode="mix_bf16_f32")

        raw_tasks.append({
            "type": "tensor",
            "input_key": key,
            "output_key": out_key,
            "shape": shape,
            "dtype_str": dtype_str,
            "byte_size": num_el * bpe,
        })

    if active_arch == "MOSS_Image" and restore_moss_rope_bands and not has_rope_bands:
        rope_out_key = format_out_key("rope.bands")
        raw_tasks.append({
            "type": "synthetic_rope_bands",
            "output_key": rope_out_key,
            "shape": [16],
            "dtype_str": "F32",
            "byte_size": 64,
        })

    tasks = sorted(raw_tasks, key=lambda x: x["output_key"])
    header_index = {}
    current_offset = 0
    for t in tasks:
        header_index[t["output_key"]] = {
            "dtype": t["dtype_str"],
            "shape": t["shape"],
            "data_offsets": [current_offset, current_offset + t["byte_size"]],
        }
        current_offset += t["byte_size"]

    header_bytes = encode_aligned_safetensors_header(header_index)

    with open(output_model_path, "wb") as f_out:
        f_out.write(header_bytes)
        with safe_open(input_model_path, framework="pt", device="cpu") as f_in:
            for idx, task in enumerate(tasks):
                if task["type"] == "synthetic_rope_bands":
                    exponent = torch.arange(16, device="cpu", dtype=torch.float32) / 16.0
                    rope_bands = (1.0 / (10000.0 ** exponent)).contiguous()
                    f_out.write(rope_bands.numpy().tobytes())
                else:
                    in_k = task["input_key"]
                    t = f_in.get_tensor(in_k).to(device)
                    if t.dtype == torch.int8:
                        scale_k = find_scale_key_for_int8(in_k, all_keys_set, MossImageSpec.strip_prefix)
                        if scale_k is not None:
                            s_t = f_in.get_tensor(scale_k)
                            t_out_fp32 = dequantize_int8_convrot_to_fp32(t, s_t, device, CONVROT_GROUP_SIZE)
                        else:
                            t_out_fp32 = t.float()
                    elif t.dtype == getattr(torch, "float8_e4m3fn", None):
                        scale_k = find_scale_key_for_int8(in_k, all_keys_set, MossImageSpec.strip_prefix)
                        t_out_fp32 = t.float()
                        if scale_k is not None:
                            s_t = f_in.get_tensor(scale_k).to(device, dtype=torch.float32)
                            if s_t.ndim == 1 and t_out_fp32.ndim == 2:
                                s_t = s_t.unsqueeze(1)
                            t_out_fp32 = t_out_fp32 * s_t
                    else:
                        t_out_fp32 = t.float()

                    out_dt = torch.float32 if task["dtype_str"] == "F32" else torch.bfloat16
                    write_tensor_binary(f_out, t_out_fp32, out_dt)
                    del t, t_out_fp32

                if idx % 10 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()

        f_out.flush()

    return {
        "output_path": output_model_path,
        "architecture": active_arch,
        "total_tensors": len(tasks),
        "size_gb": round(os.path.getsize(output_model_path) / (1024 ** 3), 4),
    }
