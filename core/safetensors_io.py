"""Streaming Safetensors Header & Binary I/O Engine for Merging Studio v2.0.

Guarantees 8-byte header alignment, zero-RAM-spike streaming writes, and exact
tensor ordering for:
- Standard BF16 / F16 / F32 / Mixed Precision models
- MOSS Image 757-Tensor INT8 ConvRot layout (1. comfy_quant -> 2. weight -> 3. weight_scale + comfy_attention.config + rope.bands)
- Krea 2 12.56GB Tensor.Art INT8 ConvRot layout (1. weight -> 2. weight_scale -> 3. comfy_quant + __metadata__)
"""

from __future__ import annotations
import json
import os
import struct
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from architectures.moss_image import MossImageSpec
from architectures.krea2 import Krea2Spec

try:
    import torch
    from safetensors import safe_open
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DTYPE_SIZE_MAP: Dict[str, int] = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1,
}


def read_safetensors_header_fast(file_path: str) -> Dict[str, Any]:
    """Reads only the JSON header of a .safetensors file in < 0.05s without loading tensors."""
    with open(file_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
        return json.loads(header_json)


def encode_aligned_safetensors_header(header_index: Dict[str, Any]) -> bytes:
    """Encodes a Safetensors header dictionary into 8-byte space-padded UTF-8 bytes."""
    header_bytes = json.dumps(header_index, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    return struct.pack("<Q", len(header_bytes)) + header_bytes


def write_tensor_binary(f_out, tensor: "torch.Tensor", out_dtype: "torch.dtype") -> None:
    """Writes a PyTorch tensor directly to an open binary file stream on CPU."""
    t_out = tensor.to(dtype=out_dtype).contiguous().cpu()
    if out_dtype == getattr(torch, "float8_e4m3fn", None):
        f_out.write(t_out.view(torch.int8).numpy().tobytes())
    elif out_dtype in (torch.bfloat16, torch.float16):
        f_out.write(t_out.view(torch.int16).numpy().tobytes())
    else:
        f_out.write(t_out.numpy().tobytes())


def find_scale_key_for_int8(raw_key: str, available_keys: Set[str], strip_fn) -> Optional[str]:
    """Finds the matching `.weight_scale` or `_scale` key for an INT8 weight tensor
    whether the model uses `.weight_scale` (from `.weight` base prefix) or `raw_key + '_scale'`.
    """
    candidates = [raw_key + "_scale"]
    if raw_key.endswith(".weight"):
        base_pref = raw_key[:-7]
        candidates.insert(0, f"{base_pref}.weight_scale")
        clean_pref = strip_fn(base_pref)
        candidates.append(f"{clean_pref}.weight_scale")
    clean_k = strip_fn(raw_key)
    candidates.append(clean_k + "_scale")

    for cand in candidates:
        if cand in available_keys:
            return cand
    return None


def build_int8_convrot_plan(
    key_shapes: Dict[str, Sequence[int]],
    active_arch: str,
    prefix_mode: str = "Auto",
    inject_moss_attention_config: bool = True,
    restore_moss_rope_bands: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Builds the exact task list and 8-byte aligned Safetensors header index for
    MOSS Image (757 tensors) or Krea 2 (Exact 12.56GB Tensor.Art layout).
    Preserves 100% of the logic from Cell 16 & Cell 17.
    """
    all_keys = sorted(list(key_shapes.keys()))
    raw_tasks: List[Dict[str, Any]] = []
    quantized_count = 0
    has_rope_bands = False
    moss_layer_indices: Set[int] = set()

    def format_out_key(key: str) -> str:
        clean_k = MossImageSpec.strip_prefix(key)
        if prefix_mode == "Auto":
            if active_arch == "MOSS_Image":
                return clean_k
            return key
        elif prefix_mode == "Strip_Prefix":
            return clean_k
        elif prefix_mode == "Add_Prefix":
            return f"model.diffusion_model.{clean_k}"
        return key

    comfy_quant_bytes = (
        MossImageSpec.COMFY_QUANT_PAYLOAD
        if active_arch == "MOSS_Image"
        else Krea2Spec.COMFY_QUANT_PAYLOAD
    )
    attn_config_bytes = MossImageSpec.ATTN_CONFIG_PAYLOAD

    for key in all_keys:
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale") or key.endswith(".comfy_attention.config"):
            continue

        clean_k = MossImageSpec.strip_prefix(key)
        if clean_k == "rope.bands":
            has_rope_bands = True

        if active_arch == "MOSS_Image" and clean_k.startswith("layers."):
            parts = clean_k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                moss_layer_indices.add(int(parts[1]))

        shape = list(key_shapes[key])
        num_elements = int(np.prod(shape)) if shape else 1
        out_key = format_out_key(key)

        is_int8 = (
            MossImageSpec.is_int8_convrot_target(clean_k, shape)
            if active_arch == "MOSS_Image"
            else Krea2Spec.is_int8_convrot_target(clean_k, shape)
        )

        if is_int8:
            out_features = shape[0]
            base_prefix = out_key[:-7] if out_key.endswith(".weight") else out_key
            raw_tasks.append({
                "type": "quantize",
                "sort_key": f"{base_prefix}.comfy_quant" if active_arch == "MOSS_Image" else out_key,
                "input_key": key,
                "base_prefix": base_prefix,
                "weight_key": out_key,
                "scale_key": f"{base_prefix}.weight_scale",
                "quant_key": f"{base_prefix}.comfy_quant",
                "shape": shape,
                "out_features": out_features,
                "num_elements": num_elements,
            })
            quantized_count += 1
        else:
            if active_arch == "MOSS_Image":
                dtype_name, bpe = MossImageSpec.get_layer_dtype_spec(clean_k)
            else:
                dtype_name, bpe = Krea2Spec.get_layer_dtype_spec(key, mode="int8_convrot_non_quant")

            raw_tasks.append({
                "type": "normal",
                "sort_key": out_key,
                "input_key": key,
                "output_key": out_key,
                "shape": shape,
                "dtype_name": dtype_name,
                "byte_size": num_elements * bpe,
            })

    if active_arch == "MOSS_Image" and inject_moss_attention_config:
        for layer_idx in sorted(moss_layer_indices):
            attn_key = format_out_key(f"layers.{layer_idx}.comfy_attention.config")
            raw_tasks.append({
                "type": "attn_config",
                "sort_key": attn_key,
                "output_key": attn_key,
                "shape": [len(attn_config_bytes)],
                "dtype_name": "U8",
                "byte_size": len(attn_config_bytes),
            })

    if active_arch == "MOSS_Image" and restore_moss_rope_bands and not has_rope_bands:
        rope_key = format_out_key("rope.bands")
        raw_tasks.append({
            "type": "synthetic_rope_bands",
            "sort_key": rope_key,
            "output_key": rope_key,
            "shape": [16],
            "dtype_name": "F32",
            "byte_size": 16 * 4,
        })

    tasks = sorted(raw_tasks, key=lambda x: x["sort_key"])
    header_index: Dict[str, Any] = {}
    current_offset = 0

    for task in tasks:
        if task["type"] == "quantize":
            byte_size_w = task["num_elements"] * 1
            byte_size_s = task["out_features"] * 4
            byte_size_q = len(comfy_quant_bytes)

            if active_arch == "MOSS_Image":
                # Official MOSS Image order: 1. comfy_quant -> 2. weight -> 3. weight_scale
                header_index[task["quant_key"]] = {
                    "dtype": "U8", "shape": [byte_size_q],
                    "data_offsets": [current_offset, current_offset + byte_size_q],
                }
                current_offset += byte_size_q

                header_index[task["weight_key"]] = {
                    "dtype": "I8", "shape": task["shape"],
                    "data_offsets": [current_offset, current_offset + byte_size_w],
                }
                current_offset += byte_size_w

                header_index[task["scale_key"]] = {
                    "dtype": "F32", "shape": [task["out_features"], 1],
                    "data_offsets": [current_offset, current_offset + byte_size_s],
                }
                current_offset += byte_size_s
            else:
                # Official Krea 2 Tensor.Art order: 1. weight -> 2. weight_scale -> 3. comfy_quant
                header_index[task["weight_key"]] = {
                    "dtype": "I8", "shape": task["shape"],
                    "data_offsets": [current_offset, current_offset + byte_size_w],
                }
                current_offset += byte_size_w

                header_index[task["scale_key"]] = {
                    "dtype": "F32", "shape": [task["out_features"], 1],
                    "data_offsets": [current_offset, current_offset + byte_size_s],
                }
                current_offset += byte_size_s

                header_index[task["quant_key"]] = {
                    "dtype": "U8", "shape": [byte_size_q],
                    "data_offsets": [current_offset, current_offset + byte_size_q],
                }
                current_offset += byte_size_q
        else:
            header_index[task["output_key"]] = {
                "dtype": task["dtype_name"],
                "shape": task["shape"],
                "data_offsets": [current_offset, current_offset + task["byte_size"]],
            }
            current_offset += task["byte_size"]

    if active_arch == "KREA_2":
        header_index["__metadata__"] = {
            "format": "pt",
            "model_architecture": "Krea2_MMDiT",
            "quantization": "int8_tensorwise",
            "convrot": "true",
            "quantized_layers": str(quantized_count),
        }

    return tasks, header_index
