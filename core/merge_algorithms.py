"""Unified Checkpoint Merge Engine for MOSS Image, Krea 2, Anima, & MiniMax H3.

Supports:
1. MOSS Image v2.0 (`True_Disjoint_TIES`, `Norm_Preserving_TIES`, `Pure_Weighted_Linear`)
2. MOSS Image v3.0 (`Block_Guided_V3` with Unit-Normalized Voting & 42-Layer Early/Mid/Late Strength)
3. Krea 2 (`DARE_Hydra_Krea2` & `Krea2_Zone_Guided_DARE` with 28-Block + TextFusion safety & FULL Direct INT8 ConvRot Input/Output!)
4. Anima (`DARE_Hydra_Anima` with automatic `net.` -> `model.diffusion_model.` key translation)
5. MiniMax H3 (`DARE_Hydra_Minimax` with dynamic key translation & native INT8/U8/BF16 dtype preservation)
"""

from __future__ import annotations
from contextlib import ExitStack
import gc
import json
import os
import struct
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from architectures import ARCH_REGISTRY, auto_detect_architecture
from architectures.moss_image import MossImageSpec
from architectures.krea2 import Krea2Spec
from architectures.anima import AnimaSpec
from architectures.minimax_h3 import MinimaxH3Spec
from .convrot_math import (
    CONVROT_GROUP_SIZE,
    dequantize_int8_convrot_to_fp32,
    quantize_fp32_to_int8_convrot_krea2,
    quantize_fp32_to_int8_convrot_moss,
)
from .safetensors_io import (
    DTYPE_SIZE_MAP,
    build_int8_convrot_plan,
    encode_aligned_safetensors_header,
    find_scale_key_for_int8,
    read_safetensors_header_fast,
    write_tensor_binary,
)

try:
    import torch
    from safetensors import safe_open
    from tqdm import tqdm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def parse_weights_list(weights_input: str, num_models: int) -> List[float]:
    """Parses comma-separated donor weights and normalizes them to sum to 1.0."""
    if not weights_input or not weights_input.strip():
        return [1.0 / max(1, num_models)] * num_models
    try:
        raw = [float(x.strip()) for x in weights_input.split(",") if x.strip()]
    except ValueError:
        raw = []
    if len(raw) != num_models or sum(raw) <= 0:
        return [1.0 / max(1, num_models)] * num_models
    w_sum = sum(raw)
    return [w / w_sum for w in raw]


def compute_merged_delta_torch(
    deltas: List["torch.Tensor"],
    valid_weights: List[float],
    is_protected: bool,
    merge_mode: str,
    block_alpha: float = 1.0,
    ties_density: float = 0.40,
    p_drop: float = 0.0,
    generator: Optional["torch.Generator"] = None,
    compute_device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """Executes the exact mathematical delta combination for the selected merge_mode."""
    if not deltas:
        raise ValueError("deltas list cannot be empty")

    total_delta = torch.zeros_like(deltas[0], device=compute_device)
    w_sum = sum(valid_weights)
    norm_weights = [w / w_sum for w in valid_weights] if w_sum > 0 else [1.0 / len(deltas)] * len(deltas)
    rescale_factor = 1.0 / (1.0 - p_drop) if 0.0 < p_drop < 0.99 else 1.0

    # Protected layers or single-donor layers always receive smooth linear combination
    if is_protected or merge_mode == "Pure_Weighted_Linear" or len(deltas) == 1:
        for d, w_m in zip(deltas, norm_weights):
            total_delta += d * (w_m * block_alpha)
        return total_delta

    # --- MODE A: MOSS v3.0 Block-Guided Unit-Normalized Disjoint-TIES ---
    if merge_mode == "Block_Guided_V3":
        d_norms = [float(torch.norm(d, p=2).item()) for d in deltas]
        target_layer_norm = sum(w_m * dn for w_m, dn in zip(norm_weights, d_norms))

        weighted_dir = torch.zeros_like(deltas[0])
        for d, w_m, dn in zip(deltas, norm_weights, d_norms):
            if dn > 1e-6:
                weighted_dir += (d / dn) * w_m

        dominant_sign = torch.sign(weighted_dir)
        del weighted_dir

        agree_weight_sum = torch.zeros_like(deltas[0])
        for d, w_m in zip(deltas, norm_weights):
            sign_mask = (torch.sign(d) == dominant_sign) & (dominant_sign != 0)
            total_delta += (d * sign_mask) * w_m
            agree_weight_sum += sign_mask.to(torch.float32) * w_m
            del sign_mask

        total_delta = torch.where(
            agree_weight_sum > 1e-6,
            total_delta / agree_weight_sum,
            total_delta,
        )
        del dominant_sign, agree_weight_sum

        curr_norm = float(torch.norm(total_delta, p=2).item())
        if curr_norm > 1e-6 and target_layer_norm > 1e-6:
            total_delta *= (target_layer_norm / curr_norm) * block_alpha
        else:
            total_delta *= block_alpha
        return total_delta

    # --- MODE B: MOSS v2.0 True_Disjoint_TIES & Norm_Preserving_TIES ---
    if merge_mode in ("True_Disjoint_TIES", "Norm_Preserving_TIES"):
        weighted_mass = torch.zeros_like(deltas[0])
        target_norm_weighted = 0.0

        for d, w_m in zip(deltas, norm_weights):
            weighted_mass += d * w_m
            if merge_mode == "Norm_Preserving_TIES":
                target_norm_weighted += w_m * float(torch.norm(d, p=2).item())

        dominant_sign = torch.sign(weighted_mass)
        del weighted_mass

        agree_weight_sum = torch.zeros_like(deltas[0])
        for d, w_m in zip(deltas, norm_weights):
            if p_drop > 0.0:
                keep_mask = torch.rand(d.shape, generator=generator, dtype=d.dtype, device=compute_device) >= p_drop
                d_proc = d * keep_mask * rescale_factor
                del keep_mask
            else:
                d_proc = d

            sign_mask = (torch.sign(d_proc) == dominant_sign) & (dominant_sign != 0)
            total_delta += (d_proc * sign_mask) * w_m
            agree_weight_sum += sign_mask.to(torch.float32) * w_m
            del sign_mask, d_proc

        total_delta = torch.where(
            agree_weight_sum > 1e-6,
            total_delta / agree_weight_sum,
            total_delta,
        )
        del dominant_sign, agree_weight_sum

        if merge_mode == "Norm_Preserving_TIES" and target_norm_weighted > 1e-6:
            current_norm = float(torch.norm(total_delta, p=2).item())
            if current_norm > 1e-6:
                norm_scale = min(max(target_norm_weighted / current_norm, 0.85), 1.35)
                total_delta *= norm_scale

        total_delta *= block_alpha
        return total_delta

    # --- MODE C: DARE Hydra Engine (Krea 2, Anima, MiniMax H3) ---
    trimmed_deltas = []
    for d in deltas:
        flat = d.abs().flatten()
        k_val = max(1, int(ties_density * flat.numel()))
        if k_val == 0:
            continue
        threshold = torch.topk(flat, k_val, largest=True, sorted=False).values.min()
        trimmed_deltas.append(d * (d.abs() >= threshold))

    if trimmed_deltas:
        if len(trimmed_deltas) > 1:
            dominant_sign = torch.sign(torch.stack(trimmed_deltas).sum(dim=0))
            final_deltas = [
                td * ((torch.sign(td) == dominant_sign) & (dominant_sign != 0))
                for td in trimmed_deltas
            ]
            del dominant_sign, trimmed_deltas
        else:
            final_deltas = trimmed_deltas
            del trimmed_deltas

        for i, (fd, w_m) in enumerate(zip(final_deltas, norm_weights)):
            if p_drop > 0.0:
                keep_mask = torch.rand(fd.shape, generator=generator, dtype=fd.dtype, device=compute_device) >= p_drop
                total_delta += (fd * keep_mask * rescale_factor) * (w_m * block_alpha)
                del keep_mask
            else:
                total_delta += fd * (w_m * block_alpha)
            del fd
            final_deltas[i] = None
        del final_deltas

    return total_delta


def run_unified_checkpoint_merge(
    base_model_path: str,
    donor_model_paths: List[str],
    output_path: str,
    arch_id: str = "Auto-Detect",
    merge_mode: str = "Auto",
    output_format: str = "bf16",  # "bf16", "fp16", "mix_bf16_f32", "int8_convrot", "native"
    donor_weights_str: str = "",
    ties_density: float = 0.40,
    p_drop: float = 0.0,
    seed: int = 42,
    early_strength: float = 0.55,
    mid_strength: float = 0.90,
    late_strength: float = 0.30,
    other_strength: float = 0.50,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Unified streaming checkpoint merger supporting BF16 & Direct INT8 ConvRot inputs/outputs
    across MOSS Image, Krea 2, Anima, and MiniMax H3 while strictly preserving every architecture's
    layer invariants.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch and safetensors are required to execute checkpoint merging.")

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    valid_donors = [p.strip() for p in donor_model_paths if p.strip() and os.path.exists(p.strip())]
    if not valid_donors:
        raise FileNotFoundError("No valid donor model files found.")

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=compute_device).manual_seed(seed)

    base_header = read_safetensors_header_fast(base_model_path)
    raw_base_keys_all = [k for k in base_header.keys() if k != "__metadata__"]

    if arch_id == "Auto-Detect":
        spec = auto_detect_architecture(raw_base_keys_all, os.path.basename(base_model_path))
    else:
        spec = ARCH_REGISTRY.get(arch_id, Krea2Spec)

    active_arch = spec.arch_id
    if merge_mode == "Auto":
        merge_mode = spec.default_merge_mode

    weights = parse_weights_list(donor_weights_str, len(valid_donors))

    # Filter out quant metadata keys so we iterate over logical model tensors
    logical_base_keys = [k for k in raw_base_keys_all if not spec.is_quant_metadata_key(k)]

    with ExitStack() as stack:
        f_base = stack.enter_context(safe_open(base_model_path, framework="pt", device="cpu"))
        base_all_set = set(f_base.keys())

        f_donors = []
        donor_key_maps = []
        donor_all_sets = []

        for p in valid_donors:
            fh = stack.enter_context(safe_open(p, framework="pt", device="cpu"))
            f_donors.append(fh)
            d_keys_set = set(fh.keys())
            donor_all_sets.append(d_keys_set)

            k_map: Dict[str, str] = {}
            if active_arch == "MOSS_Image":
                base_clean_map = {MossImageSpec.strip_prefix(k): k for k in logical_base_keys}
                for rk in d_keys_set:
                    if not MossImageSpec.is_quant_metadata_key(rk):
                        ck = MossImageSpec.strip_prefix(rk)
                        if ck in base_clean_map:
                            k_map[ck] = rk
            elif active_arch == "ANIMA":
                base_set = set(logical_base_keys)
                for rk in d_keys_set:
                    tk = AnimaSpec.translate_to_base_key(rk)
                    if tk in base_set:
                        k_map[tk] = rk
            else:
                # Krea 2 & MiniMax H3
                base_set = set(logical_base_keys)
                base_clean_map = {spec.strip_prefix(k): k for k in logical_base_keys}
                for rk in d_keys_set:
                    if spec.is_quant_metadata_key(rk):
                        continue
                    tk = MinimaxH3Spec.get_translated_key(rk, base_set)
                    if tk is not None:
                        k_map[tk] = rk
                    else:
                        ck = spec.strip_prefix(rk)
                        if ck in base_clean_map:
                            k_map[base_clean_map[ck]] = rk
            donor_key_maps.append(k_map)

        def load_decoded_fp32(fh, all_set: Set[str], raw_key: str) -> "torch.Tensor":
            t = fh.get_tensor(raw_key).to(compute_device)
            if t.dtype == torch.int8:
                scale_k = find_scale_key_for_int8(raw_key, all_set, spec.strip_prefix)
                if scale_k is not None:
                    scale_t = fh.get_tensor(scale_k)
                    t_float = dequantize_int8_convrot_to_fp32(t, scale_t, compute_device, CONVROT_GROUP_SIZE)
                    del t, scale_t
                    return t_float
            elif t.dtype == getattr(torch, "float8_e4m3fn", None):
                scale_k = find_scale_key_for_int8(raw_key, all_set, spec.strip_prefix)
                t_float = t.float()
                if scale_k is not None:
                    scale_t = fh.get_tensor(scale_k).to(compute_device, dtype=torch.float32)
                    if scale_t.ndim == 1 and t_float.ndim == 2:
                        scale_t = scale_t.unsqueeze(1)
                    t_float = t_float * scale_t
                del t
                return t_float
            t_float = t.float()
            del t
            return t_float

        # --- BRANCH 1: DIRECT INT8 CONVROT OUTPUT (FOR MOSS IMAGE OR KREA 2) ---
        if output_format == "int8_convrot" and active_arch in ("MOSS_Image", "KREA_2"):
            key_shapes = {k: base_header[k]["shape"] for k in logical_base_keys}
            tasks, header_index = build_int8_convrot_plan(
                key_shapes=key_shapes,
                active_arch=active_arch,
                prefix_mode="Auto",
                inject_moss_attention_config=(active_arch == "MOSS_Image"),
                restore_moss_rope_bands=(active_arch == "MOSS_Image"),
            )
            header_bytes = encode_aligned_safetensors_header(header_index)
            comfy_quant_bytes = (
                MossImageSpec.COMFY_QUANT_PAYLOAD
                if active_arch == "MOSS_Image"
                else Krea2Spec.COMFY_QUANT_PAYLOAD
            )

            with open(output_path, "wb") as f_out:
                f_out.write(header_bytes)
                total_steps = len(tasks)

                for idx, task in enumerate(tasks):
                    if task["type"] == "attn_config":
                        f_out.write(MossImageSpec.ATTN_CONFIG_PAYLOAD)
                    elif task["type"] == "synthetic_rope_bands":
                        exponent = torch.arange(16, device="cpu", dtype=torch.float32) / 16.0
                        rope_bands = (1.0 / (10000.0 ** exponent)).contiguous()
                        f_out.write(rope_bands.numpy().tobytes())
                    else:
                        in_key = task["input_key"]
                        clean_k = spec.strip_prefix(in_key)
                        lookup_k = clean_k if active_arch == "MOSS_Image" else in_key

                        t_base_fp32 = load_decoded_fp32(f_base, base_all_set, in_key)

                        if active_arch == "MOSS_Image" and clean_k == "rope.bands":
                            write_tensor_binary(f_out, t_base_fp32, torch.float32)
                            del t_base_fp32
                            continue

                        deltas = []
                        valid_w = []
                        for m_idx, fh in enumerate(f_donors):
                            k_map = donor_key_maps[m_idx]
                            if lookup_k in k_map:
                                t_m_fp32 = load_decoded_fp32(fh, donor_all_sets[m_idx], k_map[lookup_k])
                                if t_m_fp32.shape == t_base_fp32.shape:
                                    d = t_m_fp32 - t_base_fp32
                                    if merge_mode != "Block_Guided_V3" or float(torch.max(d.abs()).item()) > 1e-7:
                                        deltas.append(d)
                                        valid_w.append(weights[m_idx])
                                    else:
                                        del d
                                del t_m_fp32

                        if deltas:
                            is_prot = spec.is_protected_layer(clean_k, len(t_base_fp32.shape))
                            b_alpha = 1.0
                            if merge_mode == "Block_Guided_V3" and active_arch == "MOSS_Image":
                                b_alpha = MossImageSpec.get_block_strength(
                                    clean_k, early_strength, mid_strength, late_strength, other_strength
                                )
                            elif merge_mode == "Krea2_Zone_Guided_DARE" and active_arch == "KREA_2":
                                b_alpha = Krea2Spec.get_zone_strength(
                                    clean_k, early_strength, mid_strength, late_strength, other_strength
                                )

                            total_delta = compute_merged_delta_torch(
                                deltas=deltas,
                                valid_weights=valid_w,
                                is_protected=is_prot,
                                merge_mode=merge_mode,
                                block_alpha=b_alpha,
                                ties_density=ties_density,
                                p_drop=p_drop,
                                generator=generator,
                                compute_device=compute_device,
                            )
                            final_fp32 = t_base_fp32 + total_delta
                            del total_delta
                        else:
                            final_fp32 = t_base_fp32

                        del t_base_fp32, deltas

                        if task["type"] == "quantize":
                            if active_arch == "MOSS_Image":
                                q, scale = quantize_fp32_to_int8_convrot_moss(final_fp32, compute_device)
                                f_out.write(comfy_quant_bytes)
                                f_out.write(q.cpu().numpy().tobytes())
                                f_out.write(scale.cpu().numpy().tobytes())
                                del q, scale
                            else:
                                q, scale = quantize_fp32_to_int8_convrot_krea2(final_fp32, compute_device)
                                f_out.write(q.cpu().numpy().tobytes())
                                f_out.write(scale.cpu().numpy().tobytes())
                                f_out.write(comfy_quant_bytes)
                                del q, scale
                        else:
                            out_dt = torch.float32 if task["dtype_name"] == "F32" else torch.bfloat16
                            write_tensor_binary(f_out, final_fp32, out_dt)

                        del final_fp32

                    if progress_callback:
                        progress_callback(idx + 1, total_steps, task.get("sort_key", ""))
                    if idx % 8 == 0 and compute_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if idx % 25 == 0:
                        gc.collect()
                f_out.flush()

            return {
                "output_path": output_path,
                "architecture": active_arch,
                "merge_mode": merge_mode,
                "output_format": "int8_convrot",
                "total_tensors": len([k for k in header_index.keys() if k != "__metadata__"]),
                "size_bytes": os.path.getsize(output_path),
            }

        # --- BRANCH 2: BF16 / F16 / NATIVE DTYPE OUTPUT ---
        header_index = {}
        current_offset = 0

        if active_arch == "MOSS_Image":
            base_clean_to_raw = {MossImageSpec.strip_prefix(k): k for k in logical_base_keys}
            iter_keys = sorted(list(base_clean_to_raw.keys()))
        else:
            base_clean_to_raw = {k: k for k in logical_base_keys}
            iter_keys = sorted(logical_base_keys)

        for out_k in iter_keys:
            raw_k = base_clean_to_raw[out_k]
            shape = base_header[raw_k]["shape"]
            num_el = int(np.prod(shape)) if shape else 1

            if active_arch == "MOSS_Image":
                dtype_str, bpe = MossImageSpec.get_layer_dtype_spec(out_k)
            elif active_arch == "KREA_2":
                if output_format == "mix_bf16_f32":
                    dtype_str, bpe = Krea2Spec.get_layer_dtype_spec(out_k, mode="mix_bf16_f32")
                else:
                    orig_dt = base_header[raw_k]["dtype"]
                    dtype_str = "BF16" if orig_dt == "I8" else orig_dt
                    if dtype_str not in ("F32", "BF16"):
                        dtype_str = "BF16"
                    bpe = DTYPE_SIZE_MAP[dtype_str]
            elif active_arch == "ANIMA":
                dtype_str = "F16" if output_format == "fp16" else "BF16"
                bpe = 2
            else:
                # MiniMax H3 preserves native dtype from base_header
                dtype_str = base_header[raw_k]["dtype"]
                bpe = DTYPE_SIZE_MAP[dtype_str]

            byte_size = num_el * bpe
            header_index[out_k] = {
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [current_offset, current_offset + byte_size],
            }
            current_offset += byte_size

        if active_arch == "KREA_2":
            header_index["__metadata__"] = {
                "format": "pt",
                "model_architecture": "Krea2_MMDiT",
                "merged_by": f"DARE_HYDRA_Krea2_{output_format}",
            }
        elif active_arch == "ANIMA":
            header_index["__metadata__"] = AnimaSpec.get_checkpoint_metadata()
        elif active_arch == "MINIMAX_H3":
            base_meta = dict(base_header.get("__metadata__", {}))
            base_meta["merged_by"] = "DARE_HYDRA_Minimax_H3_GPU_Smart"
            header_index["__metadata__"] = base_meta

        header_bytes = encode_aligned_safetensors_header(header_index)
        dtype_torch_map = {
            "F64": torch.float64, "I64": torch.int64,
            "F32": torch.float32, "I32": torch.int32,
            "F16": torch.float16, "BF16": torch.bfloat16, "I16": torch.int16,
            "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
        }

        with open(output_path, "wb") as f_out:
            f_out.write(header_bytes)
            total_steps = len(iter_keys)

            for idx, out_k in enumerate(iter_keys):
                raw_k = base_clean_to_raw[out_k]
                clean_k = spec.strip_prefix(out_k)
                dtype_str = header_index[out_k]["dtype"]
                out_dtype = dtype_torch_map[dtype_str]

                if active_arch == "MINIMAX_H3":
                    # MiniMax H3 native tensor load
                    t_base_fp32 = f_base.get_tensor(raw_k).float().to(compute_device)
                else:
                    t_base_fp32 = load_decoded_fp32(f_base, base_all_set, raw_k)

                if active_arch == "MOSS_Image" and clean_k == "rope.bands":
                    write_tensor_binary(f_out, t_base_fp32, torch.float32)
                    del t_base_fp32
                    continue

                deltas = []
                valid_w = []
                for m_idx, fh in enumerate(f_donors):
                    k_map = donor_key_maps[m_idx]
                    if out_k in k_map:
                        actual_k = k_map[out_k]
                        if active_arch == "MINIMAX_H3":
                            t_m_fp32 = fh.get_tensor(actual_k).float().to(compute_device)
                        else:
                            t_m_fp32 = load_decoded_fp32(fh, donor_all_sets[m_idx], actual_k)
                        if t_m_fp32.shape == t_base_fp32.shape:
                            d = t_m_fp32 - t_base_fp32
                            if merge_mode != "Block_Guided_V3" or float(torch.max(d.abs()).item()) > 1e-7:
                                deltas.append(d)
                                valid_w.append(weights[m_idx])
                            else:
                                del d
                        del t_m_fp32

                if deltas:
                    is_prot = spec.is_protected_layer(clean_k, len(t_base_fp32.shape))
                    b_alpha = 1.0
                    if merge_mode == "Block_Guided_V3" and active_arch == "MOSS_Image":
                        b_alpha = MossImageSpec.get_block_strength(
                            clean_k, early_strength, mid_strength, late_strength, other_strength
                        )
                    elif merge_mode == "Krea2_Zone_Guided_DARE" and active_arch == "KREA_2":
                        b_alpha = Krea2Spec.get_zone_strength(
                            clean_k, early_strength, mid_strength, late_strength, other_strength
                        )

                    total_delta = compute_merged_delta_torch(
                        deltas=deltas,
                        valid_weights=valid_w,
                        is_protected=is_prot,
                        merge_mode=merge_mode,
                        block_alpha=b_alpha,
                        ties_density=ties_density,
                        p_drop=p_drop,
                        generator=generator,
                        compute_device=compute_device,
                    )
                    final_tensor = t_base_fp32 + total_delta
                    del total_delta
                else:
                    final_tensor = t_base_fp32

                # Safe integer clamping for MiniMax H3 native INT8/U8 layers
                if dtype_str == "I8":
                    final_tensor = torch.clamp(torch.round(final_tensor), -128, 127)
                elif dtype_str == "U8":
                    final_tensor = torch.clamp(torch.round(final_tensor), 0, 255)
                elif dtype_str in ("I16", "I32", "I64"):
                    final_tensor = torch.round(final_tensor)

                write_tensor_binary(f_out, final_tensor, out_dtype)
                del t_base_fp32, final_tensor, deltas

                if progress_callback:
                    progress_callback(idx + 1, total_steps, out_k)
                if idx % 8 == 0 and compute_device.type == "cuda":
                    torch.cuda.empty_cache()
                if idx % 25 == 0:
                    gc.collect()

            f_out.flush()

        return {
            "output_path": output_path,
            "architecture": active_arch,
            "merge_mode": merge_mode,
            "output_format": output_format,
            "total_tensors": len(iter_keys),
            "size_bytes": os.path.getsize(output_path),
        }
