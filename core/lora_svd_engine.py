"""Unified LoRA & LoKr SVD Merger, Baker, & Extractor Engine for Merging Studio v2.0.

Preserves 100% of the exact logic and layer mappings from:
- Cell 5: MOSS Image Multi-LoRA & LoKr SVD Merger (Fused QKV [9216, 3072] <-> Split Q/K/V [3072, 3072])
- Cell 7: Krea 2 Universal LoRA Baker V5.2 (Full ModelScope & Diffusers Alias Support + Mixed Precision)
- Cell 8: Krea 2 Fast LoRA Extractor (Low-Rank SVD Turbo)
- Cell 9: Krea 2 Universal Multi-LoRA Merger V3.1 (Full `mlp.down` recovery & Tensor.Art metadata)
- Cell 13: MiniMax H3 Multi-LoRA SVD Merger
"""

from __future__ import annotations
import gc
import json
import os
import re
import struct
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from architectures.moss_image import MossImageSpec
from architectures.krea2 import Krea2Spec
from architectures.minimax_h3 import MinimaxH3Spec
from .safetensors_io import encode_aligned_safetensors_header, write_tensor_binary

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tqdm import tqdm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def parse_lora_role_strict(k_lower: str) -> Optional[str]:
    """Bulletproof LoRA / LoKr role detector from Cell 5 & Cell 9.
    Strictly checks adapter suffixes so `mlp.down` and `ffn_down` are NEVER misclassified as 'down'!
    """
    if k_lower.endswith(".alpha") or ".lora_alpha" in k_lower or "alpha" in k_lower.split(".")[-1]:
        return "alpha"
    if k_lower.endswith(".diff") or k_lower.endswith(".diff.weight"):
        return "diff"

    if any(
        k_lower.endswith(x)
        for x in [
            ".lora_a.weight",
            ".lora_a",
            ".lora_down.weight",
            ".lora_down",
            ".lora.down.weight",
            ".lora_a.default.weight",
        ]
    ):
        return "down"
    if re.search(r"[\._]lora[_\.]a[\._]", k_lower) or re.search(r"[\._]lora[_\.]down[\._]", k_lower):
        return "down"

    if any(
        k_lower.endswith(x)
        for x in [
            ".lora_b.weight",
            ".lora_b",
            ".lora_up.weight",
            ".lora_up",
            ".lora.up.weight",
            ".lora_b.default.weight",
        ]
    ):
        return "up"
    if re.search(r"[\._]lora[_\.]b[\._]", k_lower) or re.search(r"[\._]lora[_\.]up[\._]", k_lower):
        return "up"

    if "lokr_w1_a" in k_lower:
        return "w1_a"
    if "lokr_w1_b" in k_lower:
        return "w1_b"
    if "lokr_w2_a" in k_lower:
        return "w2_a"
    if "lokr_w2_b" in k_lower:
        return "w2_b"
    if "lokr_w1" in k_lower:
        return "w1"
    if "lokr_w2" in k_lower:
        return "w2"

    return None


def load_adapter_bundles(lora_path: str, arch_id: str, for_baker: bool = False) -> Dict[str, Dict[str, Any]]:
    """Loads a LoRA / LoKr / Diff `.safetensors` adapter and normalizes keys according to the target architecture."""
    if not os.path.exists(lora_path):
        return {}

    if arch_id == "MOSS_Image":
        norm_fn = MossImageSpec.normalize_lora_clean_key
    elif arch_id == "MINIMAX_H3":
        norm_fn = MinimaxH3Spec.normalize_lora_clean_key
    else:
        norm_fn = Krea2Spec.normalize_baker_key if for_baker else Krea2Spec.normalize_lora_merger_key

    raw_bundles: Dict[str, Dict[str, Any]] = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f_lora:
        for k in f_lora.keys():
            role = parse_lora_role_strict(k.lower())
            if not role:
                continue
            clean = norm_fn(k)
            if clean not in raw_bundles:
                raw_bundles[clean] = {}

            if role == "alpha":
                raw_bundles[clean]["alpha"] = float(f_lora.get_tensor(k).float().item())
            else:
                raw_bundles[clean][role] = f_lora.get_tensor(k).float()

    if arch_id == "MOSS_Image":
        for clean in list(raw_bundles.keys()):
            if clean.endswith(".proj.qkv") and "alpha" not in raw_bundles[clean]:
                base_block = clean[:-9]
                q_key = f"{base_block}.proj.q"
                if q_key in raw_bundles and "alpha" in raw_bundles[q_key]:
                    raw_bundles[clean]["alpha"] = raw_bundles[q_key]["alpha"]

    processed_bundles: Dict[str, Dict[str, Any]] = {}
    for clean, b in raw_bundles.items():
        has_diff = "diff" in b
        has_lokr = any(x in b for x in ["w1", "w2", "w1_a", "w1_b", "w2_a", "w2_b"])
        has_lora = "down" in b and "up" in b
        if not (has_diff or has_lokr or has_lora):
            continue

        processed: Dict[str, Any] = {}
        if "alpha" in b:
            processed["alpha"] = b["alpha"]

        if has_diff:
            processed["type"] = "diff"
            processed["diff"] = b["diff"]
            processed_bundles[clean] = processed
        elif has_lokr:
            processed["type"] = "lokr"
            w1 = b.get("w1")
            dim = None
            if w1 is None and "w1_a" in b and "w1_b" in b:
                w1 = torch.mm(b["w1_a"], b["w1_b"])
                dim = b["w1_b"].shape[0]
            w2 = b.get("w2")
            if w2 is None and "w2_a" in b and "w2_b" in b:
                w2 = torch.mm(b["w2_a"], b["w2_b"])
                dim = b["w2_b"].shape[0]
            if w1 is not None and w2 is not None:
                processed["w1"] = w1
                processed["w2"] = w2
                processed["dim"] = dim
                processed_bundles[clean] = processed
        elif has_lora:
            processed["type"] = "lora"
            processed["down"] = b["down"]
            processed["up"] = b["up"]
            processed_bundles[clean] = processed

    del raw_bundles
    gc.collect()
    return processed_bundles


def compute_bundle_delta(bundle: Dict[str, Any], strength: float, device: "torch.device") -> Optional["torch.Tensor"]:
    """Computes the 2D delta matrix [out_dim, in_dim] in FP32 on the target device."""
    if bundle["type"] == "diff":
        return bundle["diff"].to(device=device, dtype=torch.float32) * strength
    elif bundle["type"] == "lora":
        up = bundle["up"].to(device=device, dtype=torch.float32)
        down = bundle["down"].to(device=device, dtype=torch.float32)
        rank = down.shape[0]
        alpha = bundle.get("alpha", float(rank))
        scale = (alpha / float(rank)) * strength
        return torch.mm(up, down) * scale
    elif bundle["type"] == "lokr":
        w1 = bundle["w1"].to(device=device, dtype=torch.float32)
        w2 = bundle["w2"].to(device=device, dtype=torch.float32)
        dim = bundle.get("dim", None)
        alpha_val = bundle.get("alpha", None)
        alpha_scale = (alpha_val / float(dim)) if (alpha_val is not None and dim is not None) else 1.0
        if len(w2.shape) == 4 and len(w1.shape) == 2:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        return torch.kron(w1, w2) * (strength * alpha_scale)
    return None


def get_moss_qkv_unified_delta(
    bundles: Dict[str, Dict[str, Any]],
    target_key: str,
    strength: float,
    device: "torch.device",
) -> Optional["torch.Tensor"]:
    """Exact bi-directional QKV converter from Cell 5 (`get_moss_layer_delta_from_lora`)."""
    if target_key in bundles:
        return compute_bundle_delta(bundles[target_key], strength, device)

    if target_key.endswith(".proj.qkv"):
        base_block = target_key[:-9]
        q_k, k_k, v_k = f"{base_block}.proj.q", f"{base_block}.proj.k", f"{base_block}.proj.v"
        if any(x in bundles for x in (q_k, k_k, v_k)):
            dq = compute_bundle_delta(bundles[q_k], strength, device) if q_k in bundles else torch.zeros((3072, 3072), device=device, dtype=torch.float32)
            dk = compute_bundle_delta(bundles[k_k], strength, device) if k_k in bundles else torch.zeros((3072, 3072), device=device, dtype=torch.float32)
            dv = compute_bundle_delta(bundles[v_k], strength, device) if v_k in bundles else torch.zeros((3072, 3072), device=device, dtype=torch.float32)
            return torch.cat([dq, dk, dv], dim=0)

    for idx_slice, suffix in enumerate((".proj.q", ".proj.k", ".proj.v")):
        if target_key.endswith(suffix):
            base_block = target_key[:-len(suffix)]
            qkv_key = f"{base_block}.proj.qkv"
            if qkv_key in bundles:
                d_qkv = compute_bundle_delta(bundles[qkv_key], strength, device)
                if d_qkv is not None and d_qkv.shape[0] == 9216:
                    return d_qkv.chunk(3, dim=0)[idx_slice].contiguous()

    return None


def merge_multi_loras_svd(
    loras_cfg: List[Dict[str, Any]],
    output_lora_path: str,
    arch_id: str = "MOSS_Image",
    target_rank: int = 128,
    target_alpha: int = 128,
    clamp_quantile: float = 1.0,
    save_precision: str = "bf16",
    qkv_output_mode: str = "fused_qkv",
) -> Dict[str, Any]:
    """Unified Multi-LoRA / LoKr SVD Merger preserving exact structures for MOSS Image, Krea 2, & MiniMax H3."""
    svd_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.bfloat16 if save_precision.lower() == "bf16" else torch.float16

    loaded_loras = []
    discovered_blocks: Set[int] = set()
    extra_keys: Set[str] = set()
    all_unique_keys: Set[str] = set()

    for item in loras_cfg:
        path = item["path"].strip()
        strg = float(item["strength"])
        if not os.path.exists(path):
            continue
        bundles = load_adapter_bundles(path, arch_id=arch_id, for_baker=False)
        loaded_loras.append({"name": os.path.basename(path), "bundles": bundles, "strength": strg})
        all_unique_keys.update(bundles.keys())

        if arch_id == "MOSS_Image":
            for k in bundles.keys():
                m = re.match(r"^layers\.(\d+)\.proj\.(ffn_down|ffn_up_gate|o|qkv|q|k|v)$", k)
                if m:
                    discovered_blocks.add(int(m.group(1)))
                else:
                    extra_keys.add(k)

    if not loaded_loras:
        raise ValueError("No valid LoRA files found.")

    if arch_id == "MOSS_Image":
        target_keys = []
        projs = ["ffn_down", "ffn_up_gate", "k", "o", "q", "v"] if qkv_output_mode == "split_q_k_v" else ["ffn_down", "ffn_up_gate", "o", "qkv"]
        for b_idx in sorted(list(discovered_blocks)):
            for p_name in projs:
                target_keys.append(f"layers.{b_idx}.proj.{p_name}")
        for ek in sorted(list(extra_keys)):
            target_keys.append(ek)
        target_keys = sorted(target_keys)
    else:
        target_keys = sorted(list(all_unique_keys))

    merged_state_dict: Dict[str, "torch.Tensor"] = {}

    for t_key in target_keys:
        total_delta = None
        for lora in loaded_loras:
            if arch_id == "MOSS_Image":
                delta = get_moss_qkv_unified_delta(lora["bundles"], t_key, lora["strength"], svd_device)
            else:
                delta = compute_bundle_delta(lora["bundles"][t_key], lora["strength"], svd_device) if t_key in lora["bundles"] else None

            if delta is not None:
                if total_delta is None:
                    total_delta = delta
                else:
                    if total_delta.shape == delta.shape:
                        total_delta += delta
                    elif total_delta.shape == delta.T.shape:
                        total_delta += delta.T
                del delta

        if total_delta is None:
            continue

        if total_delta.ndim > 2:
            total_delta = total_delta.flatten(start_dim=1)

        out_dim, in_dim = total_delta.shape
        curr_rank = min(int(target_rank), out_dim, in_dim)

        if arch_id == "MOSS_Image":
            q_oversample = min(curr_rank + 8, out_dim, in_dim)
            try:
                U, S, V = torch.svd_lowrank(total_delta, q=q_oversample, niter=4)
            except Exception:
                U, S, V = torch.svd_lowrank(total_delta.cpu(), q=q_oversample, niter=4)
            U = U[:, :curr_rank]
            S = S[:curr_rank]
            Vh = V[:, :curr_rank].T
            sqrt_S = torch.sqrt(torch.clamp(S, min=0.0))
            lora_B = (U * sqrt_S.unsqueeze(0)).to("cpu")
            lora_A = (sqrt_S.unsqueeze(1) * Vh).to("cpu")
            if clamp_quantile < 1.0:
                dist = torch.cat([lora_B.flatten(), lora_A.flatten()])
                hi_val = torch.quantile(dist.abs(), clamp_quantile)
                lora_B = lora_B.clamp(-hi_val, hi_val)
                lora_A = lora_A.clamp(-hi_val, hi_val)

            full_base_name = t_key if t_key.startswith("diffusion_model.") else f"diffusion_model.{t_key}"
            merged_state_dict[f"{full_base_name}.lora_A.weight"] = lora_A.to(dtype=target_dtype).contiguous()
            merged_state_dict[f"{full_base_name}.lora_B.weight"] = lora_B.to(dtype=target_dtype).contiguous()
        else:
            try:
                U, S, V = torch.svd_lowrank(total_delta, q=curr_rank, niter=4)
                Vh = V.T
            except Exception:
                U, S, V = torch.svd_lowrank(total_delta.cpu(), q=curr_rank, niter=4)
                Vh = V.T
            scale_factor = float(curr_rank) / float(target_alpha)
            U = (U @ torch.diag(S)) * scale_factor
            U = U.to("cpu")
            Vh = Vh.to("cpu")
            if clamp_quantile < 1.0:
                dist = torch.cat([U.flatten(), Vh.flatten()])
                hi_val = torch.quantile(dist.abs(), clamp_quantile)
                U = U.clamp(-hi_val, hi_val)
                Vh = Vh.clamp(-hi_val, hi_val)

            base_name = t_key if t_key.startswith("diffusion_model.") else f"diffusion_model.{t_key}"
            merged_state_dict[f"{base_name}.lora_B.weight"] = U.to(dtype=target_dtype).contiguous()
            merged_state_dict[f"{base_name}.lora_A.weight"] = Vh.to(dtype=target_dtype).contiguous()
            merged_state_dict[f"{base_name}.alpha"] = torch.tensor(int(target_alpha), dtype=target_dtype)

        del total_delta

    out_name = os.path.splitext(os.path.basename(output_lora_path))[0]
    if arch_id == "MOSS_Image":
        metadata = MossImageSpec.get_lora_metadata(out_name, target_rank)
    elif arch_id == "MINIMAX_H3":
        metadata = MinimaxH3Spec.get_lora_metadata(target_rank, target_alpha)
    else:
        metadata = Krea2Spec.get_lora_metadata(target_rank, target_alpha)

    os.makedirs(os.path.dirname(output_lora_path) or ".", exist_ok=True)
    save_file(merged_state_dict, output_lora_path, metadata=metadata)
    return {
        "output_path": output_lora_path,
        "total_tensors": len(merged_state_dict),
        "size_mb": round(os.path.getsize(output_lora_path) / (1024 ** 2), 2),
        "metadata": metadata,
    }


def bake_loras_to_checkpoint(
    base_model_path: str,
    loras_cfg: List[Dict[str, Any]],
    output_path: str,
    precision: str = "bf16",
    krea2_mix_precision_bf16: bool = True,
    normalize_strengths: bool = False,
) -> Dict[str, Any]:
    """Executes the exact Universal LoRA Baker V5.2 from Cell 7."""
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    total_strength = sum(float(item["strength"]) for item in loras_cfg)
    loaded_loras = []
    for item in loras_cfg:
        path = item["path"].strip()
        if not os.path.exists(path):
            continue
        strg = float(item["strength"])
        if normalize_strengths and total_strength > 1.0:
            strg = strg / total_strength
        bundles = load_adapter_bundles(path, arch_id="KREA_2", for_baker=True)
        loaded_loras.append({
            "name": os.path.basename(path),
            "bundles": bundles,
            "strength": strg,
            "keys": set(bundles.keys()),
        })

    if not loaded_loras:
        raise ValueError("No valid LoRAs found to bake.")

    header_index: Dict[str, Any] = {}
    current_offset = 0
    fallback_dtype = torch.bfloat16 if precision in ("bf16", "fp8") else torch.float16
    fallback_dtype_str = "BF16" if fallback_dtype == torch.bfloat16 else "F16"

    with safe_open(base_model_path, framework="pt", device="cpu") as f_base:
        all_keys_sorted = sorted(f_base.keys())
        for k in all_keys_sorted:
            if "comfy_quant" in k:
                continue
            shape = list(f_base.get_slice(k).get_shape())
            num_el = 1
            for d in shape:
                num_el *= d

            if krea2_mix_precision_bf16:
                dtype_str, bpe = Krea2Spec.get_layer_dtype_spec(k, mode="mix_bf16_f32")
            else:
                dtype_str, bpe = fallback_dtype_str, 2

            byte_size = num_el * bpe
            header_index[k] = {
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [current_offset, current_offset + byte_size],
            }
            current_offset += byte_size

    header_bytes = encode_aligned_safetensors_header(header_index)
    modified_tensors = 0
    total_ops = 0

    with open(output_path, "wb") as f_out:
        f_out.write(header_bytes)
        with safe_open(base_model_path, framework="pt", device="cpu") as f_base:
            for k in all_keys_sorted:
                if k not in header_index:
                    continue
                base_tensor = f_base.get_tensor(k).float()
                total_delta = torch.zeros_like(base_tensor)
                modified = False

                for lora in loaded_loras:
                    mode, match_data = Krea2Spec.get_baker_flux_mapping(k, lora["keys"])
                    if mode == "direct" and match_data in lora["bundles"]:
                        delta = compute_bundle_delta(lora["bundles"][match_data], lora["strength"], torch.device("cpu"))
                        if delta is not None:
                            if delta.shape == base_tensor.shape:
                                total_delta += delta
                                modified = True
                                total_ops += 1
                            elif delta.T.shape == base_tensor.shape:
                                total_delta += delta.T
                                modified = True
                                total_ops += 1
                    elif mode == "qkv_split" and base_tensor.shape[0] % 3 == 0:
                        split_dim = base_tensor.shape[0] // 3
                        for idx_qkv, qkv_char in enumerate(("q", "k", "v")):
                            t_k = match_data[qkv_char]
                            if t_k in lora["bundles"]:
                                delta = compute_bundle_delta(lora["bundles"][t_k], lora["strength"], torch.device("cpu"))
                                if delta is not None and delta.shape == (split_dim, base_tensor.shape[1]):
                                    total_delta[idx_qkv * split_dim : (idx_qkv + 1) * split_dim, :] += delta
                                    modified = True
                                    total_ops += 1

                if modified:
                    base_tensor += total_delta
                    modified_tensors += 1

                out_dt = torch.float32 if header_index[k]["dtype"] == "F32" else fallback_dtype
                write_tensor_binary(f_out, base_tensor, out_dt)
                del base_tensor, total_delta

    return {
        "output_path": output_path,
        "modified_tensors": modified_tensors,
        "total_applied_operations": total_ops,
        "size_gb": round(os.path.getsize(output_path) / (1024 ** 3), 2),
    }


def extract_lora_svd_turbo(
    base_path: str,
    tuned_path: str,
    out_path: str,
    rank_val: int = 256,
    alpha_val: int = 256,
    clamp_q: float = 1.0,
    prec: str = "bf16",
) -> Dict[str, Any]:
    """Executes the exact Fast Low-Rank SVD LoRA Extractor from Cell 8."""
    if not os.path.exists(base_path) or not os.path.exists(tuned_path):
        raise FileNotFoundError("Base or Fine-Tuned model file not found.")

    target_precision = torch.bfloat16 if prec.lower() == "bf16" else torch.float16
    svd_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_sd: Dict[str, "torch.Tensor"] = {}

    with safe_open(base_path, framework="pt", device="cpu") as f_base, safe_open(
        tuned_path, framework="pt", device="cpu"
    ) as f_tuned:
        common_keys = sorted(list(set(f_base.keys()).intersection(set(f_tuned.keys()))))
        target_keys = [
            k
            for k in common_keys
            if k.endswith(".weight")
            and len(f_base.get_slice(k).get_shape()) == 2
            and not any(x in k.lower() for x in ["norm", "bias", "embed", "time_mod"])
        ]

        for key in target_keys:
            w_base = f_base.get_tensor(key).float()
            w_tuned = f_tuned.get_tensor(key).float()
            diff = w_tuned - w_base
            del w_base, w_tuned

            if diff.abs().max() < 1e-5:
                del diff
                continue

            out_dim, in_dim = diff.shape[0], diff.shape[1]
            actual_rank = min(int(rank_val), out_dim, in_dim)
            diff_gpu = diff.to(svd_device)
            try:
                U, S, V = torch.svd_lowrank(diff_gpu, q=actual_rank, niter=6)
                Vh = V.t()
            except Exception:
                U, S, V = torch.svd_lowrank(diff.cpu(), q=actual_rank, niter=6)
                Vh = V.t()
            del diff_gpu, diff

            scale_factor = float(actual_rank) / float(alpha_val)
            sqrt_s = torch.sqrt(torch.clamp(S, min=0.0))
            U = ((U * sqrt_s.unsqueeze(0)) * (scale_factor ** 0.5)).cpu()
            Vh = ((sqrt_s.unsqueeze(1) * Vh) * (scale_factor ** 0.5)).cpu()

            if clamp_q < 1.0:
                hi_u = torch.quantile(U.abs(), clamp_q)
                U = U.clamp(-hi_u, hi_u)
                hi_vh = torch.quantile(Vh.abs(), clamp_q)
                Vh = Vh.clamp(-hi_vh, hi_vh)

            clean_key = key[:-7] if key.endswith(".weight") else key
            if not clean_key.startswith("diffusion_model."):
                clean_key = clean_key[6:] if clean_key.startswith("model.diffusion_model.") else f"diffusion_model.{clean_key}"

            lora_sd[f"{clean_key}.lora_B.weight"] = U.to(dtype=target_precision).contiguous()
            lora_sd[f"{clean_key}.lora_A.weight"] = Vh.to(dtype=target_precision).contiguous()
            lora_sd[f"{clean_key}.alpha"] = torch.tensor(alpha_val, dtype=torch.float32)

    metadata = Krea2Spec.get_lora_metadata(rank_val, alpha_val)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(lora_sd, out_path, metadata=metadata)
    return {
        "output_path": out_path,
        "extracted_layers": len(lora_sd) // 3,
        "size_mb": round(os.path.getsize(out_path) / (1024 ** 2), 2),
    }
