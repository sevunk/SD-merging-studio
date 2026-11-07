"""Immutable Architecture Blueprint for MINIMAX H3 (minimax-h3).

Preserves 100% of the exact extracted layer structures from:
- Cell 12: DARE Hydra Merger - MiniMax H3 Edition (Dynamic Key Translator, INT8 Safe & Native Dtype Preservation)
- Cell 13: Universal Multi-LoRA Merger V3.1 - MiniMax H3 Edition
"""

from __future__ import annotations
import re
from typing import Dict, Optional, Sequence, Set
from .base_spec import ArchitectureSpec


class MinimaxH3Spec(ArchitectureSpec):
    arch_id = "MINIMAX_H3"
    display_name = "MiniMax H3 (Ref2VA / Hybrid INT8)"
    badge_color = "rose"
    convrot_group_size = 256
    supports_int8_convrot = True
    supports_gguf = False
    supports_block_guided = False
    default_merge_mode = "DARE_Hydra_Minimax"
    supported_merge_modes = ("DARE_Hydra_Minimax", "True_Disjoint_TIES", "Pure_Weighted_Linear")

    # Exact protected keywords from Cell 12
    protected_keywords = ("norm", "embed", "bias", "modulation", "scale", "quant", "adaln")

    @classmethod
    def get_translated_key(cls, raw_k: str, base_keys: Set[str]) -> Optional[str]:
        """Dynamic key translator from Cell 12 (`get_translated_key`) + cross-prefix translation
        (e.g. 'net.' <-> 'model.diffusion_model.').
        """
        if raw_k in base_keys:
            return raw_k

        prefixes_to_strip = ["model.diffusion_model.", "diffusion_model.", "net.", "model."]
        stripped_core = raw_k
        for p in prefixes_to_strip:
            if raw_k.startswith(p):
                stripped = raw_k[len(p):]
                stripped_core = stripped
                if stripped in base_keys:
                    return stripped
                break

        for p in prefixes_to_strip:
            added = p + raw_k
            if added in base_keys:
                return added
            added_core = p + stripped_core
            if added_core in base_keys:
                return added_core

        return None

    @classmethod
    def normalize_lora_clean_key(cls, key: str) -> str:
        """Exact LoRA key normalizer from Cell 13 (`normalize_clean_key`)."""
        k_lower = key.lower()
        prefixes = [
            "model.diffusion_model.",
            "diffusion_model.",
            "base_model.model.",
            "model.",
            "unet.",
        ]
        for p in prefixes:
            if k_lower.startswith(p):
                k_lower = k_lower[len(p):]
                break

        lora_suffixes = [
            r"[\._]lora[_\.](up|down|alpha|a|b)[\._]weight$",
            r"[\._]lora[_\.](up|down|alpha|a|b)$",
            r"[\._]weight$",
            r"[\._]bias$",
        ]
        for pattern in lora_suffixes:
            k_lower = re.sub(pattern, "", k_lower)

        return k_lower.strip("._")

    @classmethod
    def get_lora_metadata(cls, rank_target: int, alpha_target: int) -> Dict[str, str]:
        """Exact Tensor.Art deployable MiniMax-H3 LoRA metadata from Cell 13."""
        return {
            "modelspec.architecture": "minimax/lora",
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(rank_target),
            "ss_network_alpha": str(int(alpha_target)),
            "modelspec.sai_model_spec": "1.0.0",
            "ss_base_model_version": "minimax-h3",
        }
