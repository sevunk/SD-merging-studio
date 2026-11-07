"""Immutable Architecture Blueprint for ANIMA (Anima-Turbo-DiT).

Preserves 100% of the exact extracted layer structures from:
- Cell 11: DARE Hydra Merger - Special Anima Edition (Auto Key Translator 'net.' -> 'model.diffusion_model.')
"""

from __future__ import annotations
from typing import Dict, Sequence, Tuple
from .base_spec import ArchitectureSpec


class AnimaSpec(ArchitectureSpec):
    arch_id = "ANIMA"
    display_name = "Anima Turbo DiT"
    badge_color = "amber"
    convrot_group_size = 256
    supports_int8_convrot = False
    supports_gguf = False
    supports_block_guided = False
    default_merge_mode = "DARE_Hydra_Anima"
    supported_merge_modes = ("DARE_Hydra_Anima", "True_Disjoint_TIES", "Pure_Weighted_Linear")

    # Exact protected keywords from Cell 11
    protected_keywords = ("norm", "embed", "bias", "modulation")

    @classmethod
    def detect(cls, all_keys: Sequence[str]) -> bool:
        for k in all_keys:
            if k.startswith("net."):
                return True
        return False

    @classmethod
    def translate_to_base_key(cls, model_key: str) -> str:
        """Exact translator from Cell 11 (`translate_to_base_key`):
        Translates Ultrareal 'net.' prefix to ComfyUI 'model.diffusion_model.' format.
        """
        if model_key.startswith("net."):
            return "model.diffusion_model." + model_key[4:]
        return model_key

    @classmethod
    def get_checkpoint_metadata(cls) -> Dict[str, str]:
        return {
            "format": "pt",
            "model_type": "Anima-Turbo-DiT",
            "merged_by": "DARE_HYDRA_Anima_Edition",
        }
