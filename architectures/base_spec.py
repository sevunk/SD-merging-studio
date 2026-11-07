"""Base contract for isolated model architecture plugins in Merging Studio v2.0.

Adding a new model architecture in the future ONLY requires creating a new file
in `architectures/` that subclasses `ArchitectureSpec` and registering it in
`architectures/__init__.py`. Existing architectures and core engines remain
100% untouched.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class BlockZoneSpec:
    """Defines a named block zone for block-guided merging."""
    zone_id: str
    label_id: str
    label_en: str
    start_block: int
    end_block: int
    default_strength: float
    description_id: str
    description_en: str


class ArchitectureSpec:
    """Immutable blueprint contract for a Diffusion/DiT model architecture."""

    arch_id: str = "BASE"
    display_name: str = "Base Architecture"
    badge_color: str = "indigo"
    convrot_group_size: int = 256
    supports_int8_convrot: bool = False
    supports_gguf: bool = False
    supports_block_guided: bool = False
    default_merge_mode: str = "DARE_Hydra"
    supported_merge_modes: Tuple[str, ...] = ("DARE_Hydra",)
    protected_keywords: Tuple[str, ...] = ("norm", "bias", "embed", "modulation")
    known_prefixes: Tuple[str, ...] = (
        "model.diffusion_model.transformer.",
        "model.diffusion_model.",
        "diffusion_model.transformer.",
        "diffusion_model.",
        "base_model.model.",
        "model.transformer.",
        "transformer.",
        "model.",
        "unet.",
        "net.",
    )
    block_zones: Tuple[BlockZoneSpec, ...] = field(default_factory=tuple)

    @classmethod
    def detect(cls, all_keys: Sequence[str]) -> bool:
        """Returns True if the given state_dict keys match this architecture."""
        return False

    @classmethod
    def strip_prefix(cls, raw_key: str) -> str:
        """Strips known wrapper prefixes from a tensor key."""
        for p in cls.known_prefixes:
            if raw_key.startswith(p):
                return raw_key[len(p):]
        return raw_key

    @classmethod
    def is_quant_metadata_key(cls, raw_key: str) -> bool:
        """Returns True if the key is an auxiliary quantization or attention config tensor."""
        return (
            raw_key.endswith(".comfy_quant")
            or raw_key.endswith(".weight_scale")
            or raw_key.endswith(".comfy_attention.config")
            or "comfy_attention.config" in raw_key
        )

    @classmethod
    def is_protected_layer(cls, clean_key: str, ndim: int) -> bool:
        """Returns True if a layer must be protected from TIES sign trimming & DARE drop."""
        if ndim < 2:
            return True
        k_low = clean_key.lower()
        return any(kw in k_low for kw in cls.protected_keywords)

    @classmethod
    def get_layer_dtype_spec(cls, clean_key: str, mode: str = "bf16") -> Tuple[str, int]:
        """Returns (safetensors_dtype_str, bytes_per_element) for output."""
        return ("BF16", 2)
