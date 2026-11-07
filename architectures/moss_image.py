"""Immutable Architecture Blueprint for MOSS Image (moss_image).

Preserves 100% of the exact extracted layer structures from:
- Cell 3: MOSS Image True Disjoint-TIES Merger v2.0
- Cell 4: MOSS Image Block-Guided Norm-Preserving Merger v3.0
- Cell 5: MOSS Image Multi-LoRA & LoKr SVD Merger (Fused QKV <-> Split Q/K/V)
- Cell 6: MOSS Image Pre-Merge Tensor Analyzer v2.0
- Cell 15: Universal Unquantizer v5.0 (MOSS rope.bands [16] F32 restoration)
- Cell 16: Universal INT8 ConvRot Native Quantizer (MOSS 757-tensor blueprint)
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple
from .base_spec import ArchitectureSpec, BlockZoneSpec


class MossImageSpec(ArchitectureSpec):
    arch_id = "MOSS_Image"
    display_name = "MOSS Image (42-Layer DiT)"
    badge_color = "emerald"
    convrot_group_size = 256
    supports_int8_convrot = True
    supports_gguf = False
    supports_block_guided = True
    default_merge_mode = "Block_Guided_V3"
    supported_merge_modes = (
        "Block_Guided_V3",
        "True_Disjoint_TIES",
        "Norm_Preserving_TIES",
        "Pure_Weighted_Linear",
    )

    # Exact signatures from Cell 15 & Cell 16
    SIGNATURE_KEYWORDS = (
        "final_linear_video",
        "video_head_modulation",
        "shared_block_modulation",
        "text_embedder.blocks",
    )

    # Exact protected keywords from Cell 3, Cell 4, & Cell 6
    protected_keywords = (
        "norm",
        "bias",
        "modulation",
        "time_embedding",
        "video_embedder",
        "final_linear_video",
        "text_embedder.proj",
    )

    known_prefixes = (
        "model.diffusion_model.transformer.",
        "model.diffusion_model.",
        "diffusion_model.transformer.",
        "diffusion_model.",
        "net.",
        "model.",
    )

    # Exact 42-Layer Block-Guided regions from Cell 4 & Cell 6
    block_zones = (
        BlockZoneSpec(
            zone_id="early",
            label_id="Early Blocks (0–13) — Lensa, Perspektif & Pose",
            label_en="Early Blocks (0–13) — Camera Lens, 3D Perspective & Pose",
            start_block=0,
            end_block=13,
            default_strength=0.55,
            description_id="Menjaga lensa dinamis & komposisi panggung dari Base Model.",
            description_en="Preserves dynamic camera lens & composition from Base Model.",
        ),
        BlockZoneSpec(
            zone_id="mid",
            label_id="Mid Blocks (14–27) — Anatomi, Bentuk Tubuh & Objek",
            label_en="Mid Blocks (14–27) — Anatomy, Body Structure & Objects",
            start_block=14,
            end_block=27,
            default_strength=0.90,
            description_id="Menyerap kemampuan anatomi & struktur utama dari Model Donor.",
            description_en="Absorbs core anatomy & structural capabilities from Donor Models.",
        ),
        BlockZoneSpec(
            zone_id="late",
            label_id="Late Blocks (28–41) — Pori-Pori Kulit, Cahaya & Tekstur Mikro",
            label_en="Late Blocks (28–41) — Skin Pores, Lighting & Micro-Texture",
            start_block=28,
            end_block=41,
            default_strength=0.30,
            description_id="Melindungi tekstur pori-pori kulit & detail mikro Base Official.",
            description_en="Protects skin pore realism & micro-details of Base Official.",
        ),
    )

    # Exact INT8 ConvRot payloads from Cell 16
    COMFY_QUANT_PAYLOAD: bytes = json.dumps({
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }).encode("utf-8")  # Exactly 72 bytes ([72], U8)

    ATTN_CONFIG_PAYLOAD: bytes = json.dumps({
        "attention": "comfy_kitchen_int8"
    }).encode("utf-8")  # Exactly 35 bytes ([35], U8)

    @classmethod
    def detect(cls, all_keys: Sequence[str]) -> bool:
        for k in all_keys:
            if any(sig in k for sig in cls.SIGNATURE_KEYWORDS):
                return True
        return False

    @classmethod
    def strip_prefix(cls, raw_key: str) -> str:
        for p in ("model.diffusion_model.", "diffusion_model.", "net.", "model."):
            if raw_key.startswith(p):
                return raw_key[len(p):]
        return raw_key

    @classmethod
    def get_layer_dtype_spec(cls, clean_key: str, mode: str = "bf16") -> Tuple[str, int]:
        """In official MOSS Image (moss_image_sft_bf16), ONLY rope.bands is F32 (4 bytes).
        All other weights and biases are BF16 (2 bytes).
        """
        if clean_key == "rope.bands" or clean_key.endswith("rope.bands"):
            return ("F32", 4)
        return ("BF16", 2)

    @classmethod
    def is_int8_convrot_target(cls, clean_key: str, shape: Sequence[int]) -> bool:
        """Exact rule from Cell 16: ONLY the 168 layers in
        layers.{0..41}.proj.{ffn_down, ffn_up_gate, o, qkv}.weight are INT8.
        """
        is_moss_proj_weight = (
            clean_key.startswith("layers.")
            and ".proj." in clean_key
            and clean_key.endswith(".weight")
            and "norm" not in clean_key
            and "scale" not in clean_key
        )
        return (
            is_moss_proj_weight
            and len(shape) == 2
            and shape[1] % cls.convrot_group_size == 0
        )

    @classmethod
    def get_block_strength(
        cls,
        clean_key: str,
        early_strength: float = 0.55,
        mid_strength: float = 0.90,
        late_strength: float = 0.30,
        other_strength: float = 0.50,
    ) -> float:
        """Exact block strength router from Cell 4."""
        if clean_key.startswith("layers."):
            parts = clean_key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1])
                if idx <= 13:
                    return early_strength
                elif idx <= 27:
                    return mid_strength
                else:
                    return late_strength
        return other_strength

    @classmethod
    def classify_region(cls, clean_key: str) -> str:
        """Exact 5-region architecture classifier from Cell 6."""
        if clean_key.startswith("text_embedder"):
            return "1. Text Embedder (Pemahaman Prompt)"
        elif clean_key.startswith("layers."):
            parts = clean_key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1])
                if idx <= 13:
                    return "2. Early DiT Blocks 0-13 (Komposisi & Pose/Anatomi)"
                elif idx <= 27:
                    return "3. Mid DiT Blocks 14-27 (Struktur & Bentuk Objek)"
                else:
                    return "4. Late DiT Blocks 28-41 (Tekstur Kulit, Lighting & Detail)"
        return "5. Modulation, Time & Video Head (Gaya Global & Kontras)"

    @classmethod
    def normalize_lora_clean_key(cls, key: str) -> str:
        """Exact LoRA/LoKr key normalizer from Cell 5 (`normalize_moss_clean_key`)."""
        k = key.strip()
        prefixes = [
            "model.diffusion_model.transformer.",
            "model.diffusion_model.",
            "diffusion_model.transformer.",
            "diffusion_model.",
            "base_model.model.",
            "transformer.",
            "model.",
            "lora_unet_",
        ]
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break

        if k.startswith("layers_"):
            k = re.sub(r"^layers_(\d+)_proj_", r"layers.\1.proj.", k)

        suffix_patterns = [
            r"\.lora_[ABab](\.default)?(\.weight)?$",
            r"\.lora_(up|down)(\.default)?(\.weight)?$",
            r"\.lokr_(w1|w2|w1_a|w1_b|w2_a|w2_b|t2)(\.weight)?$",
            r"\.diff(\.weight)?$",
            r"\.alpha$",
            r"\.lora_alpha$",
        ]
        for pat in suffix_patterns:
            k = re.sub(pat, "", k)

        return k.strip("._")

    @classmethod
    def get_lora_metadata(cls, out_name: str, rank_target: int) -> Dict[str, str]:
        """Exact ai-toolkit compatible MOSS Image LoRA metadata from Cell 5."""
        return {
            "format": "pt",
            "name": out_name,
            "ss_output_name": out_name,
            "ss_base_model_version": "moss_image",
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(rank_target),
            "ss_network_alpha": str(rank_target),
            "software": json.dumps({
                "name": "ai-toolkit",
                "repo": "https://github.com/ostris/ai-toolkit",
                "version": "0.13.34",
            }),
            "version": "1.0",
        }
