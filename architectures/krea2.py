"""Immutable Architecture Blueprint for KREA 2 (Krea2_MMDiT / qwen_image).

Preserves 100% of the exact extracted layer structures from:
- Cell 7: Universal LoRA Baker V5.2 (Full ModelScope & Diffusers Alias Support + Mixed Precision)
- Cell 8: Fast LoRA Extractor from Krea 2
- Cell 9: Universal Multi-LoRA Merger V3.1 (mlp.down recovery & TensorArt metadata)
- Cell 10: DARE Hydra Merger - Krea 2 Exclusive Edition (+ Fixed INT8 ConvRot Un-rotate/Re-rotate)
- Cell 14: Krea 2 to GGUF Converter v3.0 (BYTESHAPE_BLOCK_QUANTS & True_Q4_K_M)
- Cell 16 & 17: Krea 2 INT8 ConvRot Native Quantizer (Exact 12.56GB Tensor.Art layout)
- Official ComfyUI ModelMergeKrea2 & LCKrea2SemanticVectorTuner 28-block zones (0-8, 9-17, 18-27 + txtfusion)
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple
from .base_spec import ArchitectureSpec, BlockZoneSpec


class Krea2Spec(ArchitectureSpec):
    arch_id = "KREA_2"
    display_name = "KREA 2 (28-Block Single-Stream DiT + TextFusion)"
    badge_color = "violet"
    convrot_group_size = 256
    supports_int8_convrot = True
    supports_gguf = True
    supports_block_guided = True
    default_merge_mode = "DARE_Hydra_Krea2"
    supported_merge_modes = (
        "DARE_Hydra_Krea2",
        "Krea2_Zone_Guided_DARE",
        "True_Disjoint_TIES",
        "Norm_Preserving_TIES",
        "Pure_Weighted_Linear",
    )

    SIGNATURE_KEYWORDS = (
        "txtfusion.",
        "txtmlp.",
        "tmlp.",
        "tproj.",
        "qknorm.qnorm.scale",
    )

    # Exact protected keywords from Cell 10 (`merge_krea2_models`)
    protected_keywords = (
        "norm",
        "mod",
        "bias",
        "projector",
        "first",
        "last",
    )

    # Exact F32 keywords for Tensor.Art Mix Precision from Cell 7 & Cell 15
    F32_MIX_KEYWORDS = (
        "scale",
        "bias",
        "mod.lin",
        "modulation.lin",
        "tmlp",
        "tproj",
        "projector",
        "last.linear.weight",
        "txtmlp",
    )

    # Validated 28-Block + TextFusion zones from ComfyUI ModelMergeKrea2 & LCKrea2SemanticVectorTuner
    block_zones = (
        BlockZoneSpec(
            zone_id="low_blocks",
            label_id="Low Zone (blocks.0–8) — Struktur Dasar & Komposisi",
            label_en="Low Zone (blocks.0–8) — Core Structure & Layout",
            start_block=0,
            end_block=8,
            default_strength=1.0,
            description_id="Mengatur struktur komposisi awal Krea 2 (blocks.0 s/d blocks.8).",
            description_en="Controls early structural layout of Krea 2 (blocks.0 to blocks.8).",
        ),
        BlockZoneSpec(
            zone_id="mid_blocks",
            label_id="Mid Zone (blocks.9–17) — Anatomi, Karakter & Pose",
            label_en="Mid Zone (blocks.9–17) — Anatomy, Subject & Pose",
            start_block=9,
            end_block=17,
            default_strength=1.0,
            description_id="Mengatur anatomi subjek & interaksi konsep (blocks.9 s/d blocks.17).",
            description_en="Controls subject anatomy & core concepts (blocks.9 to blocks.17).",
        ),
        BlockZoneSpec(
            zone_id="high_blocks",
            label_id="High Zone (blocks.18–27) — Detail Halus & Pencahayaan",
            label_en="High Zone (blocks.18–27) — Fine Details & Lighting",
            start_block=18,
            end_block=27,
            default_strength=1.0,
            description_id="Mengatur tekstur permukaan & pencahayaan akhir (blocks.18 s/d blocks.27).",
            description_en="Controls surface micro-textures & final lighting (blocks.18 to blocks.27).",
        ),
    )

    # Exact 104-byte hard-padded .comfy_quant payload from Cell 16 & Cell 17
    _RAW_QUANT_JSON = json.dumps(
        {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    COMFY_QUANT_PAYLOAD: bytes = _RAW_QUANT_JSON + b" " * (104 - len(_RAW_QUANT_JSON))

    # Exact GGUF block quantization table from Cell 14
    BLOCK_WEIGHT_SUFFIXES = (
        "attn.gate.weight",
        "attn.wk.weight",
        "attn.wo.weight",
        "attn.wq.weight",
        "attn.wv.weight",
        "mlp.down.weight",
        "mlp.gate.weight",
        "mlp.up.weight",
    )

    BYTESHAPE_BLOCK_QUANTS: Dict[int, Tuple[str, ...]] = {
        0:  ("Q3_K",   "Q5_K",   "Q5_K",   "Q3_K",   "Q6_K",   "Q5_K",   "Q3_K",   "IQ4_XS"),
        1:  ("IQ4_XS", "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "Q3_K",   "Q3_K"),
        2:  ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q3_K"),
        3:  ("IQ4_XS", "IQ4_XS", "IQ4_XS", "Q3_K",   "IQ4_XS", "Q3_K",   "Q3_K",   "Q3_K"),
        4:  ("IQ4_XS", "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "Q3_K",   "Q3_K",   "Q3_K"),
        5:  ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "Q3_K",   "Q3_K"),
        6:  ("Q3_K",   "Q5_K",   "IQ4_XS", "Q3_K",   "Q5_K",   "Q3_K",   "Q3_K",   "Q3_K"),
        7:  ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q3_K"),
        8:  ("Q3_K",   "Q5_K",   "IQ4_XS", "IQ4_XS", "IQ4_XS", "Q3_K",   "IQ4_XS", "Q3_K"),
        9:  ("IQ4_XS", "IQ4_XS", "IQ4_XS", "IQ4_XS", "Q5_K",   "Q3_K",   "IQ4_XS", "Q3_K"),
        10: ("Q3_K",   "IQ4_XS", "Q3_K",   "Q3_K",   "Q5_K",   "IQ4_XS", "Q3_K",   "Q3_K"),
        11: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "IQ4_XS", "Q3_K"),
        12: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "IQ4_XS", "IQ4_XS", "IQ4_XS", "IQ4_XS"),
        13: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "IQ4_XS", "Q3_K"),
        14: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "Q3_K",   "IQ4_XS"),
        15: ("Q3_K",   "IQ4_XS", "IQ4_XS", "IQ4_XS", "Q5_K",   "Q3_K",   "Q3_K",   "Q3_K"),
        16: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "IQ4_XS", "Q3_K",   "Q3_K",   "IQ4_XS"),
        17: ("IQ4_XS", "Q3_K",   "IQ4_XS", "IQ4_XS", "Q5_K",   "Q3_K",   "IQ4_XS", "Q3_K"),
        18: ("Q3_K",   "Q5_K",   "IQ4_XS", "IQ4_XS", "Q6_K",   "IQ4_XS", "Q3_K",   "IQ4_XS"),
        19: ("IQ4_XS", "Q5_K",   "IQ4_XS", "IQ4_XS", "Q5_K",   "IQ4_XS", "Q3_K",   "Q3_K"),
        20: ("IQ4_XS", "Q5_K",   "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "IQ4_XS", "Q3_K"),
        21: ("IQ4_XS", "Q5_K",   "IQ4_XS", "IQ4_XS", "Q5_K",   "IQ4_XS", "IQ4_XS", "IQ4_XS"),
        22: ("IQ4_XS", "Q5_K",   "IQ4_XS", "IQ4_XS", "Q6_K",   "IQ4_XS", "Q3_K",   "IQ4_XS"),
        23: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q6_K",   "IQ4_XS", "Q3_K",   "IQ4_XS"),
        24: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "Q3_K",   "Q3_K",   "Q3_K"),
        25: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "Q3_K",   "IQ4_XS"),
        26: ("Q3_K",   "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "IQ4_XS", "Q3_K",   "IQ4_XS"),
        27: ("IQ4_XS", "IQ4_XS", "IQ4_XS", "Q3_K",   "Q5_K",   "Q5_K",   "Q3_K",   "IQ4_XS"),
    }

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
    def get_layer_dtype_spec(cls, clean_key: str, mode: str = "mix_bf16_f32") -> Tuple[str, int]:
        """Exact Tensor.Art Mix Precision rule from Cell 7 & Cell 15."""
        c_key = cls.strip_prefix(clean_key)
        if mode == "pure_bf16":
            return ("BF16", 2)
        if mode == "int8_convrot_non_quant":
            if "scale" in c_key.lower():
                return ("F32", 4)
            return ("BF16", 2)
        # Default mix_bf16_f32 (Cell 7 & Cell 15)
        if c_key == "first.weight" or c_key.endswith("first.weight"):
            return ("BF16", 2)
        if any(kw in c_key.lower() for kw in cls.F32_MIX_KEYWORDS):
            return ("F32", 4)
        return ("BF16", 2)

    @classmethod
    def is_int8_convrot_target(cls, raw_or_clean_key: str, shape: Sequence[int]) -> bool:
        """Exact 12.56GB Tensor.Art targeting logic from Cell 16 & Cell 17 (`get_exact_target_type`)."""
        clean_key = cls.strip_prefix(raw_or_clean_key)
        is_block_weight = (
            clean_key.startswith("blocks.")
            and (".attn." in clean_key or ".mlp." in clean_key)
            and clean_key.endswith(".weight")
            and "scale" not in clean_key
            and "norm" not in clean_key
        )
        return (
            is_block_weight
            and len(shape) == 2
            and shape[1] % cls.convrot_group_size == 0
        )

    @classmethod
    def get_gguf_qtype(cls, clean_key: str, profile: str) -> str:
        """Exact GGUF quantization type selector from Cell 14 (`get_krea2_gguf_qtype`)."""
        c_key = cls.strip_prefix(clean_key)
        if c_key.startswith("blocks."):
            parts = c_key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                b_idx = int(parts[1])
                sub_key = parts[2]
                if b_idx in cls.BYTESHAPE_BLOCK_QUANTS and sub_key in cls.BLOCK_WEIGHT_SUFFIXES:
                    w_idx = cls.BLOCK_WEIGHT_SUFFIXES.index(sub_key)
                    base_qtype = cls.BYTESHAPE_BLOCK_QUANTS[b_idx][w_idx]
                    if profile == "True_Q4_K_M_HighQuality_8.28GiB" and base_qtype in ("Q3_K", "IQ4_XS"):
                        return "Q4_K"
                    return base_qtype

        if c_key.startswith("txtfusion.") and (".attn." in c_key or ".mlp." in c_key) and c_key.endswith(".weight"):
            return "BF16"

        return "F32"

    @classmethod
    def get_zone_strength(
        cls,
        clean_key: str,
        low_strength: float = 1.0,
        mid_strength: float = 1.0,
        high_strength: float = 1.0,
        txtfusion_strength: float = 1.0,
    ) -> float:
        """Safe 28-block + TextFusion zone router for Krea 2 based on ModelMergeKrea2 & LCKrea2SemanticVectorTuner.
        Always keeps first., last., tmlp., txtmlp., and tproj. at 1.0 multiplier to prevent latent garbage!
        """
        c_key = cls.strip_prefix(clean_key)
        if c_key.startswith("blocks."):
            parts = c_key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                b_idx = int(parts[1])
                if b_idx <= 8:
                    return low_strength
                elif b_idx <= 17:
                    return mid_strength
                else:
                    return high_strength
        elif c_key.startswith("txtfusion."):
            return txtfusion_strength
        return 1.0

    @classmethod
    def normalize_baker_key(cls, key: str) -> str:
        """Exact key normalizer from Cell 7 (`Universal LoRA Baker V5.2`)."""
        key = key.lower()
        for suffix in [".weight", ".bias", ".default"]:
            if key.endswith(suffix):
                key = key[:-len(suffix)]

        lora_suffixes = [
            r"[\._]lora[_\.](up|down|alpha|a|b)$",
            r"[\._]lokr[_\.](w1|w2|w1_a|w1_b|w2_a|w2_b|alpha|t2)$",
            r"[\._](up|down|alpha|a|b)$",
            r"[\._]diff$",
        ]
        for pattern in lora_suffixes:
            key = re.sub(pattern, "", key)

        prefixes = [
            "model.diffusion_model.transformer.",
            "model.diffusion_model.",
            "base_model.model.",
            "diffusion_model.transformer.",
            "diffusion_model.",
            "model.transformer.",
            "model.",
            "transformer.",
            "unet.",
            "lora_unet.",
            "lora_unet_",
            "te_",
        ]
        stripped = True
        while stripped:
            stripped = False
            for p in prefixes:
                if key.startswith(p):
                    key = key[len(p):]
                    stripped = True
                    break

        key = key.replace("transformer_blocks", "blocks")
        key = key.replace("text_fusion", "txtfusion")
        key = key.replace("ff.down", "mlp.down")
        key = key.replace("ff.up", "mlp.up")
        key = key.replace("ff.gate", "mlp.gate")
        key = key.replace("_", ".")
        return key.strip("._")

    @classmethod
    def get_baker_flux_mapping(cls, base_key: str, lora_clean_keys: Set[str]):
        """Exact Diffusers/ModelScope <-> Native Krea 2 mapping from Cell 7 (`get_flux_mapping`)."""
        base_clean = cls.normalize_baker_key(base_key)
        if "adaln" in base_clean or "modulation" in base_clean:
            return None, None

        for target_suffix, aliases in [
            (".wq", [".wq", ".to.q", ".q"]),
            (".wk", [".wk", ".to.k", ".k"]),
            (".wv", [".wv", ".to.v", ".v"]),
            (".wo", [".wo", ".to.out.0", ".to.out", ".out"]),
            (".gate", [".gate", ".to.gate"]),
        ]:
            if base_clean.endswith(target_suffix):
                prefix = base_clean[:-len(target_suffix)]
                for alias in aliases:
                    candidate = prefix + alias
                    if candidate in lora_clean_keys:
                        return "direct", candidate

        direct_aliases = {
            "first": ["img.in", "img_in"],
            "last.linear": ["final.layer.linear", "final_layer.linear"],
            "txtmlp.1": ["txt.in.linear.1", "txt_in.linear_1"],
            "txtmlp.3": ["txt.in.linear.2", "txt_in.linear_2"],
            "tmlp.0": ["time.embed.linear.1", "time_embed.linear_1"],
            "tmlp.2": ["time.embed.linear.2", "time_embed.linear_2"],
            "tproj.1": ["time.mod.proj", "time_mod_proj"],
            "txtfusion.projector": ["txtfusion.projector", "text_fusion.projector"],
        }
        if base_clean in direct_aliases:
            for alias in direct_aliases[base_clean]:
                if alias in lora_clean_keys:
                    return "direct", alias

        if ".qkv" in base_clean or base_clean.endswith("qkv"):
            prefix_attn = base_clean.rsplit(".qkv", 1)[0] if ".qkv" in base_clean else base_clean[:-3]
            target_q = prefix_attn + ".to.q"
            target_k = prefix_attn + ".to.k"
            target_v = prefix_attn + ".to.v"
            if target_q in lora_clean_keys or target_k in lora_clean_keys or target_v in lora_clean_keys:
                return "qkv_split", {"q": target_q, "k": target_k, "v": target_v}

        if base_clean in lora_clean_keys:
            return "direct", base_clean

        return None, None

    @classmethod
    def normalize_lora_merger_key(cls, key: str) -> str:
        """Exact LoRA SVD Merger V3.1 normalizer from Cell 9 (`normalize_clean_key`)
        that preserves `mlp.down` and underscore format on `txtfusion.layerwise_blocks`.
        """
        k_lower = key.lower()
        prefixes = [
            "model.diffusion_model.transformer.",
            "model.diffusion_model.",
            "base_model.model.",
            "diffusion_model.transformer.",
            "diffusion_model.",
            "model.transformer.",
            "model.",
            "transformer.",
            "unet.",
            "lora_unet.",
            "lora_unet_",
            "te_",
        ]
        for p in prefixes:
            if k_lower.startswith(p):
                k_lower = k_lower[len(p):]
                break

        lora_suffixes = [
            r"[\._]lora[_\.](up|down|alpha|a|b)[\._]weight$",
            r"[\._]lora[_\.](up|down|alpha|a|b)$",
            r"[\._]lokr[_\.](w1|w2|w1_a|w1_b|w2_a|w2_b|alpha|t2)[\._]weight$",
            r"[\._]lokr[_\.](w1|w2|w1_a|w1_b|w2_a|w2_b|alpha|t2)$",
            r"[\._]diff[\._]weight$",
            r"[\._]diff$",
            r"[\._]weight$",
            r"[\._]bias$",
        ]
        for pattern in lora_suffixes:
            k_lower = re.sub(pattern, "", k_lower)

        k_lower = k_lower.replace("transformer_blocks", "blocks")
        k_lower = k_lower.replace("text_fusion", "txtfusion")

        k_lower = k_lower.replace("ff.down", "mlp.down").replace("ff_down", "mlp.down")
        k_lower = k_lower.replace("ff.up", "mlp.up").replace("ff_up", "mlp.up")
        k_lower = k_lower.replace("ff.gate", "mlp.gate").replace("ff_gate", "mlp.gate")

        k_lower = k_lower.replace("attn.to_q", "attn.wq").replace("attn.to.q", "attn.wq")
        k_lower = k_lower.replace("attn.to_k", "attn.wk").replace("attn.to.k", "attn.wk")
        k_lower = k_lower.replace("attn.to_v", "attn.wv").replace("attn.to.v", "attn.wv")
        k_lower = k_lower.replace("attn.to_out.0", "attn.wo").replace("attn.to.out.0", "attn.wo").replace("attn.to_out", "attn.wo")
        k_lower = k_lower.replace("attn.to_gate", "attn.gate").replace("attn.to.gate", "attn.gate")

        k_lower = k_lower.replace("txtfusion.layerwise.blocks", "txtfusion.layerwise_blocks")
        k_lower = k_lower.replace("txtfusion.refiner.blocks", "txtfusion.refiner_blocks")

        k_lower = k_lower.replace("img_in", "first").replace("img.in", "first")
        k_lower = k_lower.replace("final_layer.linear", "last.linear").replace("final.layer.linear", "last.linear")
        k_lower = k_lower.replace("txt_in.linear_1", "txtmlp.1").replace("txt.in.linear.1", "txtmlp.1")
        k_lower = k_lower.replace("txt_in.linear_2", "txtmlp.3").replace("txt.in.linear.2", "txtmlp.3")
        k_lower = k_lower.replace("time_embed.linear_1", "tmlp.0").replace("time.embed.linear.1", "tmlp.0")
        k_lower = k_lower.replace("time_embed.linear_2", "tmlp.2").replace("time.embed.linear.2", "tmlp.2")

        return k_lower.strip("._")

    @classmethod
    def get_lora_metadata(cls, rank_target: int, alpha_target: int) -> Dict[str, str]:
        """Exact Tensor.Art deployable Krea 2 LoRA metadata from Cell 8 & Cell 9."""
        return {
            "modelspec.architecture": "krea2/lora",
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(rank_target),
            "ss_network_alpha": str(int(alpha_target)),
            "modelspec.sai_model_spec": "1.0.0",
            "ss_base_model_version": "krea2",
        }
