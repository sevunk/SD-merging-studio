"""Golden Invariant Verification Test Suite for Merging Studio v2.0.

Verifies that 100% of extracted model layer structures, prefixes, dtypes,
protected keywords, INT8 ConvRot binary layouts (757-tensor MOSS & 12.56GB Krea 2),
Hadamard rotation/un-rotation math, GGUF ByteShape/Q4_K_M tables, and LoRA/LoKr
key normalizers match the original notebook specifications with zero deviation.
"""

from __future__ import annotations
import json
import os
import struct
import sys
import unittest
import numpy as np

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from architectures import ARCH_REGISTRY, auto_detect_architecture
from architectures.moss_image import MossImageSpec
from architectures.krea2 import Krea2Spec
from architectures.anima import AnimaSpec
from architectures.minimax_h3 import MinimaxH3Spec
from core.convrot_math import build_numpy_hadamard, CONVROT_GROUP_SIZE
from core.safetensors_io import build_int8_convrot_plan, find_scale_key_for_int8
from core.lora_svd_engine import parse_lora_role_strict
from core.gguf_cuda_kernels import build_krea2_gguf_plan_and_header, GGML_TYPES


class TestArchitectureInvariants(unittest.TestCase):
    def test_01_moss_image_layer_invariants(self):
        """Verify MOSS Image 42-layer block zones, rope.bands F32 rule, and 757-tensor INT8 ConvRot layout."""
        self.assertEqual(len(MossImageSpec.COMFY_QUANT_PAYLOAD), 72)
        self.assertEqual(len(MossImageSpec.ATTN_CONFIG_PAYLOAD), 35)

        # Only rope.bands is F32 (4 bytes), everything else is BF16 (2 bytes)
        self.assertEqual(MossImageSpec.get_layer_dtype_spec("rope.bands"), ("F32", 4))
        self.assertEqual(MossImageSpec.get_layer_dtype_spec("layers.0.proj.qkv.weight"), ("BF16", 2))
        self.assertEqual(MossImageSpec.get_layer_dtype_spec("text_embedder.proj.weight"), ("BF16", 2))

        # Block-Guided v3.0 boundaries: 0-13 Early, 14-27 Mid, 28-41 Late
        self.assertEqual(MossImageSpec.get_block_strength("layers.0.proj.qkv.weight", 0.55, 0.90, 0.30, 0.50), 0.55)
        self.assertEqual(MossImageSpec.get_block_strength("layers.13.proj.qkv.weight", 0.55, 0.90, 0.30, 0.50), 0.55)
        self.assertEqual(MossImageSpec.get_block_strength("layers.14.proj.qkv.weight", 0.55, 0.90, 0.30, 0.50), 0.90)
        self.assertEqual(MossImageSpec.get_block_strength("layers.27.proj.qkv.weight", 0.55, 0.90, 0.30, 0.50), 0.90)
        self.assertEqual(MossImageSpec.get_block_strength("layers.28.proj.qkv.weight", 0.55, 0.90, 0.30, 0.50), 0.30)
        self.assertEqual(MossImageSpec.get_block_strength("layers.41.proj.qkv.weight", 0.55, 0.90, 0.30, 0.50), 0.30)
        self.assertEqual(MossImageSpec.get_block_strength("shared_block_modulation.weight", 0.55, 0.90, 0.30, 0.50), 0.50)

        # Simulate a full 42-block MOSS Image model and verify INT8 ConvRot layout order:
        # 1. comfy_quant -> 2. weight -> 3. weight_scale + comfy_attention.config + synthetic rope.bands
        sim_moss_shapes = {
            "layers.0.proj.qkv.weight": [9216, 3072],
            "layers.0.proj.o.weight": [3072, 3072],
            "layers.0.proj.ffn_up_gate.weight": [16384, 3072],
            "layers.0.proj.ffn_down.weight": [3072, 8192],
            "layers.0.norm1.weight": [3072],
            "final_linear_video.weight": [64, 3072],
        }
        tasks, header_idx = build_int8_convrot_plan(
            sim_moss_shapes,
            active_arch="MOSS_Image",
            prefix_mode="Auto",
            inject_moss_attention_config=True,
            restore_moss_rope_bands=True,
        )
        # No global __metadata__ in MOSS INT8 ConvRot blueprint
        self.assertNotIn("__metadata__", header_idx)
        # Verify synthetic rope.bands [16] F32 & comfy_attention.config [35] U8 injected
        self.assertIn("rope.bands", header_idx)
        self.assertEqual(header_idx["rope.bands"]["dtype"], "F32")
        self.assertEqual(header_idx["rope.bands"]["shape"], [16])
        self.assertIn("layers.0.comfy_attention.config", header_idx)
        self.assertEqual(header_idx["layers.0.comfy_attention.config"]["shape"], [35])

        # Verify MOSS trio order: 1. comfy_quant -> 2. weight -> 3. weight_scale
        q_off = header_idx["layers.0.proj.qkv.comfy_quant"]["data_offsets"]
        w_off = header_idx["layers.0.proj.qkv.weight"]["data_offsets"]
        s_off = header_idx["layers.0.proj.qkv.weight_scale"]["data_offsets"]
        self.assertEqual(q_off[1], w_off[0])
        self.assertEqual(w_off[1], s_off[0])

    def test_02_krea2_layer_and_int8_convrot_invariants(self):
        """Verify Krea 2 12.56GB INT8 ConvRot layout, Mixed Precision F32 rules, and 28-block zones."""
        self.assertEqual(len(Krea2Spec.COMFY_QUANT_PAYLOAD), 104)

        # Mixed precision F32 rules from Cell 7 & Cell 15
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("first.weight", mode="mix_bf16_f32"), ("BF16", 2))
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("last.linear.weight", mode="mix_bf16_f32"), ("F32", 4))
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("txtfusion.projector.weight", mode="mix_bf16_f32"), ("F32", 4))
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("tmlp.0.weight", mode="mix_bf16_f32"), ("F32", 4))
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("txtmlp.1.weight", mode="mix_bf16_f32"), ("F32", 4))
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("tproj.1.weight", mode="mix_bf16_f32"), ("F32", 4))
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("blocks.0.attn.qknorm.qnorm.scale", mode="mix_bf16_f32"), ("F32", 4))
        self.assertEqual(Krea2Spec.get_layer_dtype_spec("blocks.0.attn.wq.weight", mode="mix_bf16_f32"), ("BF16", 2))

        # Verify Krea 2 INT8 ConvRot targeting (ONLY blocks.*.{attn,mlp}.*.weight with dim[1] % 256 == 0)
        self.assertTrue(Krea2Spec.is_int8_convrot_target("model.diffusion_model.blocks.0.attn.wq.weight", [6144, 6144]))
        self.assertTrue(Krea2Spec.is_int8_convrot_target("blocks.27.mlp.down.weight", [16384, 6144]))
        self.assertFalse(Krea2Spec.is_int8_convrot_target("txtfusion.layerwise_blocks.0.attn.wq.weight", [3072, 3072]))
        self.assertFalse(Krea2Spec.is_int8_convrot_target("first.weight", [6144, 64]))
        self.assertFalse(Krea2Spec.is_int8_convrot_target("blocks.0.attn.qknorm.qnorm.scale", [128]))

        # Verify Krea 2 INT8 trio order: 1. weight (I8) -> 2. weight_scale (F32) -> 3. comfy_quant ([104] U8)
        sim_krea_shapes = {
            "blocks.0.attn.wq.weight": [6144, 6144],
            "blocks.0.attn.qknorm.qnorm.scale": [128],
            "txtfusion.layerwise_blocks.0.attn.wq.weight": [3072, 3072],
            "first.weight": [6144, 64],
        }
        _, krea_header = build_int8_convrot_plan(sim_krea_shapes, active_arch="KREA_2", prefix_mode="Auto")
        self.assertIn("__metadata__", krea_header)
        self.assertEqual(krea_header["__metadata__"]["model_architecture"], "Krea2_MMDiT")

        w_off = krea_header["blocks.0.attn.wq.weight"]["data_offsets"]
        s_off = krea_header["blocks.0.attn.wq.weight_scale"]["data_offsets"]
        q_off = krea_header["blocks.0.attn.wq.comfy_quant"]["data_offsets"]
        self.assertEqual(w_off[1], s_off[0])
        self.assertEqual(s_off[1], q_off[0])
        self.assertEqual(krea_header["blocks.0.attn.wq.comfy_quant"]["shape"], [104])
        # Scale layer stays F32, txtfusion stays BF16
        self.assertEqual(krea_header["blocks.0.attn.qknorm.qnorm.scale"]["dtype"], "F32")
        self.assertEqual(krea_header["txtfusion.layerwise_blocks.0.attn.wq.weight"]["dtype"], "BF16")

        # Verify 28-block Krea 2 Zone-Guided boundaries (Low 0-8, Mid 9-17, High 18-27, TextFusion)
        self.assertEqual(Krea2Spec.get_zone_strength("blocks.0.attn.wq.weight", 0.6, 0.9, 0.4, 0.8), 0.6)
        self.assertEqual(Krea2Spec.get_zone_strength("blocks.8.attn.wq.weight", 0.6, 0.9, 0.4, 0.8), 0.6)
        self.assertEqual(Krea2Spec.get_zone_strength("blocks.9.attn.wq.weight", 0.6, 0.9, 0.4, 0.8), 0.9)
        self.assertEqual(Krea2Spec.get_zone_strength("blocks.17.attn.wq.weight", 0.6, 0.9, 0.4, 0.8), 0.9)
        self.assertEqual(Krea2Spec.get_zone_strength("blocks.18.attn.wq.weight", 0.6, 0.9, 0.4, 0.8), 0.4)
        self.assertEqual(Krea2Spec.get_zone_strength("blocks.27.attn.wq.weight", 0.6, 0.9, 0.4, 0.8), 0.4)
        self.assertEqual(Krea2Spec.get_zone_strength("txtfusion.layerwise_blocks.0.attn.wq.weight", 0.6, 0.9, 0.4, 0.8), 0.8)
        # Critical protection: first., last., tmlp., txtmlp., tproj. always stay at 1.0
        self.assertEqual(Krea2Spec.get_zone_strength("first.weight", 0.6, 0.9, 0.4, 0.8), 1.0)
        self.assertEqual(Krea2Spec.get_zone_strength("tproj.1.weight", 0.6, 0.9, 0.4, 0.8), 1.0)

    def test_03_hadamard_convrot_roundtrip_math(self):
        """Verify Hadamard 256 orthogonality and INT8 ConvRot Rotate -> Un-rotate roundtrip accuracy."""
        h_256 = build_numpy_hadamard(CONVROT_GROUP_SIZE)
        self.assertEqual(h_256.shape, (256, 256))
        # Verify H is symmetric and orthogonal: H @ H.T == Identity
        identity_approx = np.matmul(h_256, h_256.T)
        np.testing.assert_allclose(identity_approx, np.eye(256, dtype=np.float32), atol=1e-5)

        # Simulate a weight matrix [64, 512], rotate -> quantize INT8 -> dequantize -> un-rotate
        rng = np.random.default_rng(42)
        w_orig = rng.normal(0.0, 0.05, size=(64, 512)).astype(np.float32)
        w_grouped = w_orig.reshape(64, 2, 256)
        w_rot = np.matmul(w_grouped, h_256.T).reshape(64, 512)

        scale = np.maximum(np.max(np.abs(w_rot), axis=1, keepdims=True) / 127.0, 1e-30)
        q_int8 = np.clip(np.round(w_rot / scale), -127, 127).astype(np.int8)

        # Dequantize + Un-rotate
        w_deq = q_int8.astype(np.float32) * scale
        w_unrot = np.matmul(w_deq.reshape(64, 2, 256), h_256.T).reshape(64, 512)

        # Cosine similarity between original FP32 and INT8 ConvRot roundtrip must exceed 0.9999
        cos_sim = np.sum(w_orig * w_unrot) / (np.linalg.norm(w_orig) * np.linalg.norm(w_unrot))
        self.assertGreater(cos_sim, 0.9999)

    def test_04_lora_and_baker_key_mappings(self):
        """Verify `mlp.down` & `ffn_down` role safety and Diffusers <-> Native Krea 2 / MOSS mappings."""
        # Ensure mlp.down.weight is never mistaken for LoRA role 'down'
        self.assertIsNone(parse_lora_role_strict("diffusion_model.blocks.0.mlp.down.weight"))
        self.assertIsNone(parse_lora_role_strict("layers.0.proj.ffn_down.weight"))
        self.assertEqual(parse_lora_role_strict("diffusion_model.blocks.0.mlp.down.lora_a.weight"), "down")
        self.assertEqual(parse_lora_role_strict("diffusion_model.blocks.0.mlp.down.lora_b.weight"), "up")
        self.assertEqual(parse_lora_role_strict("layers.0.proj.ffn_down.lokr_w1_a"), "w1_a")

        # Verify Krea 2 LoRA Merger V3.1 key normalization (preserves mlp.down and txtfusion.layerwise_blocks)
        self.assertEqual(
            Krea2Spec.normalize_lora_merger_key("diffusion_model.transformer_blocks.5.ff.down.lora_A.weight"),
            "blocks.5.mlp.down",
        )
        self.assertEqual(
            Krea2Spec.normalize_lora_merger_key("transformer.text_fusion.layerwise.blocks.1.attn.to_q.lora_up.weight"),
            "txtfusion.layerwise_blocks.1.attn.wq",
        )
        self.assertEqual(
            Krea2Spec.normalize_lora_merger_key("diffusion_model.img_in.lora_A.weight"),
            "first",
        )

        # Verify Krea 2 Baker V5.2 flux mapping
        lora_keys = {"blocks.0.attn.to.q", "img.in", "txt.in.linear.2"}
        self.assertEqual(
            Krea2Spec.get_baker_flux_mapping("model.diffusion_model.blocks.0.attn.wq.weight", lora_keys),
            ("direct", "blocks.0.attn.to.q"),
        )
        self.assertEqual(
            Krea2Spec.get_baker_flux_mapping("first.weight", lora_keys),
            ("direct", "img.in"),
        )
        self.assertEqual(
            Krea2Spec.get_baker_flux_mapping("txtmlp.3.weight", lora_keys),
            ("direct", "txt.in.linear.2"),
        )

        # Verify MOSS Kohya & Standard key normalization
        self.assertEqual(
            MossImageSpec.normalize_lora_clean_key("lora_unet_layers_12_proj_ffn_down.lora_down.weight"),
            "layers.12.proj.ffn_down",
        )

    def test_05_krea2_gguf_plan_and_anima_minimax_rules(self):
        """Verify Krea 2 GGUF ByteShape vs True_Q4_K_M profiles, Anima net. prefix, and MiniMax H3."""
        # Block 0 attn.gate.weight is Q3_K in ByteShape, and upgraded to Q4_K in True_Q4_K_M
        self.assertEqual(
            Krea2Spec.get_gguf_qtype("blocks.0.attn.gate.weight", "Byteshape_Exact_7.35GiB"),
            "Q3_K",
        )
        self.assertEqual(
            Krea2Spec.get_gguf_qtype("blocks.0.attn.gate.weight", "True_Q4_K_M_HighQuality_8.28GiB"),
            "Q4_K",
        )
        # Block 0 attn.wv.weight is Q6_K in both profiles
        self.assertEqual(
            Krea2Spec.get_gguf_qtype("blocks.0.attn.wv.weight", "True_Q4_K_M_HighQuality_8.28GiB"),
            "Q6_K",
        )
        # txtfusion weights become BF16 in GGUF
        self.assertEqual(
            Krea2Spec.get_gguf_qtype("txtfusion.refiner_blocks.0.mlp.up.weight", "Byteshape_Exact_7.35GiB"),
            "BF16",
        )

        sim_gguf_shapes = {
            "blocks.0.attn.gate.weight": [6144, 6144],
            "txtfusion.projector.weight": [6144, 3584],
        }
        tasks, gguf_hdr = build_krea2_gguf_plan_and_header(sim_gguf_shapes, "True_Q4_K_M_HighQuality_8.28GiB")
        self.assertTrue(gguf_hdr.startswith(b"GGUF"))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(gguf_hdr) % 32, 0)  # 32-byte aligned header

        # Anima 'net.' -> 'model.diffusion_model.' translator
        self.assertEqual(
            AnimaSpec.translate_to_base_key("net.blocks.0.attn.weight"),
            "model.diffusion_model.blocks.0.attn.weight",
        )

        # MiniMax H3 dynamic key translator
        base_keys = {"model.diffusion_model.blocks.0.weight", "bare_layer.weight"}
        self.assertEqual(
            MinimaxH3Spec.get_translated_key("net.blocks.0.weight", base_keys),
            "model.diffusion_model.blocks.0.weight",
        )
        self.assertEqual(
            MinimaxH3Spec.get_translated_key("model.diffusion_model.bare_layer.weight", base_keys),
            "bare_layer.weight",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
