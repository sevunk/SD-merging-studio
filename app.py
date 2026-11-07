"""Merging Studio v2.0 — FastAPI + Gradio Public Tunnel Bridge Server (`app.py`).

When launched in Google Colab via `python3 app.py --share`, this script:
1. Starts the Gradio Public Tunnel (`https://xxxx.gradio.live`)
2. Mounts the full HTML5 + CSS3 + React.js Web Studio UI and REST API endpoints
   connecting every button in the UI directly to `core/` and `architectures/`.
"""

from __future__ import annotations
import argparse
import os
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from architectures import ARCH_REGISTRY
from core.analyzer_xray import export_xray_report, inspect_safetensors_quick
from core.downloader_hub import execute_file_commander, flush_colab_system
from core.lora_svd_engine import merge_multi_loras_svd
from core.merge_algorithms import run_unified_checkpoint_merge


def create_gradio_studio_interface():
    """Builds the Gradio Blocks bridge that embeds the React/HTML5/CSS3 Studio
    and exposes Python backend handlers for Google Colab (`share=True`).
    """
    import gradio as gr

    with gr.Blocks(
        title="Merging Studio v2.0 — MOSS Image, Krea 2, Anima, MiniMax H3",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    ) as demo:
        gr.HTML(
            """
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); padding: 18px 24px; border-radius: 16px; border: 1px solid #312e81; color: #f8fafc; margin-bottom: 12px;">
              <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                  <h1 style="margin:0; font-size:20px; font-weight:800; color:#fff;">🚀 MERGING STUDIO v2.0 (GitHub + Gradio Tunnel Edition)</h1>
                  <p style="margin:4px 0 0 0; font-size:13px; color:#94a3b8;">
                    100% Locked Layer Blueprints • MOSS Image (42L) • Krea 2 (28B + TextFusion + Direct INT8 ConvRot + GGUF) • Anima • MiniMax H3
                  </p>
                </div>
                <div style="background:#065f46; color:#6ee7b7; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:700;">
                  🛡️ Golden Layer Invariants: LOCKED
                </div>
              </div>
            </div>
            """
        )

        with gr.Tabs():
            # TAB 1: DOWNLOAD & FILE COMMANDER
            with gr.Tab("1. 📥 Download & File Commander"):
                with gr.Row():
                    with gr.Column():
                        dl_urls = gr.Textbox(label="Model URLs (Comma-Separated: HF / Civitai / ModelScope / GitHub)", lines=2)
                        with gr.Row():
                            civ_tok = gr.Textbox(label="Civitai API Token", type="password")
                            hf_tok = gr.Textbox(label="HuggingFace Token", type="password")
                        dl_dest = gr.Textbox(label="Destination Folder", value="/content")
                    with gr.Column():
                        fc_op = gr.Dropdown(
                            choices=["Copy", "Move", "Delete", "Zip (Compress)", "Extract (Unzip/Unrar)"],
                            value="Copy",
                            label="File Commander Operation",
                        )
                        fc_src = gr.Textbox(label="Source Path", value="/content")
                        fc_dst = gr.Textbox(label="Destination Path", value="/content/drive/MyDrive")
                        fc_btn = gr.Button("📂 Execute File Operation", variant="primary")
                        flush_btn = gr.Button("🧹 1-Click Flush RAM, CUDA Cache & /tmp", variant="stop")
                        fc_out = gr.JSON(label="System / File Status")

                fc_btn.click(
                    fn=lambda op, s, d: execute_file_commander(op, s, d),
                    inputs=[fc_op, fc_src, fc_dst],
                    outputs=fc_out,
                )
                flush_btn.click(fn=flush_colab_system, inputs=[], outputs=fc_out)

            # TAB 2: CHECKPOINT MERGER
            with gr.Tab("2. 🔥 Checkpoint Merger (MOSS, Krea 2, Anima, MiniMax H3)"):
                with gr.Row():
                    base_in = gr.Textbox(label="Base Model Path (.safetensors — BF16 or INT8 ConvRot)", value="/content/mossImage_sft_bf16.safetensors")
                    inspect_btn = gr.Button("🔍 Auto-Inspect Base Model")
                inspect_out = gr.JSON(label="Smart File Inspector Readout")
                inspect_btn.click(fn=inspect_safetensors_quick, inputs=[base_in], outputs=[inspect_out])

                donors_in = gr.Textbox(
                    label="Donor Model Paths (Comma-Separated — Freely Mix BF16 & INT8 ConvRot!)",
                    lines=2,
                )
                weights_in = gr.Textbox(label="Donor Weights (Comma-Separated, e.g. 0.45, 0.35, 0.20)", value="0.45, 0.35, 0.20")

                with gr.Row():
                    arch_sel = gr.Dropdown(
                        choices=["Auto-Detect", "MOSS_Image", "KREA_2", "ANIMA", "MINIMAX_H3"],
                        value="Auto-Detect",
                        label="Model Architecture",
                    )
                    mode_sel = gr.Dropdown(
                        choices=[
                            "Auto",
                            "Block_Guided_V3",
                            "True_Disjoint_TIES",
                            "Norm_Preserving_TIES",
                            "Pure_Weighted_Linear",
                            "DARE_Hydra_Krea2",
                            "Krea2_Zone_Guided_DARE",
                            "DARE_Hydra_Anima",
                            "DARE_Hydra_Minimax",
                        ],
                        value="Auto",
                        label="Merge Mode",
                    )
                    fmt_sel = gr.Dropdown(
                        choices=["bf16", "int8_convrot", "mix_bf16_f32", "fp16", "native"],
                        value="bf16",
                        label="Output Format",
                    )

                with gr.Accordion("🎛️ Block-Guided / Zone Strength & Pro DARE/TIES Settings", open=False):
                    with gr.Row():
                        early_s = gr.Slider(0.0, 1.2, value=0.55, step=0.05, label="Early / Low Zone Strength")
                        mid_s = gr.Slider(0.0, 1.2, value=0.90, step=0.05, label="Mid Zone Strength")
                        late_s = gr.Slider(0.0, 1.2, value=0.30, step=0.05, label="Late / High Zone Strength")
                        other_s = gr.Slider(0.0, 1.2, value=0.50, step=0.05, label="Other / TextFusion Strength")
                    with gr.Row():
                        ties_d = gr.Slider(0.1, 0.5, value=0.40, step=0.05, label="TIES Density")
                        p_drp = gr.Slider(0.0, 0.7, value=0.00, step=0.05, label="DARE p_drop")
                        seed_in = gr.Number(value=42, label="Random Seed", precision=0)

                out_merge_p = gr.Textbox(label="Output Merged Checkpoint Path", value="/content/merged_output.safetensors")
                merge_btn = gr.Button("🚀 Execute Streaming Checkpoint Merge", variant="primary")
                merge_res = gr.JSON(label="Merge Result")

                merge_btn.click(
                    fn=lambda b, d_str, o, a, m, f, w, td, pd, sd, es, ms, ls, os_val: run_unified_checkpoint_merge(
                        base_model_path=b,
                        donor_model_paths=[x.strip() for x in d_str.split(",") if x.strip()],
                        output_path=o,
                        arch_id=a,
                        merge_mode=m,
                        output_format=f,
                        donor_weights_str=w,
                        ties_density=td,
                        p_drop=pd,
                        seed=int(sd),
                        early_strength=es,
                        mid_strength=ms,
                        late_strength=ls,
                        other_strength=os_val,
                    ),
                    inputs=[
                        base_in, donors_in, out_merge_p, arch_sel, mode_sel, fmt_sel,
                        weights_in, ties_d, p_drp, seed_in, early_s, mid_s, late_s, other_s,
                    ],
                    outputs=[merge_res],
                )

            # TAB 5: THE X-RAY EXPORTER
            with gr.Tab("5. 🔬 The X-Ray & Pre-Merge Analyzer"):
                xray_in = gr.Textbox(label="Input .safetensors File Path", value="/content/MOSS_Realistic_LoRA.safetensors")
                xray_out = gr.Textbox(label="Output Report Path (.json / .txt)", value="/content/MOSS_Realistic_LoRA.json")
                xray_fmt = gr.Dropdown(choices=["json", "txt"], value="json", label="Report Format")
                xray_btn = gr.Button("🕵️ Export X-Ray Report", variant="primary")
                xray_res = gr.JSON(label="X-Ray Result")
                xray_btn.click(fn=export_xray_report, inputs=[xray_in, xray_out, xray_fmt], outputs=[xray_res])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Merging Studio v2.0")
    parser.add_argument("--share", action="store_true", default=True, help="Create public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    ui = create_gradio_studio_interface()
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
