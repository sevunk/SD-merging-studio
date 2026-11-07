"""Merging Studio v2.0 — Full React.js + HTML5/CSS3 Web Studio Server with Gradio Tunnel (`app.py`).

FIX FOR GRADIO DEFAULT UI:
Instead of showing plain white Gradio input boxes, this server:
1. Mounts all REST API endpoints (`/api/status`, `/api/download`, `/api/file-commander`,
   `/api/inspect`, `/api/merge-checkpoint`, `/api/lora-studio`, `/api/quantize`,
   `/api/xray`, `/api/hf-upload`, `/api/flush`) directly onto Gradio's underlying
   FastAPI instance (`demo.app`).
2. Serves the custom Dark Glassmorphism **HTML5 + CSS3 + React.js 18 Single-Page Application**
   (`frontend/index.html`) at `/studio` and embeds it full-screen (`100vw x 100vh`, zero Gradio
   chrome/padding) inside the `*.gradio.live` root page!
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Any, Dict, List

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from core.analyzer_xray import export_xray_report, inspect_safetensors_quick
from core.downloader_hub import (
    download_batch_and_modelscope,
    execute_file_commander,
    flush_colab_system,
    get_system_resources,
    publish_to_huggingface_hub,
)
from core.gguf_cuda_kernels import run_krea2_to_gguf_converter
from core.lora_svd_engine import (
    bake_loras_to_checkpoint,
    extract_lora_svd_turbo,
    merge_multi_loras_svd,
)
from core.merge_algorithms import run_unified_checkpoint_merge
from core.quantizer_engine import (
    run_universal_int8_convrot_quantizer,
    run_universal_unquantizer_v5,
)


FRONTEND_HTML_PATH = os.path.join(REPO_DIR, "frontend", "index.html")

# CSS to strip away all default white Gradio padding/borders/footers so the React.js SPA fills 100% of the screen
FULLSCREEN_REACT_CSS = """
body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100vw !important;
    width: 100vw !important;
    min-height: 100vh !important;
    background: #020617 !important;
    overflow-x: hidden !important;
}
footer { display: none !important; }
#react-studio-frame {
    width: 100vw;
    height: 100vh;
    border: none;
    display: block;
    background: #020617;
}
"""


def register_studio_api_routes(fastapi_app) -> None:
    """Registers the React SPA at root ('/') and all REST APIs directly onto
    the live server running behind the Gradio tunnel.
    """

    # 1. Sajikan React SPA langsung di Root "/" DAN "/studio"
    @fastapi_app.get("/", response_class=HTMLResponse)
    @fastapi_app.get("/studio", response_class=HTMLResponse)
    async def serve_react_studio():
        if os.path.exists(FRONTEND_HTML_PATH):
            with open(FRONTEND_HTML_PATH, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(
            content=f"<h1>Error: {FRONTEND_HTML_PATH} not found</h1>", 
            status_code=404
        )

    # 2. REST API Endpoints
    @fastapi_app.get("/api/status")
    async def api_status():
        try:
            return JSONResponse(get_system_resources("/content"))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @fastapi_app.post("/api/flush")
    async def api_flush():
        try:
            return JSONResponse(flush_colab_system())
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @fastapi_app.post("/api/inspect")
    async def api_inspect(req: Request):
        try:
            body = await req.json()
            res = inspect_safetensors_quick(body.get("file_path", ""))
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @fastapi_app.post("/api/download")
    async def api_download(req: Request):
        try:
            body = await req.json()
            res = download_batch_and_modelscope(
                urls_csv=body.get("urls_csv", ""),
                destination=body.get("destination", "/content"),
                hf_token=body.get("hf_token", ""),
                civ_token=body.get("civ_token", ""),
                modelscope_repo=body.get("modelscope_repo", ""),
                modelscope_token=body.get("modelscope_token", ""),
            )
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @fastapi_app.post("/api/file-commander")
    async def api_file_commander(req: Request):
        try:
            body = await req.json()
            res = execute_file_commander(
                operation=body.get("operation", "Copy"),
                source_path=body.get("source_path", ""),
                destination_path=body.get("destination_path", ""),
            )
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @fastapi_app.post("/api/merge-checkpoint")
    async def api_merge_checkpoint(req: Request):
        try:
            body = await req.json()
            res = run_unified_checkpoint_merge(
                base_model_path=body.get("base_model_path", ""),
                donor_model_paths=body.get("donor_model_paths", []),
                output_path=body.get("output_path", "/content/merged_output.safetensors"),
                arch_id=body.get("arch_id", "Auto-Detect"),
                merge_mode=body.get("merge_mode", "Auto"),
                output_format=body.get("output_format", "bf16"),
                donor_weights_str=body.get("donor_weights_str", ""),
                ties_density=float(body.get("ties_density", 0.40)),
                p_drop=float(body.get("p_drop", 0.0)),
                seed=int(body.get("seed", 42)),
                early_strength=float(body.get("early_strength", 0.55)),
                mid_strength=float(body.get("mid_strength", 0.90)),
                late_strength=float(body.get("late_strength", 0.30)),
                other_strength=float(body.get("other_strength", 0.50)),
            )
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @fastapi_app.post("/api/lora-studio")
    async def api_lora_studio(req: Request):
        try:
            body = await req.json()
            mode = body.get("mode", "svd_merge")
            if mode == "svd_merge":
                res = merge_multi_loras_svd(
                    loras_cfg=body.get("loras_cfg", []),
                    output_lora_path=body.get("output_path", "/content/merged_lora.safetensors"),
                    arch_id=body.get("arch_id", "MOSS_Image"),
                    target_rank=int(body.get("target_rank", 128)),
                    target_alpha=int(body.get("target_alpha", 128)),
                    qkv_output_mode=body.get("qkv_output_mode", "fused_qkv"),
                )
            elif mode == "baker":
                res = bake_loras_to_checkpoint(
                    base_model_path=body.get("base_model_path", ""),
                    loras_cfg=body.get("loras_cfg", []),
                    output_path=body.get("output_path", "/content/baked_model.safetensors"),
                    krea2_mix_precision_bf16=bool(body.get("krea2_mix_precision_bf16", True)),
                )
            else:
                res = extract_lora_svd_turbo(
                    base_path=body.get("base_model_path", ""),
                    tuned_path=body.get("tuned_model_path", ""),
                    out_path=body.get("output_path", "/content/extracted_lora.safetensors"),
                    rank_val=int(body.get("target_rank", 256)),
                    alpha_val=int(body.get("target_alpha", 256)),
                )
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @fastapi_app.post("/api/quantize")
    async def api_quantize(req: Request):
        try:
            body = await req.json()
            mode = body.get("mode", "int8_convrot")
            if mode == "int8_convrot":
                res = run_universal_int8_convrot_quantizer(
                    input_model_path=body.get("input_path", ""),
                    output_model_path=body.get("output_path", ""),
                    prefix_mode=body.get("prefix_mode", "Auto"),
                    inject_moss_attention_config=bool(body.get("inject_moss_attention_config", True)),
                    restore_moss_rope_bands=bool(body.get("restore_moss_rope_bands", True)),
                )
            elif mode == "unquantize":
                res = run_universal_unquantizer_v5(
                    input_model_path=body.get("input_path", ""),
                    output_model_path=body.get("output_path", ""),
                    prefix_mode=body.get("prefix_mode", "Auto"),
                    restore_moss_rope_bands=bool(body.get("restore_moss_rope_bands", True)),
                )
            else:
                res = run_krea2_to_gguf_converter(
                    input_safetensors_path=body.get("input_path", ""),
                    output_gguf_path=body.get("output_path", ""),
                    quant_profile=body.get("gguf_profile", "True_Q4_K_M_HighQuality_8.28GiB"),
                )
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @fastapi_app.post("/api/xray")
    async def api_xray(req: Request):
        try:
            body = await req.json()
            res = export_xray_report(
                input_p=body.get("input_path", ""),
                out_p=body.get("output_path", ""),
                fmt=body.get("format", "json"),
            )
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @fastapi_app.post("/api/hf-upload")
    async def api_hf_upload(req: Request):
        try:
            body = await req.json()
            res = publish_to_huggingface_hub(
                hf_token=body.get("hf_token", ""),
                model_repo_name=body.get("model_repo_name", ""),
                paths_csv=body.get("paths_csv", ""),
                commit_message=body.get("commit_message", "feat: Upload model from Merging Studio v2.0"),
            )
            return JSONResponse(res)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)


def launch_studio_with_gradio_tunnel(port: int = 7860, share: bool = True):
    import gradio as gr

    # Fallback iframe layar penuh (jika sewaktu-waktu diakses via komponen Gradio)
    with gr.Blocks(title="MERGING STUDIO v2.0 — React.js Web Studio") as demo:
        gr.HTML(
            '<iframe id="react-studio-frame" src="/studio" allow="clipboard-read; clipboard-write"></iframe>'
        )

    print("✨ Launching Merging Studio via Gradio Tunnel...")

    # 1. Jalankan server dengan prevent_thread_lock=True
    app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        css=FULLSCREEN_REACT_CSS,
        prevent_thread_lock=True,
    )

    # 2. HAPUS rute "/" putih bawaan Gradio agar tidak menghalangi React SPA
    app.router.routes = [
        r for r in app.router.routes
        if getattr(r, "path", None) != "/"
    ]

    # 3. Pasang semua REST API (/api/*) serta rute "/" dan "/studio" milik kita
    register_studio_api_routes(app)

    # 4. Pindahkan rute React SPA ("/" dan "/studio") ke urutan paling atas (Index 0)
    spa_routes = [
        r for r in app.router.routes
        if getattr(r, "name", "") == "serve_react_studio"
    ]
    other_routes = [
        r for r in app.router.routes
        if getattr(r, "name", "") != "serve_react_studio"
    ]
    app.router.routes = spa_routes + other_routes

    print(f"\n🚀 SUCCESS! Merging Studio running at:")
    print(f"   Local URL:  {local_url}")
    if share_url:
        print(f"   Public URL: {share_url}")
        print(f"   Studio URL: {share_url.rstrip('/')}/studio\n")
    print("💡 Tekan tombol STOP di Colab untuk mematikan server.\n")

    # 5. Kunci server agar tetap menyala
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Mematikan server Merging Studio...")
        demo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Merging Studio v2.0 Full React SPA")
    parser.add_argument("--share", action="store_true", default=True, help="Create public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    launch_studio_with_gradio_tunnel(port=args.port, share=args.share)
