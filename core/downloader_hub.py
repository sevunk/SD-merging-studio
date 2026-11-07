"""Downloader, File & Archive Commander, System Flush, & HuggingFace Hub Engine.

Preserves 100% of the functionality from:
- Cell 1: Dependencies & Google Drive Mount
- Cell 2: Download Models & Repositories (Aria2c Turbo -> Curl -> Python Stream fallback, Civitai/HF/ModelScope)
- Cell 11B & 12: Configure, Login, & Multi-File Upload to Hugging Face Hub
- Cell 17: File & Archive Commander (Copy/Move with 16MB chunk progress, Delete, Zip, Extract)
- Cell 18: FLUSH! (RAM gc, CUDA cache, /tmp, Trash, pip cache)
"""

from __future__ import annotations
import gc
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse
import zipfile

# Matikan progress bar interaktif tqdm & HuggingFace Hub secara global
# agar tidak mencetak ribuan baris "Processing Files" yang membuat Colab hang
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"


def get_filename_from_url(url: str) -> Optional[str]:
    """Extracts the original filename from a URL, ignoring query parameters (Cell 2)."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    filename = os.path.basename(path)
    valid_extensions = (
        ".safetensors", ".ckpt", ".bin", ".pth", ".pt", ".gguf", ".onnx",
        ".zip", ".rar", ".7z", ".tar", ".gz", ".json", ".yaml", ".yml", ".txt",
    )
    if filename.lower().endswith(valid_extensions):
        return filename
    return None


def resolve_url_and_filename(url: str, headers: Optional[Dict[str, str]] = None) -> tuple[str, Optional[str]]:
    """Resolves redirect URLs (especially Civitai) and extracts the correct filename (Cell 2)."""
    import requests
    headers = headers or {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        # Gunakan context manager 'with' agar koneksi stream langsung ditutup setelah membaca header
        with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=15) as response:
            final_url = response.url
            filename = None

            cd = response.headers.get("content-disposition")
            if cd:
                filenames = re.findall(r'filename=["\']?([^"\']+)["\']?', cd)
                if filenames:
                    filename = filenames[0]
                else:
                    filenames = re.findall(r"filename\*=UTF-8''(.+)", cd)
                    if filenames:
                        filename = unquote(filenames[0])

            if not filename:
                parsed_url = urlparse(final_url)
                query_params = parse_qs(parsed_url.query)
                for key in ["b2ContentDisposition", "response-content-disposition", "content-disposition"]:
                    if key in query_params:
                        val = query_params[key][0]
                        filenames = re.findall(r'filename=["\']?([^"\']+)["\']?', val)
                        if filenames:
                            filename = filenames[0]
                            break

            if not filename:
                parsed_url = urlparse(final_url)
                filename = os.path.basename(unquote(parsed_url.path))

            return final_url, filename
    except Exception:
        return url, None


def download_single_url(
    url: str,
    destination: str = "/content",
    hf_token: str = "",
    civ_token: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Executes the exact 3-tier download engine from Cell 2 (Aria2c -> Curl -> Python Stream) with clean logs."""
    def log(msg: str):
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    url = url.strip()
    if not url:
        return {"status": "skipped", "reason": "empty_url"}

    os.makedirs(destination, exist_ok=True)
    clean_hf_token = hf_token.strip() if hf_token and hf_token.strip() else None

    if "civitai.red" in url:
        url = url.replace("civitai.red", "civitai.com")
        log("🔄 Detected 'civitai.red', automatically replaced with 'civitai.com'")

    # 1. Handle GitHub (Quiet mode agar tidak spam log)
    if "github.com" in url:
        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
        target = os.path.join(destination, repo_name)
        if not os.path.exists(target):
            log(f"🐙 Cloning GitHub repo: {repo_name}...")
            subprocess.run(["git", "clone", "-q", "--depth", "1", url, target], check=False)
            log(f"✅ Finished cloning {repo_name}")
        return {"status": "cloned", "target": target}

    # 2. Handle HF Repo menggunakan API resmi snapshot_download (Tanpa spam progress bar)
    if "huggingface.co" in url and all(x not in url for x in ["/resolve/", "/blob/"]):
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        repo_id = "/".join(url.split("huggingface.co/")[1].split("/")[:2])
        t0 = time.time()
        log(f"🤗 Downloading HF Repo: {repo_id} (mohon tunggu, log progress bar diringkas)...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            token=clean_hf_token,
        )
        elapsed = round(time.time() - t0, 1)
        log(f"✅ HF Repo '{repo_id}' downloaded in {elapsed}s!")
        return {"status": "downloaded_repo", "repo_id": repo_id, "elapsed_sec": elapsed}

    # 3. Direct File Download (Civitai / HF File)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    if "civitai.com" in url and civ_token and "token=" not in url:
        url = url + ("&" if "?" in url else "?") + f"token={civ_token.strip()}"
        log("🔑 Civitai Token applied to URL.")
    elif "huggingface.co" in url and clean_hf_token:
        headers["Authorization"] = f"Bearer {clean_hf_token}"

    log(f"🔍 Resolving redirect link for: {url[:80]}...")
    final_url, filename = resolve_url_and_filename(url, headers)
    if not filename:
        filename = get_filename_from_url(url) or "downloaded_model.safetensors"

    output_path = os.path.join(destination, filename)
    if os.path.exists(output_path):
        if os.path.getsize(output_path) < 1024 * 1024:
            log(f"⚠️ Existing file '{filename}' is <1MB. Deleting and re-downloading...")
            os.remove(output_path)
        else:
            sz_gb = os.path.getsize(output_path) / (1024 ** 3)
            log(f"✅ File '{filename}' already exists ({sz_gb:.2f} GiB). Skipping.")
            return {"status": "exists", "file_path": output_path, "filename": filename}

    t0 = time.time()
    log(f"🚀 Downloading '{filename}' via Aria2c Turbo (silent mode)...")
    aria_cmd = [
        "aria2c",
        "--console-log-level=error",
        "--summary-interval=0",
        "--download-result=hide",
        "--quiet=true",
        "-x", "8", "-s", "8", "-k", "1M",
        f"--user-agent={headers['User-Agent']}",
        "-d", destination,
        "-o", filename,
    ]
    if "civitai.com" not in url:
        aria_cmd.append("-c")
    for k, v in headers.items():
        aria_cmd.extend(["--header", f"{k}: {v}"])
    aria_cmd.append(final_url)

    result = subprocess.run(aria_cmd)
    if result.returncode == 0 and os.path.exists(output_path):
        sz_gb = os.path.getsize(output_path) / (1024 ** 3)
        elapsed = round(time.time() - t0, 1)
        log(f"✅ Downloaded '{filename}' ({sz_gb:.2f} GiB) in {elapsed}s via Aria2c Turbo!")
        return {"status": "downloaded", "method": "aria2c", "file_path": output_path, "filename": filename}

    log("⚠️ Aria2c failed. Trying fallback (Curl)...")
    for ext in [".aria2", ".part"]:
        if os.path.exists(output_path + ext):
            os.remove(output_path + ext)
    if os.path.exists(output_path):
        os.remove(output_path)

    curl_cmd = ["curl", "-f", "-s", "-S", "-g", "-L", "--user-agent", headers["User-Agent"], "-o", output_path]
    if "civitai.com" not in url:
        curl_cmd.extend(["-C", "-"])
    for k, v in headers.items():
        curl_cmd.extend(["-H", f"{k}: {v}"])
    curl_cmd.append(final_url)
    res_curl = subprocess.run(curl_cmd)
    if res_curl.returncode == 0 and os.path.exists(output_path):
        sz_gb = os.path.getsize(output_path) / (1024 ** 3)
        elapsed = round(time.time() - t0, 1)
        log(f"✅ Downloaded '{filename}' ({sz_gb:.2f} GiB) in {elapsed}s via Curl!")
        return {"status": "downloaded", "method": "curl", "file_path": output_path, "filename": filename}

    log("⚠️ Curl failed. Trying ultimate fallback (Python Stream)...")
    import requests
    with requests.get(final_url, headers=headers, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for data in response.iter_content(16 * 1024 * 1024):
                if data:
                    f.write(data)
    sz_gb = os.path.getsize(output_path) / (1024 ** 3)
    elapsed = round(time.time() - t0, 1)
    log(f"✅ Downloaded '{filename}' ({sz_gb:.2f} GiB) in {elapsed}s via Python Stream!")
    return {"status": "downloaded", "method": "python_stream", "file_path": output_path, "filename": filename}


def download_batch_and_modelscope(
    urls_csv: str,
    destination: str = "/content",
    hf_token: str = "",
    civ_token: str = "",
    modelscope_repo: str = "",
    modelscope_token: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Downloads comma-separated URLs and/or clones a ModelScope repository (Cell 2)."""
    results = []
    if urls_csv and urls_csv.strip():
        for u in [x.strip() for x in urls_csv.split(",") if x.strip()]:
            results.append(download_single_url(u, destination, hf_token, civ_token, log_fn))

    if modelscope_repo and modelscope_repo.strip():
        repo_id = modelscope_repo.strip()
        msg = f"⏳ Cloning ModelScope repository: {repo_id}..."
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)
        subprocess.run("git lfs install > /dev/null 2>&1", shell=True)
        git_url = (
            f"https://oauth2:{modelscope_token.strip()}@www.modelscope.ai/{repo_id}.git"
            if modelscope_token and modelscope_token.strip()
            else f"https://www.modelscope.ai/{repo_id}.git"
        )
        subprocess.run(["git", "clone", "-q", git_url], cwd=destination)
        results.append({"status": "modelscope_cloned", "repo": repo_id})

    return {"results": results, "destination": destination}


def publish_to_huggingface_hub(
    hf_token: str,
    model_repo_name: str,
    paths_csv: str,
    path_in_repo: str = "",
    commit_message: str = "feat: Upload model from Merging Studio v2.0",
    organization_name: str = "",
    make_private: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Authenticates with HfApi(token=...), creates the repo if needed, and uploads files/folders cleanly."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars, validate_repo_id

    # Pastikan progress bar "Processing Files" dimatikan agar tidak membuat Colab hang
    disable_progress_bars()

    def log(msg: str):
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    clean_token = hf_token.strip() if hf_token else ""
    if not clean_token or not clean_token.startswith("hf_"):
        raise ValueError("Invalid Hugging Face token (must start with 'hf_').")

    log("🔑 Authenticating with Hugging Face Hub...")
    api = HfApi(token=clean_token)
    user = api.whoami()
    username = user["name"]
    repo_owner = organization_name.strip() if organization_name and organization_name.strip() else username
    repo_id = f"{repo_owner}/{model_repo_name.strip().strip('/')}"

    validate_repo_id(repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model", private=make_private, exist_ok=True)
    log(f"✅ Model repository ready: https://huggingface.co/{repo_id}")

    paths_list = [p.strip() for p in paths_csv.split(",") if p.strip()]
    uploaded_urls = []

    for item_path in paths_list:
        local_path = Path(item_path)
        if not local_path.exists():
            log(f"⚠️ Skipping missing path: {item_path}")
            continue

        is_folder = local_path.is_dir()
        target_repo_path = path_in_repo.strip().strip("/")

        if is_folder:
            # Jika meng-upload 1 folder dan path_in_repo kosong, upload ke root repo ("")
            if not target_repo_path and len(paths_list) > 1:
                target_repo_path = local_path.name
        else:
            if not target_repo_path:
                target_repo_path = local_path.name
            elif len(paths_list) > 1 or path_in_repo.strip().endswith("/"):
                target_repo_path = f"{target_repo_path}/{local_path.name}"

        t0 = time.time()
        sz_info = (
            f" ({local_path.stat().st_size / (1024**3):.2f} GiB)"
            if not is_folder
            else " (folder)"
        )
        log(f"🚀 Uploading '{local_path.name}'{sz_info} -> {repo_id}/{target_repo_path or '(root)'}...")

        if is_folder:
            api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=target_repo_path or None,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=[".ipynb_checkpoints", "__pycache__", "*.tmp"],
            )
            url = (
                f"https://huggingface.co/{repo_id}/tree/main/{target_repo_path}"
                if target_repo_path
                else f"https://huggingface.co/{repo_id}/tree/main"
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=target_repo_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            url = f"https://huggingface.co/{repo_id}/blob/main/{target_repo_path}"

        elapsed = round(time.time() - t0, 1)
        uploaded_urls.append(url)
        log(f"✅ Uploaded '{local_path.name}' in {elapsed}s -> {url}")

    return {"repo_id": repo_id, "uploaded_urls": uploaded_urls}


def get_system_resources(scan_dir: str = "/content") -> Dict[str, Any]:
    """Returns real-time GPU VRAM, System RAM, Disk usage, and list of local .safetensors/.gguf files."""
    total_d, used_d, free_d = shutil.disk_usage("/")
    gpu_name = "CPU Mode"
    vram_used_gb = 0.0
    vram_total_gb = 0.0

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_used_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
            vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    except Exception:
        pass

    local_models = []
    if os.path.exists(scan_dir):
        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("sample_data", "Merging-Studio-v2")]
            for f in sorted(files):
                if f.endswith((".safetensors", ".gguf", ".ckpt", ".pt")):
                    fp = os.path.join(root, f)
                    try:
                        sz_gb = round(os.path.getsize(fp) / (1024 ** 3), 3)
                        local_models.append({"name": f, "path": fp, "size_gb": sz_gb})
                    except Exception:
                        pass

    return {
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "disk_used_gb": round(used_d / (1024 ** 3), 2),
        "disk_total_gb": round(total_d / (1024 ** 3), 2),
        "disk_free_gb": round(free_d / (1024 ** 3), 2),
        "local_models": local_models,
    }


def flush_colab_system() -> Dict[str, Any]:
    """Executes the exact cleanup steps from Cell 18 (`FLUSH!`)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    for trash_dir in ["/root/.local/share/Trash", "/tmp"]:
        if os.path.exists(trash_dir):
            for item in os.listdir(trash_dir):
                p = os.path.join(trash_dir, item)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.remove(p)
                except Exception:
                    pass

    subprocess.run("pip cache purge -q > /dev/null 2>&1", shell=True)
    subprocess.run("sync", shell=True)
    return get_system_resources()


def execute_file_commander(
    operation: str,
    source_path: str,
    destination_path: str = "",
) -> Dict[str, Any]:
    """Executes Copy (16MB chunked), Move, Delete, Zip, or Extract operations (Cell 17)."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path not found: {source_path}")

    if operation == "Delete":
        if os.path.isdir(source_path):
            shutil.rmtree(source_path)
        else:
            os.remove(source_path)
        return {"operation": "Delete", "target": source_path, "status": "deleted"}

    if operation in ("Copy", "Move"):
        if not destination_path:
            raise ValueError("Destination path is required for Copy/Move.")
        os.makedirs(destination_path, exist_ok=True)
        dest_full = os.path.join(destination_path, os.path.basename(source_path))
        if os.path.isdir(source_path):
            if os.path.exists(dest_full):
                shutil.rmtree(dest_full)
            shutil.copytree(source_path, dest_full)
            if operation == "Move":
                shutil.rmtree(source_path)
        else:
            chunk_size = 16 * 1024 * 1024
            with open(source_path, "rb") as fsrc, open(dest_full, "wb") as fdst:
                while True:
                    buf = fsrc.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
            if operation == "Move" and os.path.getsize(source_path) == os.path.getsize(dest_full):
                os.remove(source_path)
        return {"operation": operation, "destination": dest_full, "status": "completed"}

    if operation == "Zip (Compress)":
        if not destination_path.endswith(".zip"):
            os.makedirs(destination_path or ".", exist_ok=True)
            zip_path = os.path.join(destination_path, f"{os.path.basename(source_path)}.zip")
        else:
            os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
            zip_path = destination_path

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if os.path.isdir(source_path):
                for root, _, files in os.walk(source_path):
                    for file in files:
                        abs_p = os.path.join(root, file)
                        zf.write(abs_p, os.path.relpath(abs_p, source_path))
            else:
                zf.write(source_path, os.path.basename(source_path))
        return {"operation": "Zip", "zip_path": zip_path, "status": "completed"}

    if operation == "Extract (Unzip/Unrar)":
        dest = destination_path or os.path.splitext(source_path)[0]
        os.makedirs(dest, exist_ok=True)
        if source_path.endswith(".zip"):
            with zipfile.ZipFile(source_path, "r") as zf:
                zf.extractall(dest)
            return {"operation": "Extract", "destination": dest, "status": "completed"}
        elif source_path.endswith(".rar"):
            subprocess.run(["pip", "install", "-q", "rarfile"])
            import rarfile
            with rarfile.RarFile(source_path) as rf:
                rf.extractall(dest)
            return {"operation": "Extract", "destination": dest, "status": "completed"}
        raise ValueError("Unsupported archive format (only .zip and .rar supported).")

    raise ValueError(f"Unknown operation: {operation}")
