# 🚀 Merging Studio v2.0 (GitHub + Gradio Edition)

Studio *merging*, *LoRA/LoKr SVD*, dan *quantization* (`INT8 ConvRot` & `GGUF`) untuk **MOSS Image**, **Krea 2**, **Anima**, dan **MiniMax H3** dengan **100% Jaminan Keutuhan Struktur Layer Asli (*Zero Layer Drift*)**.

## 🌟 Keunggulan Arsitektur v2.0

1. **100% Invarian Layer Terkunci (`architectures/*.py`)**:
   - **MOSS Image**: Mengunci `rope.bands` `[16]` ke `F32` (64 bytes), seluruh layer lain ke `BF16`, mendukung *Block-Guided 42-Layer (`0–13`, `14–27`, `28–41`)*, konversi dua arah otomatis LoRA `fused_qkv` $\leftrightarrow$ `split_q_k_v`, serta *blueprint* **757-Tensor INT8 ConvRot** (`comfy_quant` `[72]` U8 $\rightarrow$ `weight` I8 $\rightarrow$ `weight_scale` F32 + 42 `comfy_attention.config` `[35]` U8).
   - **Krea 2**: Mengunci *Tensor.Art Mixed Precision* (`first.weight` = `BF16`, `scale/bias/mod.lin/tmlp/tproj/projector/last.linear/txtmlp` = `F32`), kamus translasi alias *Diffusers / ModelScope*, pemulihan penuh `mlp.down`, konverter GPU T4 **GGUF** (`Byteshape_Exact_7.35GiB` & `True_Q4_K_M_HighQuality_8.28GiB`), serta **Direct INT8 ConvRot Input/Output** (lengkap dengan *Hadamard 256 Un-rotation* saat membaca donor INT8 dan *Forward Hadamard Rotation* saat menyimpan ke format eksak **12.56 GB Tensor.Art**).
   - **Anima & MiniMax H3**: Translasi otomatis prefix `net.` $\rightarrow$ `model.diffusion_model.` untuk Anima, serta preservasi *native dtype* (`I8`, `U8`, `BF16`) untuk MiniMax H3.
2. **Cara Menambahkan Model Baru di Masa Depan (Risiko 0%)**:
   - Cukup buat 1 file baru di dalam folder `architectures/` (contoh: `architectures/flux2.py`) yang mewarisi `ArchitectureSpec` dari `architectures/base_spec.py`, lalu daftarkan ke `REGISTERED_ARCHITECTURES` di `architectures/__init__.py`.
   - File `moss_image.py`, `krea2.py`, `anima.py`, dan `minimax_h3.py` **tidak pernah disentuh**, sehingga mustahil merusak model yang sudah berjalan.

## ⚡ Cara Menjalankan di Google Colab (Hanya 1 Sel!)

Buka file `Colab_Launcher.ipynb` di Google Colab, masukkan URL repositori GitHub Anda, lalu klik **Run**. Tautan publik `https://xxxx.gradio.live` akan langsung muncul!

## 🧪 Verifikasi Unit Test Invarian Layer

```bash
python3 tests/test_invariants.py
```
