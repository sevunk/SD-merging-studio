import React, { useState, useMemo } from "react";
import {
  Layers,
  Download,
  Cpu,
  Sliders,
  Zap,
  Search,
  UploadCloud,
  Trash2,
  FolderOpen,
  ShieldCheck,
  Sparkles,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Play,
  Plus,
  RefreshCw,
  HardDrive,
  FileCode,
  Info,
  Globe,
  Terminal,
  Eye,
  Lock,
  ArrowRight,
  Settings,
  Database
} from "lucide-react";

// ============================================================================
// 🗺️ ARCHITECTURE BLUEPRINTS & PRESETS (100% LOCKED TO EXTRACTED INVARIANTS)
// ============================================================================
const ARCH_BLUEPRINTS = {
  MOSS_Image: {
    id: "MOSS_Image",
    name: "MOSS Image (42-Layer DiT)",
    badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
    blocksCount: 42,
    keyPrefixRule: "Bare Keys (Tanpa Prefix) | LoRA: diffusion_model.layers.{0..41}.proj.*",
    dtypeRule: "BF16 Global + F32 Locked pada rope.bands [16] (64 Bytes)",
    int8Spec: "757 Tensors Exact: 168 INT8 ConvRot (FP32 Hadamard) + 42 comfy_attention.config [35] U8 + comfy_quant [72] U8",
    protectedKeywords: ["norm", "bias", "modulation", "time_embedding", "video_embedder", "final_linear_video", "text_embedder.proj"],
    mergeModes: [
      {
        id: "Block_Guided_V3",
        name_id: "v3.0 Block-Guided Unit-Norm TIES (Rekomendasi Utama)",
        name_en: "v3.0 Block-Guided Unit-Norm TIES (Recommended)",
        desc_id: "Menjaga realisme lensa & pori-pori kulit Base Official + menyerap anatomi Donor dengan Unit-Normalized Voting.",
        desc_en: "Preserves Base Official camera lens & skin pores + absorbs Donor anatomy via Unit-Normalized Voting."
      },
      {
        id: "True_Disjoint_TIES",
        name_id: "v2.0 True Disjoint-TIES (Anti-50% Halving)",
        name_en: "v2.0 True Disjoint-TIES (Anti-50% Halving)",
        desc_id: "Membagi hanya dengan jumlah model yang sepakat di tiap koordinat sehingga fitur unik tetap 100% utuh.",
        desc_en: "Divides only by agreeing model weights at each coordinate so unique features stay 100% intact."
      },
      {
        id: "Norm_Preserving_TIES",
        name_id: "v2.0 Norm-Preserving Disjoint-TIES (Kunci Energi L2)",
        name_en: "v2.0 Norm-Preserving Disjoint-TIES (L2 Energy Lock)",
        desc_id: "Sama seperti True Disjoint-TIES ditambah penguncian L2 Norm otomatis (0.85x–1.35x) setara ketajaman Donor.",
        desc_en: "True Disjoint-TIES plus automatic L2 Norm energy locking (0.85x–1.35x) matching Donor sharpness."
      },
      {
        id: "Pure_Weighted_Linear",
        name_id: "Pure Weighted Linear (Halus & Tanpa Pemotongan Tanda)",
        name_en: "Pure Weighted Linear (Smooth & No Sign Trimming)",
        desc_id: "Rata-rata bobot murni tanpa pemotongan tanda, cocok jika seluruh model donor satu aliran.",
        desc_en: "Pure weighted average without sign trimming, ideal when all donor models share the same training lineage."
      }
    ],
    zones: [
      { id: "early", label_id: "Early Blocks (layers.0–13) — Lensa, Perspektif & Pose", label_en: "Early Blocks (layers.0–13) — Camera, Perspective & Pose", defaultVal: 0.55 },
      { id: "mid", label_id: "Mid Blocks (layers.14–27) — Anatomi & Bentuk Objek", label_en: "Mid Blocks (layers.14–27) — Anatomy & Object Structure", defaultVal: 0.90 },
      { id: "late", label_id: "Late Blocks (layers.28–41) — Pori-Pori Kulit & Cahaya", label_en: "Late Blocks (layers.28–41) — Skin Pores & Lighting", defaultVal: 0.30 },
      { id: "other", label_id: "Other Layers (Embedder & Head)", label_en: "Other Layers (Embedder & Head)", defaultVal: 0.50 }
    ]
  },
  KREA_2: {
    id: "KREA_2",
    name: "KREA 2 (12B 28-Block Single-Stream + TextFusion)",
    badgeColor: "bg-violet-500/20 text-violet-300 border-violet-500/40",
    blocksCount: 28,
    keyPrefixRule: "blocks.{0..27}.{attn,mlp}.* + txtfusion.{layerwise_blocks,projector,refiner_blocks}",
    dtypeRule: "Tensor.Art Mixed Precision: first.weight=BF16 | scale/bias/mod.lin/tmlp/tproj/projector/last.linear/txtmlp=F32",
    int8Spec: "12.56 GB Exact Tensor.Art Layout: 224 INT8 ConvRot (BF16 Hadamard) | Urutan: 1. weight (I8) -> 2. weight_scale (F32) -> 3. comfy_quant ([104] U8)",
    protectedKeywords: ["norm", "mod", "bias", "projector", "first", "last"],
    mergeModes: [
      {
        id: "DARE_Hydra_Krea2",
        name_id: "DARE Hydra Krea 2 Exclusive (Default Teruji + Support Direct INT8)",
        name_en: "DARE Hydra Krea 2 Exclusive (Proven Default + Direct INT8 Support)",
        desc_id: "Menggunakan Top-K TIES Trimming + Mass-Sum Sign Consensus + DARE Rescale dengan proteksi penuh pada first, last, mod, & projector.",
        desc_en: "Uses Top-K TIES Trimming + Mass-Sum Sign Consensus + DARE Rescale with full protection on first, last, mod, & projector."
      },
      {
        id: "Krea2_Zone_Guided_DARE",
        name_id: "Krea 2 Official 28-Block Zone-Guided (ModelMergeKrea2 Spec)",
        name_en: "Krea 2 Official 28-Block Zone-Guided (ModelMergeKrea2 Spec)",
        desc_id: "Mengatur zona resmi Krea 2 (Low 0–8, Mid 9–17, High 18–27, & TextFusion) sambil mengunci first/last/tmlp/tproj agar 100% bebas artefak.",
        desc_en: "Tunes official Krea 2 zones (Low 0–8, Mid 9–17, High 18–27, & TextFusion) while locking first/last/tmlp/tproj for 0% artifacts."
      },
      {
        id: "True_Disjoint_TIES",
        name_id: "True Disjoint-TIES (Dengan Hadamard Un-rotate Otomatis)",
        name_en: "True Disjoint-TIES (With Auto Hadamard Un-rotate)",
        desc_id: "Menggabungkan model Krea 2 BF16 maupun INT8 ConvRot menggunakan normalisasi Disjoint-TIES.",
        desc_en: "Merges Krea 2 BF16 or INT8 ConvRot models using Disjoint-TIES normalization."
      }
    ],
    zones: [
      { id: "early", label_id: "Low Zone (blocks.0–8) — Komposisi & Tata Letak", label_en: "Low Zone (blocks.0–8) — Composition & Layout", defaultVal: 0.75 },
      { id: "mid", label_id: "Mid Zone (blocks.9–17) — Anatomi & Karakter", label_en: "Mid Zone (blocks.9–17) — Anatomy & Subject", defaultVal: 1.00 },
      { id: "late", label_id: "High Zone (blocks.18–27) — Detail Kulit & Lighting", label_en: "High Zone (blocks.18–27) — Skin Detail & Lighting", defaultVal: 0.65 },
      { id: "other", label_id: "TextFusion Stack (txtfusion.* — Kepatuhan Prompt)", label_en: "TextFusion Stack (txtfusion.* — Prompt Adherence)", defaultVal: 0.85 }
    ]
  },
  ANIMA: {
    id: "ANIMA",
    name: "Anima Turbo DiT",
    badgeColor: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    blocksCount: 28,
    keyPrefixRule: "Auto Key Translator: 'net.' (Ultrareal) -> 'model.diffusion_model.' (ComfyUI)",
    dtypeRule: "Uniform BF16 / FP16 (2 Bytes/Element) + Metadata Anima-Turbo-DiT",
    int8Spec: "Standard BF16 / FP16 Streaming Output",
    protectedKeywords: ["norm", "embed", "bias", "modulation"],
    mergeModes: [
      {
        id: "DARE_Hydra_Anima",
        name_id: "Anima DARE Hydra (Auto 'net.' Key Translator)",
        name_en: "Anima DARE Hydra (Auto 'net.' Key Translator)",
        desc_id: "Otomatis menerjemahkan prefix 'net.' milik Ultrareal ke 'model.diffusion_model.' dengan proteksi norm/embed/modulation.",
        desc_en: "Automatically translates Ultrareal 'net.' prefix to 'model.diffusion_model.' with norm/embed/modulation protection."
      }
    ],
    zones: []
  },
  MINIMAX_H3: {
    id: "MINIMAX_H3",
    name: "MiniMax H3 (Ref2VA / Hybrid INT8)",
    badgeColor: "bg-rose-500/20 text-rose-300 border-rose-500/40",
    blocksCount: 32,
    keyPrefixRule: "Dynamic Bidirectional Prefix Matcher (model.diffusion_model. <-> net. <-> bare)",
    dtypeRule: "Native Dtype Preservation (I8 clamped [-128,127], U8 [0,255], BF16/F32 preserved)",
    int8Spec: "Preserves Native INT8 ConvRot & Scale Tensors directly from Base Header",
    protectedKeywords: ["norm", "embed", "bias", "modulation", "scale", "quant", "adaln"],
    mergeModes: [
      {
        id: "DARE_Hydra_Minimax",
        name_id: "MiniMax H3 GPU Smart DARE Hydra (INT8 Safe Clamp)",
        name_en: "MiniMax H3 GPU Smart DARE Hydra (INT8 Safe Clamp)",
        desc_id: "Mempertahankan dtype asli setiap layer di Base Header dengan pembulatan & clamping aman pada tensor INT8.",
        desc_en: "Preserves native layer dtypes from Base Header with safe rounding & clamping on INT8 tensors."
      }
    ],
    zones: []
  }
};

// Mock local files in /content for realistic interactive inspection
const INITIAL_FILES = [
  { name: "mossImage_sft_bf16.safetensors", path: "/content/mossImage_sft_bf16.safetensors", arch: "MOSS_Image", format: "BF16", tensors: 547, sizeGb: 24.12, role: "checkpoint" },
  { name: "dirtyMoss_bf16.safetensors", path: "/content/dirtyMoss_bf16.safetensors", arch: "MOSS_Image", format: "BF16", tensors: 547, sizeGb: 24.12, role: "checkpoint" },
  { name: "pornmasterMoss_v1.safetensors", path: "/content/pornmasterMoss_v1.safetensors", arch: "MOSS_Image", format: "INT8 ConvRot", tensors: 757, sizeGb: 13.48, role: "checkpoint" },
  { name: "solordzZITZIBKrea2_v10MOSSBASE_bf16.safetensors", path: "/content/solordzZITZIBKrea2_v10MOSSBASE_bf16.safetensors", arch: "MOSS_Image", format: "BF16", tensors: 547, sizeGb: 24.12, role: "checkpoint" },
  { name: "krea2TurboOfficialComfy_krea2TurboBf16.safetensors", path: "/content/krea2TurboOfficialComfy_krea2TurboBf16.safetensors", arch: "KREA_2", format: "Mixed BF16/F32", tensors: 430, sizeGb: 24.85, role: "checkpoint" },
  { name: "krea2_sickaestheticlustify.safetensors", path: "/content/krea2_sickaestheticlustify.safetensors", arch: "KREA_2", format: "INT8 ConvRot", tensors: 878, sizeGb: 12.56, role: "checkpoint" },
  { name: "finepornV5INT8FP8_v5BF16.safetensors", path: "/content/finepornV5INT8FP8_v5BF16.safetensors", arch: "KREA_2", format: "INT8 ConvRot", tensors: 878, sizeGb: 12.56, role: "checkpoint" },
  { name: "moodyKrea2MixUncensoredV8SEGoes_v80_bf16.safetensors", path: "/content/moodyKrea2MixUncensoredV8SEGoes_v80_bf16.safetensors", arch: "KREA_2", format: "Mixed BF16/F32", tensors: 430, sizeGb: 24.85, role: "checkpoint" },
  { name: "anima_baseV10.safetensors", path: "/content/anima_baseV10.safetensors", arch: "ANIMA", format: "BF16", tensors: 388, sizeGb: 16.40, role: "checkpoint" },
  { name: "ultrarealFineTune_v30AnimaBase1.safetensors", path: "/content/ultrarealFineTune_v30AnimaBase1.safetensors", arch: "ANIMA", format: "BF16 (net. prefix)", tensors: 388, sizeGb: 16.40, role: "checkpoint" },
  { name: "minimax_h3_ref2va_pruned_int8_convrot.safetensors", path: "/content/minimax_h3_ref2va_pruned_int8_convrot.safetensors", arch: "MINIMAX_H3", format: "Native INT8 ConvRot", tensors: 612, sizeGb: 11.80, role: "checkpoint" },
  { name: "PornMaster_Moss_Detail_Slider_V1.safetensors", path: "/content/PornMaster_Moss_Detail_Slider_V1.safetensors", arch: "MOSS_Image", format: "LoRA (split_q_k_v)", tensors: 504, sizeGb: 0.32, role: "lora" },
  { name: "MOSSRealisticSnapshotV1.safetensors", path: "/content/MOSSRealisticSnapshotV1.safetensors", arch: "MOSS_Image", format: "LoKr (LyCORIS)", tensors: 328, sizeGb: 0.18, role: "lora" },
  { name: "moss_pussy_lora_v2.0.safetensors", path: "/content/moss_pussy_lora_v2.0.safetensors", arch: "MOSS_Image", format: "LoRA (fused_qkv R128)", tensors: 336, sizeGb: 0.49, role: "lora" },
  { name: "Krea_2_zidiusArt_Melancholy_v2.safetensors", path: "/content/Krea_2_zidiusArt_Melancholy_v2.safetensors", arch: "KREA_2", format: "LoRA (Diffusers Alias)", tensors: 512, sizeGb: 0.28, role: "lora" }
];

export default function App() {
  const [lang, setLang] = useState("id"); // "id" | "en"
  const [userLevel, setUserLevel] = useState("beginner"); // "beginner" | "pro"
  const [activeTab, setActiveTab] = useState("merger"); // "download", "merger", "lora", "quant", "xray", "hf", "plugin"
  const [files, setFiles] = useState(INITIAL_FILES);

  // System Resource Monitor State
  const [vramUsed, setVramUsed] = useState(1.2);
  const [diskUsed, setDiskUsed] = useState(64.5);
  const [statusLogs, setStatusLogs] = useState([
    "✅ T4 GPU Environment Ready (CUDA 12.x | PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)",
    "🛡️ Architecture Blueprints Locked: MOSS_Image (42L), KREA_2 (28B+TextFusion), ANIMA, MINIMAX_H3",
    "🔬 Golden Invariant Test Suite: 5/5 PASSED (0% Layer Structure Deviation)"
  ]);
  const [isRunning, setIsRunning] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");

  // Tab 1: Download & File Commander State
  const [dlUrls, setDlUrls] = useState("https://huggingface.co/SEVUNX/omnikrea2/resolve/main/krea2_sickaestheticlustify.safetensors?download=true");
  const [civitaiToken, setCivitaiToken] = useState("");
  const [hfTokenInput, setHfTokenInput] = useState("");
  const [modelscopeRepo, setModelscopeRepo] = useState("");
  const [fileOp, setFileOp] = useState("Copy");
  const [fileSrc, setFileSrc] = useState("/content/krea2_sickaestheticlustify.safetensors");
  const [fileDest, setFileDest] = useState("/content/drive/MyDrive");
  const [driveMounted, setDriveMounted] = useState(false);

  // Tab 2: Checkpoint Merger State
  const [selectedBase, setSelectedBase] = useState("/content/krea2TurboOfficialComfy_krea2TurboBf16.safetensors");
  const [donors, setDonors] = useState([
    { path: "/content/krea2_sickaestheticlustify.safetensors", weight: 0.45 },
    { path: "/content/finepornV5INT8FP8_v5BF16.safetensors", weight: 0.35 },
    { path: "/content/moodyKrea2MixUncensoredV8SEGoes_v80_bf16.safetensors", weight: 0.20 }
  ]);
  const [mergeOutputFormat, setMergeOutputFormat] = useState("int8_convrot");
  const [selectedMergeMode, setSelectedMergeMode] = useState("DARE_Hydra_Krea2");
  const [tiesDensity, setTiesDensity] = useState(0.40);
  const [pDrop, setPDrop] = useState(0.15);
  const [zoneStrengths, setZoneStrengths] = useState({ early: 0.55, mid: 0.90, late: 0.30, other: 0.50 });
  const [mergeOutPath, setMergeOutPath] = useState("/content/Krea2_7sins_merged_int8_convrot.safetensors");

  // Tab 3: LoRA & LoKr Studio State
  const [loraSubTab, setLoraSubTab] = useState("svd_merge"); // "svd_merge", "baker", "extractor"
  const [loraArch, setLoraArch] = useState("MOSS_Image");
  const [qkvMode, setQkvMode] = useState("fused_qkv");
  const [targetRank, setTargetRank] = useState(128);
  const [targetAlpha, setTargetAlpha] = useState(128);
  const [kreaMixPrecision, setKreaMixPrecision] = useState(true);
  const [loraItems, setLoraItems] = useState([
    { path: "/content/PornMaster_Moss_Detail_Slider_V1.safetensors", strength: 4.0 },
    { path: "/content/MOSSRealisticSnapshotV1.safetensors", strength: 0.8 },
    { path: "/content/moss_pussy_lora_v2.0.safetensors", strength: 0.6 }
  ]);

  // Tab 4: Quantizer & Converter State
  const [quantSubTab, setQuantSubTab] = useState("int8_convrot"); // "int8_convrot", "unquantize", "gguf"
  const [quantInputPath, setQuantInputPath] = useState("/content/krea2_sickaestheticlustify.safetensors");
  const [quantOutputPath, setQuantOutputPath] = useState("/content/krea2_sickaestheticlustify_out.safetensors");
  const [ggufProfile, setGgufProfile] = useState("True_Q4_K_M_HighQuality_8.28GiB");
  const [prefixMode, setPrefixMode] = useState("Auto");
  const [restoreRopeBands, setRestoreRopeBands] = useState(true);
  const [injectMossAttnConfig, setInjectMossAttnConfig] = useState(true);

  // Tab 5: X-Ray & Pre-Merge Analyzer State
  const [xraySubTab, setXraySubTab] = useState("analyzer"); // "analyzer" | "xray"
  const [xrayFormat, setXrayFormat] = useState("json");

  // Tab 6: HuggingFace Hub State
  const [hfRepoName, setHfRepoName] = useState("omnikrea2");
  const [hfUploadPaths, setHfUploadPaths] = useState("/content/Krea2_7sins_merged_int8_convrot.safetensors");
  const [hfCommitMsg, setHfCommitMsg] = useState("feat: Upload merged model from Merging Studio v2.0");

  // Auto-detect architecture from selected Base Model
  const detectedBaseFile = useMemo(
    () => files.find((f) => f.path === selectedBase) || files[0],
    [files, selectedBase]
  );
  const currentArchSpec = ARCH_BLUEPRINTS[detectedBaseFile.arch] || ARCH_BLUEPRINTS.KREA_2;

  // Normalized donor weights for visual balance bar
  const normalizedDonorWeights = useMemo(() => {
    const sum = donors.reduce((acc, d) => acc + Number(d.weight || 0), 0);
    if (sum <= 0) return donors.map(() => 100 / Math.max(1, donors.length));
    return donors.map((d) => (Number(d.weight || 0) / sum) * 100);
  }, [donors]);

  const appendLog = (msg) => {
    setStatusLogs((prev) => [...prev.slice(-18), msg]);
  };

  const triggerSimulatedRun = (title, steps) => {
    if (isRunning) return;
    setIsRunning(true);
    setProgressPct(5);
    setProgressLabel(title);
    appendLog(`🚀 [START] ${title}`);

    let idx = 0;
    const timer = setInterval(() => {
      idx += 1;
      if (idx <= steps.length) {
        const s = steps[idx - 1];
        setProgressPct(Math.round((idx / steps.length) * 100));
        setProgressLabel(s);
        appendLog(`   ↳ ${s}`);
      } else {
        clearInterval(timer);
        setIsRunning(false);
        setProgressPct(100);
        appendLog(`✅ [SUCCESS] ${title} selesai dengan 0% perubahan struktur layer!`);
      }
    }, 450);
  };

  const handleFlushSystem = () => {
    setVramUsed(0.4);
    setDiskUsed((prev) => Math.max(24.0, Number((prev - 3.8).toFixed(1))));
    appendLog("🧹 [FLUSH] gc.collect() + torch.cuda.empty_cache() + /tmp & .Trash dibersihkan!");
  };

  const handleSelectBaseModel = (path) => {
    setSelectedBase(path);
    const f = files.find((item) => item.path === path);
    if (f) {
      const spec = ARCH_BLUEPRINTS[f.arch];
      if (spec && spec.mergeModes.length > 0) {
        setSelectedMergeMode(spec.mergeModes[0].id);
      }
      if (f.arch === "MOSS_Image") {
        setZoneStrengths({ early: 0.55, mid: 0.90, late: 0.30, other: 0.50 });
        setPDrop(0.0);
        setMergeOutPath("/content/moss_merged_v3_bf16.safetensors");
      } else if (f.arch === "KREA_2") {
        setZoneStrengths({ early: 0.75, mid: 1.00, late: 0.65, other: 0.85 });
        setPDrop(0.15);
        setMergeOutPath("/content/krea2_merged_out.safetensors");
      } else if (f.arch === "ANIMA") {
        setPDrop(0.40);
        setTiesDensity(0.30);
        setMergeOutPath("/content/Anima_Hydra_Turbo.safetensors");
      } else if (f.arch === "MINIMAX_H3") {
        setPDrop(0.30);
        setTiesDensity(0.40);
        setMergeOutPath("/content/Minimax_H3_Merged_Hybrid.safetensors");
      }
      appendLog(`🔍 [Auto-Inspector] Base Model '${f.name}' terdeteksi sebagai ${f.arch} (${f.format}, ${f.tensors} tensors).`);
    }
  };

  const applyBeginnerPreset = (presetType) => {
    if (detectedBaseFile.arch === "MOSS_Image") {
      if (presetType === "realism_anatomy") {
        setSelectedMergeMode("Block_Guided_V3");
        setZoneStrengths({ early: 0.55, mid: 0.90, late: 0.30, other: 0.50 });
        setPDrop(0.0);
        appendLog("🎛️ [Preset Awam MOSS] Diaktifkan: Pertahankan Lensa/Kulit Base (Early 0.55, Late 0.30) + Serap Anatomi Donor (Mid 0.90).");
      } else if (presetType === "sharp_ties") {
        setSelectedMergeMode("Norm_Preserving_TIES");
        setPDrop(0.0);
        appendLog("🎛️ [Preset Awam MOSS] Diaktifkan: Norm-Preserving Disjoint-TIES (100% Kunci Ketajaman Energi L2).");
      } else {
        setSelectedMergeMode("Pure_Weighted_Linear");
        setPDrop(0.0);
        appendLog("🎛️ [Preset Awam MOSS] Diaktifkan: Pure Weighted Linear (Halus & Bebas Pemotongan).");
      }
    } else if (detectedBaseFile.arch === "KREA_2") {
      if (presetType === "realism_anatomy") {
        setSelectedMergeMode("Krea2_Zone_Guided_DARE");
        setZoneStrengths({ early: 0.75, mid: 1.00, late: 0.65, other: 0.85 });
        setTiesDensity(0.40);
        setPDrop(0.15);
        appendLog("🎛️ [Preset Awam Krea 2] Diaktifkan: 28-Block Zone Guided (Low 0–8: 0.75, Mid 9–17: 1.0, High 18–27: 0.65, TextFusion: 0.85).");
      } else if (presetType === "sharp_ties") {
        setSelectedMergeMode("DARE_Hydra_Krea2");
        setTiesDensity(0.40);
        setPDrop(0.15);
        appendLog("🎛️ [Preset Awam Krea 2] Diaktifkan: DARE Hydra Krea 2 Exclusive (Auto-Unrotate INT8 ConvRot + Protected Layers).");
      } else {
        setSelectedMergeMode("True_Disjoint_TIES");
        setPDrop(0.0);
        appendLog("🎛️ [Preset Awam Krea 2] Diaktifkan: True Disjoint-TIES Tanpa Random Drop.");
      }
    }
  };

  const t = (idText, enText) => (lang === "id" ? idText : enText);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      {/* ================= TOP HEADER BAR ================= */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md px-4 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          {/* Brand Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 via-violet-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Layers className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold tracking-tight text-white">
                  MERGING STUDIO <span className="text-indigo-400">v2.0</span>
                </h1>
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  GitHub + Gradio Edition
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {t(
                  "Arsitektur Layer 100% Terkunci • MOSS Image • Krea 2 (BF16/INT8/GGUF) • Anima • MiniMax H3",
                  "100% Locked Layer Blueprints • MOSS Image • Krea 2 (BF16/INT8/GGUF) • Anima • MiniMax H3"
                )}
              </p>
            </div>
          </div>

          {/* Right Controls: Hardware Monitor, Mode Switch, Flush, Language Switch */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* GPU & Disk Monitor */}
            <div className="hidden md:flex items-center gap-3 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
              <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
                <Cpu className="w-3.5 h-3.5" /> T4 CUDA: {vramUsed} / 15.0 GB
              </span>
              <span className="text-slate-700">|</span>
              <span className="flex items-center gap-1.5 text-sky-400 font-medium">
                <HardDrive className="w-3.5 h-3.5" /> Disk: {diskUsed} / 113 GB
              </span>
            </div>

            {/* Beginner vs Pro Mode Switch */}
            <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => setUserLevel("beginner")}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                  userLevel === "beginner"
                    ? "bg-indigo-600 text-white shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {t("🌱 Mode Panduan Awam", "🌱 Easy Wizard")}
              </button>
              <button
                type="button"
                onClick={() => setUserLevel("pro")}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                  userLevel === "pro"
                    ? "bg-violet-600 text-white shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {t("⚡ Mode Pro Lengkap", "⚡ Pro Studio")}
              </button>
            </div>

            {/* 1-Click Flush Button (Cell 18) */}
            <button
              type="button"
              onClick={handleFlushSystem}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 text-xs font-semibold transition"
              title={t("Bersihkan RAM, Cache CUDA, dan /tmp Colab (Sel 18)", "Flush Python RAM, CUDA Cache, & Colab /tmp (Cell 18)")}
            >
              <RefreshCw className="w-3.5 h-3.5" />
              {t("🧹 Flush RAM & Disk", "🧹 Flush RAM & Disk")}
            </button>

            {/* Instant Language Switch ID / EN */}
            <button
              type="button"
              onClick={() => setLang(lang === "id" ? "en" : "id")}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700 text-xs font-bold transition"
            >
              <Globe className="w-3.5 h-3.5 text-indigo-400" />
              {lang === "id" ? "🇮🇩 ID | EN" : "🇬🇧 EN | ID"}
            </button>
          </div>
        </div>
      </header>

      {/* ================= 6 MAIN NAVIGATION TABS ================= */}
      <div className="border-b border-slate-800 bg-slate-900/60 px-4">
        <div className="max-w-7xl mx-auto flex items-center gap-1.5 overflow-x-auto py-2 no-scrollbar">
          {[
            { id: "download", icon: Download, label_id: "1. 📥 Download & File Manager", label_en: "1. 📥 Download & Files" },
            { id: "merger", icon: Layers, label_id: "2. 🔥 Checkpoint Merger", label_en: "2. 🔥 Checkpoint Merger" },
            { id: "lora", icon: Sparkles, label_id: "3. 🧬 LoRA & LoKr Studio", label_en: "3. 🧬 LoRA & LoKr Studio" },
            { id: "quant", icon: Zap, label_id: "4. ⚡ Quantizer & GGUF", label_en: "4. ⚡ Quantizer & GGUF" },
            { id: "xray", icon: Search, label_id: "5. 🔬 X-Ray & Pre-Merge Analyzer", label_en: "5. 🔬 X-Ray & Analyzer" },
            { id: "hf", icon: UploadCloud, label_id: "6. 🤗 HuggingFace Publisher", label_en: "6. 🤗 HF Hub Publisher" },
            { id: "plugin", icon: FileCode, label_id: "🔌 Plugin Model Baru & Colab Git", label_en: "🔌 New Model Plugin & Git" }
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition ${
                  active
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/25"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                }`}
              >
                <Icon className="w-4 h-4" />
                {lang === "id" ? tab.label_id : tab.label_en}
              </button>
            );
          })}
        </div>
      </div>

      {/* ================= MAIN WORKSPACE AREA ================= */}
      <main className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT / MAIN PANEL (8 COLS) */}
        <div className="lg:col-span-8 space-y-6">
          {/* -----------------------------------------------------------------
              TAB 1: DOWNLOAD & FILE MANAGER (CELL 1, 2, 17)
             ----------------------------------------------------------------- */}
          {activeTab === "download" && (
            <div className="space-y-6">
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-base font-bold text-white flex items-center gap-2">
                      <Download className="w-5 h-5 text-indigo-400" />
                      {t("1A. Unduh Model & Repositori (Aria2c Turbo + Auto-Redirect)", "1A. Download Models & Repos (Aria2c Turbo + Auto-Redirect)")}
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">
                      {t(
                        "Mendukung HuggingFace, Civitai (.red / .com otomatis), ModelScope, dan GitHub dengan 3 lapis fallback (Aria2c -> Curl -> Python Stream).",
                        "Supports HuggingFace, Civitai (.red / .com auto-fix), ModelScope, and GitHub with 3-tier fallback (Aria2c -> Curl -> Python Stream)."
                      )}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setDriveMounted(!driveMounted);
                      appendLog(driveMounted ? "☑️ Google Drive dilepas." : "✅ Google Drive berhasil di-mount di /content/drive");
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                      driveMounted
                        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                        : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
                    }`}
                  >
                    {driveMounted ? "✅ Google Drive Mounted" : t("🚀 Mount Google Drive", "🚀 Mount Google Drive")}
                  </button>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      {t("Link Model URL (Pisahkan dengan koma untuk banyak file)", "Model URLs (Comma-separated for batch download)")}
                    </label>
                    <textarea
                      rows={2}
                      value={dlUrls}
                      onChange={(e) => setDlUrls(e.target.value)}
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                      placeholder="https://huggingface.co/... , https://civitai.com/api/download/models/..."
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        🔒 Civitai API Token {t("(Tidak Hardcoded)", "(Not Hardcoded)")}
                      </label>
                      <input
                        type="password"
                        value={civitaiToken}
                        onChange={(e) => setCivitaiToken(e.target.value)}
                        placeholder={t("Masukkan token Civitai...", "Enter Civitai token...")}
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-1.5 text-xs text-slate-200"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        🔒 HuggingFace Token (Optional)
                      </label>
                      <input
                        type="password"
                        value={hfTokenInput}
                        onChange={(e) => setHfTokenInput(e.target.value)}
                        placeholder="hf_..."
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-1.5 text-xs text-slate-200"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        🌐 ModelScope Repo ID (Optional)
                      </label>
                      <input
                        type="text"
                        value={modelscopeRepo}
                        onChange={(e) => setModelscopeRepo(e.target.value)}
                        placeholder="username/model_name"
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-1.5 text-xs text-slate-200"
                      />
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      triggerSimulatedRun("Aria2c Turbo Multi-Downloader", [
                        "🔍 Resolving redirect link & Content-Disposition header...",
                        "🚀 Running aria2c -x 8 -s 8 -k 1M (Turbo Mode)...",
                        "✅ File verified (>1MB check passed) & saved to /content!"
                      ])
                    }
                    className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/25 transition"
                  >
                    <Download className="w-4 h-4" />
                    {t("Mulai Download Model Sekarang", "Start Downloading Models Now")}
                  </button>
                </div>
              </div>

              {/* 1B. FILE & ARCHIVE COMMANDER (CELL 17) */}
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <FolderOpen className="w-5 h-5 text-amber-400" />
                  {t("1B. File & Archive Commander (Salin 16MB Buffer, Pindah, Hapus, Zip & Unzip)", "1B. File & Archive Commander (16MB Buffered Copy, Move, Delete, Zip & Extract)")}
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">{t("Operasi", "Operation")}</label>
                    <select
                      value={fileOp}
                      onChange={(e) => setFileOp(e.target.value)}
                      className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                    >
                      <option value="Copy">📋 Copy (16MB Progress Buffer)</option>
                      <option value="Move">🚚 Move (Verified Size Before Delete)</option>
                      <option value="Delete">🗑️ Delete File / Folder</option>
                      <option value="Zip (Compress)">📦 Zip (Compress)</option>
                      <option value="Extract (Unzip/Unrar)">📂 Extract (Unzip / Unrar)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Source Path</label>
                    <input
                      type="text"
                      value={fileSrc}
                      onChange={(e) => setFileSrc(e.target.value)}
                      className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Destination Path</label>
                    <input
                      type="text"
                      value={fileDest}
                      onChange={(e) => setFileDest(e.target.value)}
                      className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                    />
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    triggerSimulatedRun(`File Commander: ${fileOp}`, [
                      `📂 Checking source '${fileSrc}'...`,
                      `🚀 Executing ${fileOp} with 16MB streaming buffer...`,
                      `✅ File operation '${fileOp}' verified and completed!`
                    ])
                  }
                  className="px-4 py-2 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/30 text-xs font-semibold transition"
                >
                  {t(`Jalankan Operasi: ${fileOp}`, `Execute Operation: ${fileOp}`)}
                </button>
              </div>
            </div>
          )}

          {/* -----------------------------------------------------------------
              TAB 2: CHECKPOINT MERGER (MOSS v2/v3, KREA 2 BF16/INT8, ANIMA, MINIMAX)
             ----------------------------------------------------------------- */}
          {activeTab === "merger" && (
            <div className="space-y-6">
              {/* Step 1: Select Base Model & Smart Auto-Inspector */}
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <Database className="w-5 h-5 text-indigo-400" />
                    {t("Langkah 1: Pilih Base Model (Deteksi Arsitektur & Layer Otomatis)", "Step 1: Select Base Model (Auto Architecture & Layer Inspector)")}
                  </h2>
                  <span className={`text-xs font-bold px-3 py-1 rounded-full border ${currentArchSpec.badgeColor}`}>
                    🧬 {currentArchSpec.name}
                  </span>
                </div>

                <div>
                  <select
                    value={selectedBase}
                    onChange={(e) => handleSelectBaseModel(e.target.value)}
                    className="w-full rounded-xl bg-slate-950 border border-slate-700 px-3.5 py-2.5 text-xs font-medium text-slate-100 focus:border-indigo-500"
                  >
                    {files
                      .filter((f) => f.role === "checkpoint")
                      .map((f) => (
                        <option key={f.path} value={f.path}>
                          [{f.arch}] {f.name} — ({f.format} | {f.tensors} Tensors | {f.sizeGb} GB)
                        </option>
                      ))}
                  </select>
                </div>

                {/* Smart Inspector Readout Card */}
                <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800/90 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="space-y-1">
                    <div className="text-slate-400 flex items-center gap-1.5">
                      <Lock className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="font-semibold text-slate-200">{t("Aturan Key & Prefix Terkunci:", "Locked Key & Prefix Rule:")}</span>
                    </div>
                    <p className="text-[11px] text-slate-300 font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800">
                      {currentArchSpec.keyPrefixRule}
                    </p>
                  </div>
                  <div className="space-y-1">
                    <div className="text-slate-400 flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                      <span className="font-semibold text-slate-200">{t("Proteksi Layer & Presisi:", "Layer Protection & Dtype:")}</span>
                    </div>
                    <p className="text-[11px] text-slate-300 font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800">
                      {currentArchSpec.dtypeRule}
                    </p>
                  </div>
                  {detectedBaseFile.arch === "KREA_2" && (
                    <div className="md:col-span-2 p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-[11px] text-indigo-200 flex items-start gap-2">
                      <Sparkles className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                      <div>
                        <strong>{t("Fitur Baru Krea 2 Direct INT8 ConvRot Aktif:", "New Krea 2 Direct INT8 ConvRot Active:")}</strong>{" "}
                        {t(
                          "Anda kini bebas mencampur Base/Donor Krea 2 versi BF16 maupun versi INT8 ConvRot (12.56GB) secara langsung! Engine otomatis melakukan Hadamard Un-rotation (size=256) ke FP32 saat menghitung ΔW, dan melakukan Hadamard Re-rotation saat menyimpan ke INT8 ConvRot.",
                          "You can now freely mix Krea 2 BF16 and INT8 ConvRot (12.56GB) Base/Donor models directly! The engine automatically performs Hadamard Un-rotation (size=256) in FP32 before computing ΔW, and re-rotates when saving to INT8 ConvRot."
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Step 2: Donor Models & Auto-Balance Weight Bar */}
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <Sliders className="w-5 h-5 text-emerald-400" />
                    {t("Langkah 2: Model Donor & Proporsi Kontribusi (Bisa Campur BF16 & INT8 ConvRot)", "Step 2: Donor Models & Contribution Weights (Mix BF16 & INT8 ConvRot Freely)")}
                  </h2>
                  <button
                    type="button"
                    onClick={() =>
                      setDonors([...donors, { path: files[0].path, weight: 0.25 }])
                    }
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30 text-xs font-semibold transition"
                  >
                    <Plus className="w-3.5 h-3.5" /> {t("Tambah Donor", "Add Donor")}
                  </button>
                </div>

                {/* Visual Auto-Balance Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>{t("Visualisasi Normalisasi Bobot Otomatis (Total = 100%)", "Automatic Normalized Weight Distribution (Total = 100%)")}</span>
                    <button
                      type="button"
                      onClick={() =>
                        setDonors(donors.map((d) => ({ ...d, weight: Number((1 / donors.length).toFixed(2)) })))
                      }
                      className="text-indigo-400 hover:underline font-medium"
                    >
                      {t("⚖️ Bagi Rata Otomatis (1/N)", "⚖️ Equalize Weights (1/N)")}
                    </button>
                  </div>
                  <div className="h-3 w-full rounded-full overflow-hidden flex bg-slate-950 border border-slate-800">
                    {normalizedDonorWeights.map((pct, i) => {
                      const colors = ["bg-indigo-500", "bg-emerald-500", "bg-amber-500", "bg-rose-500", "bg-sky-500"];
                      return (
                        <div
                          key={i}
                          style={{ width: `${pct}%` }}
                          className={`${colors[i % colors.length]} h-full transition-all duration-300`}
                          title={`Donor ${i + 1}: ${pct.toFixed(1)}%`}
                        />
                      );
                    })}
                  </div>
                </div>

                {/* Donor Rows */}
                <div className="space-y-2.5">
                  {donors.map((donor, idx) => (
                    <div key={idx} className="flex flex-wrap md:flex-nowrap items-center gap-2.5 p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-xs font-bold px-2 py-1 rounded bg-slate-900 text-slate-300 border border-slate-800">
                        #{idx + 1}
                      </span>
                      <select
                        value={donor.path}
                        onChange={(e) => {
                          const next = [...donors];
                          next[idx].path = e.target.value;
                          setDonors(next);
                        }}
                        className="flex-1 rounded-lg bg-slate-900 border border-slate-800 px-2.5 py-1.5 text-xs text-slate-200"
                      >
                        {files
                          .filter((f) => f.role === "checkpoint")
                          .map((f) => (
                            <option key={f.path} value={f.path}>
                              [{f.format}] {f.name}
                            </option>
                          ))}
                      </select>
                      <div className="flex items-center gap-2 w-48">
                        <input
                          type="range"
                          min="0.05"
                          max="1.0"
                          step="0.05"
                          value={donor.weight}
                          onChange={(e) => {
                            const next = [...donors];
                            next[idx].weight = parseFloat(e.target.value);
                            setDonors(next);
                          }}
                          className="w-24 accent-indigo-500"
                        />
                        <span className="text-xs font-mono text-indigo-300 w-20 text-right">
                          {donor.weight.toFixed(2)} ({normalizedDonorWeights[idx].toFixed(0)}%)
                        </span>
                      </div>
                      {donors.length > 1 && (
                        <button
                          type="button"
                          onClick={() => setDonors(donors.filter((_, i) => i !== idx))}
                          className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-slate-900"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Step 3: Beginner 1-Click Presets vs Pro Controls */}
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <Settings className="w-5 h-5 text-violet-400" />
                  {userLevel === "beginner"
                    ? t("Langkah 3: Pilih Preset Pintar 1-Klik (Ramah Pemula)", "Step 3: Choose a 1-Click Smart Preset (Beginner Friendly)")
                    : t("Langkah 3: Konfigurasi Algoritma Merge & Block Zones (Mode Pro)", "Step 3: Merge Algorithm & Block Zone Configuration (Pro Mode)")}
                </h2>

                {/* Beginner 1-Click Preset Cards */}
                {userLevel === "beginner" && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <button
                      type="button"
                      onClick={() => applyBeginnerPreset("realism_anatomy")}
                      className="text-left p-3.5 rounded-xl bg-slate-950 hover:bg-indigo-950/40 border border-slate-800 hover:border-indigo-500/60 transition space-y-1.5"
                    >
                      <div className="text-xs font-bold text-indigo-300 flex items-center justify-between">
                        <span>📸 {t("Realisme Base + Anatomi Donor", "Base Realism + Donor Anatomy")}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300">TERFAVORIT</span>
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">
                        {t(
                          "Menjaga kamera & tekstur pori-pori kulit Base Official tetap murni, sambil menyerap anatomi & pose dari model Donor.",
                          "Keeps Base Official camera & skin pores pure while absorbing anatomy & pose from Donor models."
                        )}
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => applyBeginnerPreset("sharp_ties")}
                      className="text-left p-3.5 rounded-xl bg-slate-950 hover:bg-emerald-950/40 border border-slate-800 hover:border-emerald-500/60 transition space-y-1.5"
                    >
                      <div className="text-xs font-bold text-emerald-300">
                        ⚡ {t("Gabungan Tajam (Anti-Downgrade)", "Sharp Merge (Anti-Downgrade)")}
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">
                        {t(
                          "Mengunci energi L2 Norm / DARE Hydra agar detail gambar tidak luntur atau terpotong 50% saat menggabungkan banyak model.",
                          "Locks L2 Norm / DARE Hydra energy so details never wash out or halve when merging multiple models."
                        )}
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => applyBeginnerPreset("smooth_linear")}
                      className="text-left p-3.5 rounded-xl bg-slate-950 hover:bg-amber-950/40 border border-slate-800 hover:border-amber-500/60 transition space-y-1.5"
                    >
                      <div className="text-xs font-bold text-amber-300">
                        🌊 {t("Campuran Halus Seimbang", "Smooth Balanced Blend")}
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">
                        {t(
                          "Menggabungkan seluruh model secara mulus tanpa pemangkasan ekstrem. Cocok jika model donor berasal dari satu keluarga.",
                          "Blends all models smoothly without aggressive trimming. Ideal when donors come from the same lineage."
                        )}
                      </p>
                    </button>
                  </div>
                )}

                {/* Algorithm Selection & Format */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      {t("Mode Resolusi Konflik & Algoritma Merge", "Conflict Resolution & Merge Mode")}
                    </label>
                    <select
                      value={selectedMergeMode}
                      onChange={(e) => setSelectedMergeMode(e.target.value)}
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                    >
                      {currentArchSpec.mergeModes.map((m) => (
                        <option key={m.id} value={m.id}>
                          {lang === "id" ? m.name_id : m.name_en}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      {t("Format Penyimpanan Output (.safetensors)", "Output Save Format (.safetensors)")}
                    </label>
                    <select
                      value={mergeOutputFormat}
                      onChange={(e) => setMergeOutputFormat(e.target.value)}
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                    >
                      <option value="bf16">💎 BF16 Native (Full Precision + Auto F32 rope.bands/scales)</option>
                      {(detectedBaseFile.arch === "MOSS_Image" || detectedBaseFile.arch === "KREA_2") && (
                        <option value="int8_convrot">
                          ⚡ Direct INT8 ConvRot ({detectedBaseFile.arch === "MOSS_Image" ? "757 Tensors MOSS Exact" : "12.56 GB Tensor.Art Exact"})
                        </option>
                      )}
                      {detectedBaseFile.arch === "KREA_2" && (
                        <option value="mix_bf16_f32">🎯 Krea 2 Mixed Precision BF16/F32 (Tensor.Art Spec)</option>
                      )}
                      {detectedBaseFile.arch === "MINIMAX_H3" && (
                        <option value="native">🛡️ MiniMax H3 Native Header Dtype (INT8 Safe Clamp)</option>
                      )}
                    </select>
                  </div>
                </div>

                {/* Pro Sliders: Block-Guided Zones (Architecture-Specific!) */}
                {(userLevel === "pro" || selectedMergeMode === "Block_Guided_V3" || selectedMergeMode === "Krea2_Zone_Guided_DARE") &&
                  currentArchSpec.zones.length > 0 && (
                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-indigo-300">
                          🎛️ {t(`Pengaturan Kekuatan Per Zona Blok (${currentArchSpec.name})`, `Per-Zone Block Strength (${currentArchSpec.name})`)}
                        </span>
                        <span className="text-[11px] text-emerald-400">
                          {detectedBaseFile.arch === "KREA_2"
                            ? "Protected: first., last., tmlp., txtmlp., tproj. = Locked 1.0"
                            : "Protected: norm, bias, modulation, rope.bands = Locked"}
                        </span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {currentArchSpec.zones.map((z) => (
                          <div key={z.id} className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80 space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="text-slate-300 font-medium">
                                {lang === "id" ? z.label_id : z.label_en}
                              </span>
                              <span className="font-mono font-bold text-indigo-400">
                                {zoneStrengths[z.id]?.toFixed(2)}
                              </span>
                            </div>
                            <input
                              type="range"
                              min="0.0"
                              max="1.2"
                              step="0.05"
                              value={zoneStrengths[z.id] ?? z.defaultVal}
                              onChange={(e) =>
                                setZoneStrengths({ ...zoneStrengths, [z.id]: parseFloat(e.target.value) })
                              }
                              className="w-full accent-indigo-500"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                {/* Pro Sliders: TIES Density & DARE p_drop */}
                {userLevel === "pro" && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300">TIES Top-K Density</span>
                        <span className="font-mono text-indigo-400">{tiesDensity.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.10"
                        max="0.50"
                        step="0.05"
                        value={tiesDensity}
                        onChange={(e) => setTiesDensity(parseFloat(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300">DARE Random Drop (p_drop)</span>
                        <span className="font-mono text-indigo-400">
                          {pDrop.toFixed(2)} {pDrop === 0 ? "(Zero-Noise)" : `(Rescale ${(1 / (1 - pDrop)).toFixed(2)}x)`}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.00"
                        max="0.70"
                        step="0.05"
                        value={pDrop}
                        onChange={(e) => setPDrop(parseFloat(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>
                  </div>
                )}

                {/* Output Path & Execute Button */}
                <div className="flex flex-col md:flex-row gap-3 pt-1">
                  <input
                    type="text"
                    value={mergeOutPath}
                    onChange={(e) => setMergeOutPath(e.target.value)}
                    className="flex-1 rounded-xl bg-slate-950 border border-slate-800 px-3.5 py-2.5 text-xs font-mono text-slate-200"
                    placeholder="/content/merged_output.safetensors"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      triggerSimulatedRun(
                        `Merging ${currentArchSpec.name} [${selectedMergeMode} -> ${mergeOutputFormat.toUpperCase()}]`,
                        [
                          `🔍 Reading Base Header & verifying ${detectedBaseFile.tensors} layers...`,
                          `🔄 Decoding INT8 ConvRot Donors via Hadamard 256 Un-rotation to FP32...`,
                          `🧮 Computing ${selectedMergeMode} with Protected Layer Guards...`,
                          mergeOutputFormat === "int8_convrot"
                            ? `📐 Applying Forward Hadamard ConvRot & writing ${currentArchSpec.int8Spec.split(":")[0]}...`
                            : `💾 Streaming 8-byte aligned Safetensors binary to ${mergeOutPath}...`
                        ]
                      )
                    }
                    className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/30 transition"
                  >
                    <Play className="w-4 h-4 fill-current" />
                    {t("🚀 Jalankan Merge Sekarang", "🚀 Execute Checkpoint Merge")}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* -----------------------------------------------------------------
              TAB 3: LORA & LOKR STUDIO (CELL 5, 7, 8, 9, 13)
             ----------------------------------------------------------------- */}
          {activeTab === "lora" && (
            <div className="space-y-6">
              {/* Sub-tab selector */}
              <div className="flex flex-wrap gap-2 p-1.5 rounded-xl bg-slate-900 border border-slate-800">
                {[
                  { id: "svd_merge", label_id: "3A. 🧬 Multi-LoRA / LoKr SVD Merger", label_en: "3A. 🧬 Multi-LoRA / LoKr SVD Merger" },
                  { id: "baker", label_id: "3B. 🔥 Universal LoRA Baker ke Checkpoint (v5.2)", label_en: "3B. 🔥 Universal LoRA Baker to Checkpoint (v5.2)" },
                  { id: "extractor", label_id: "3C. ⚡ Fast LoRA Extractor (Tuned - Base SVD)", label_en: "3C. ⚡ Fast LoRA Extractor (Tuned - Base SVD)" }
                ].map((st) => (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => setLoraSubTab(st.id)}
                    className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold transition ${
                      loraSubTab === st.id
                        ? "bg-indigo-600 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {lang === "id" ? st.label_id : st.label_en}
                  </button>
                ))}
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">
                      {t("Arsitektur Target LoRA", "Target LoRA Architecture")}
                    </label>
                    <select
                      value={loraArch}
                      onChange={(e) => setLoraArch(e.target.value)}
                      className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                    >
                      <option value="MOSS_Image">MOSS Image (Auto Fused QKV &lt;-&gt; Split Q/K/V)</option>
                      <option value="KREA_2">KREA 2 (Full mlp.down &amp; Diffusers Alias Support)</option>
                      <option value="MINIMAX_H3">MiniMax H3 (Tensor.Art Deployable Metadata)</option>
                    </select>
                  </div>

                  {loraSubTab !== "baker" && (
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        Target SVD Rank / Alpha
                      </label>
                      <select
                        value={targetRank}
                        onChange={(e) => {
                          const r = parseInt(e.target.value, 10);
                          setTargetRank(r);
                          setTargetAlpha(r);
                        }}
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                      >
                        <option value={8}>Rank 8 / Alpha 8 (Ringan ~58 MB)</option>
                        <option value={32}>Rank 32 / Alpha 32</option>
                        <option value={64}>Rank 64 / Alpha 64 (Standar MiniMax / Krea 2)</option>
                        <option value={128}>Rank 128 / Alpha 128 (Rekomendasi MOSS LoKr)</option>
                        <option value={256}>Rank 256 / Alpha 256 (Ultra High-Fidelity)</option>
                      </select>
                    </div>
                  )}

                  {loraArch === "MOSS_Image" && loraSubTab === "svd_merge" && (
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">
                        MOSS Attention Output Mode (`qkv_output_mode`)
                      </label>
                      <select
                        value={qkvMode}
                        onChange={(e) => setQkvMode(e.target.value)}
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                      >
                        <option value="fused_qkv">fused_qkv (layers.N.proj.qkv — 336 Tensors)</option>
                        <option value="split_q_k_v">split_q_k_v (layers.N.proj.{"{q,k,v}"} — 504 Tensors ComfyUI)</option>
                      </select>
                    </div>
                  )}

                  {loraSubTab === "baker" && (
                    <div className="md:col-span-2 flex items-center gap-3 pt-5">
                      <label className="flex items-center gap-2 text-xs text-slate-200 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={kreaMixPrecision}
                          onChange={(e) => setKreaMixPrecision(e.target.checked)}
                          className="rounded accent-indigo-500"
                        />
                        {t(
                          "Aktifkan Krea 2 Mixed Precision BF16/F32 (Standar Resmi Tensor.Art)",
                          "Enable Krea 2 Mixed Precision BF16/F32 (Official Tensor.Art Standard)"
                        )}
                      </label>
                    </div>
                  )}
                </div>

                {/* LoRA Adapter List */}
                {loraSubTab !== "extractor" ? (
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-semibold text-slate-300">
                        {t("Daftar LoRA / LoKr yang Akan Diproses:", "LoRA / LoKr Adapters to Process:")}
                      </span>
                      <button
                        type="button"
                        onClick={() =>
                          setLoraItems([...loraItems, { path: "/content/moss_pussy_lora_v2.0.safetensors", strength: 0.5 }])
                        }
                        className="text-xs text-indigo-400 hover:underline flex items-center gap-1"
                      >
                        <Plus className="w-3.5 h-3.5" /> {t("Tambah Adapter", "Add Adapter")}
                      </button>
                    </div>
                    {loraItems.map((item, i) => (
                      <div key={i} className="flex items-center gap-2.5 p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                        <input
                          type="text"
                          value={item.path}
                          onChange={(e) => {
                            const next = [...loraItems];
                            next[i].path = e.target.value;
                            setLoraItems(next);
                          }}
                          className="flex-1 rounded-lg bg-slate-900 border border-slate-800 px-3 py-1.5 text-xs text-slate-200 font-mono"
                        />
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-400">Strength:</span>
                          <input
                            type="number"
                            step="0.1"
                            value={item.strength}
                            onChange={(e) => {
                              const next = [...loraItems];
                              next[i].strength = parseFloat(e.target.value) || 0;
                              setLoraItems(next);
                            }}
                            className="w-20 rounded-lg bg-slate-900 border border-slate-800 px-2 py-1.5 text-xs text-indigo-300 font-mono"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Base Model Path</label>
                      <input
                        type="text"
                        defaultValue="/content/krea2_turbo_bf16.safetensors"
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Fine-Tuned Model Path</label>
                      <input
                        type="text"
                        defaultValue="/content/kroma-v0.3-turbo.safetensors"
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono"
                      />
                    </div>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() =>
                    triggerSimulatedRun(`LoRA Studio (${loraSubTab.toUpperCase()} - ${loraArch})`, [
                      "🔍 Parsing LoRA/LoKr roles (Strict protection on mlp.down & ffn_down)...",
                      loraArch === "MOSS_Image"
                        ? `🔀 Unifying QKV projections (${qkvMode}) & computing Kronecker LoKr deltas...`
                        : "🔀 Translating Diffusers/ModelScope aliases to Native Krea 2 / MiniMax keys...",
                      `🧮 Running Randomized Low-Rank SVD on GPU (Rank=${targetRank})...`,
                      "✅ Saved SafeTensors with official ai-toolkit / Tensor.Art deployment metadata!"
                    ])
                  }
                  className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs flex items-center justify-center gap-2 transition"
                >
                  <Sparkles className="w-4 h-4" />
                  {t("Jalankan Proses LoRA / LoKr Sekarang", "Run LoRA / LoKr Process Now")}
                </button>
              </div>
            </div>
          )}

          {/* -----------------------------------------------------------------
              TAB 4: QUANTIZER & GGUF CONVERTER (CELL 14, 15, 16, 17)
             ----------------------------------------------------------------- */}
          {activeTab === "quant" && (
            <div className="space-y-6">
              <div className="flex flex-wrap gap-2 p-1.5 rounded-xl bg-slate-900 border border-slate-800">
                {[
                  { id: "int8_convrot", label_id: "4A. 🎯 Universal INT8 ConvRot Quantizer (MOSS 757T & Krea 2 12.56GB)", label_en: "4A. 🎯 Universal INT8 ConvRot Quantizer" },
                  { id: "unquantize", label_id: "4B. 🔓 Universal Unquantizer v5.0 (INT8/FP8 -> BF16)", label_en: "4B. 🔓 Universal Unquantizer v5.0 (INT8/FP8 -> BF16)" },
                  { id: "gguf", label_id: "4C. ⚡ Krea 2 to GGUF T4 GPU Converter v3.0", label_en: "4C. ⚡ Krea 2 to GGUF T4 GPU Converter v3.0" }
                ].map((qt) => (
                  <button
                    key={qt.id}
                    type="button"
                    onClick={() => setQuantSubTab(qt.id)}
                    className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold transition ${
                      quantSubTab === qt.id
                        ? "bg-indigo-600 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {lang === "id" ? qt.label_id : qt.label_en}
                  </button>
                ))}
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Input Model Path (.safetensors)</label>
                    <input
                      type="text"
                      value={quantInputPath}
                      onChange={(e) => setQuantInputPath(e.target.value)}
                      className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono text-slate-200"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">
                      Output Path ({quantSubTab === "gguf" ? ".gguf" : ".safetensors"})
                    </label>
                    <input
                      type="text"
                      value={quantOutputPath}
                      onChange={(e) => setQuantOutputPath(e.target.value)}
                      className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono text-slate-200"
                    />
                  </div>
                </div>

                {quantSubTab === "gguf" ? (
                  <div className="space-y-3">
                    <label className="block text-xs font-semibold text-slate-300">
                      {t("Pilih Profil Kuantisasi GGUF Krea 2 (T4 CUDA Kernel):", "Select Krea 2 GGUF Quantization Profile (T4 CUDA Kernel):")}
                    </label>
                    <select
                      value={ggufProfile}
                      onChange={(e) => setGgufProfile(e.target.value)}
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2.5 text-xs text-slate-200"
                    >
                      <option value="True_Q4_K_M_HighQuality_8.28GiB">
                        💎 True_Q4_K_M_HighQuality_8.28GiB (Upgrade Q3_K &amp; IQ4_XS ke Q4_K + Pertahankan Q5_K &amp; Q6_K)
                      </option>
                      <option value="Byteshape_Exact_7.35GiB">
                        📦 Byteshape_Exact_7.35GiB (100% Identik JSON ByteShape: Q3_K + IQ4_XS + Q5_K + Q6_K)
                      </option>
                    </select>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-slate-400 mb-1">Prefix Alignment Mode</label>
                      <select
                        value={prefixMode}
                        onChange={(e) => setPrefixMode(e.target.value)}
                        className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                      >
                        <option value="Auto">Auto (Bare keys untuk MOSS, Original untuk Krea 2)</option>
                        <option value="Strip_Prefix">Strip_Prefix (Hapus model.diffusion_model.)</option>
                        <option value="Add_Prefix">Add_Prefix (Tambah model.diffusion_model.)</option>
                        <option value="Keep_Original">Keep_Original</option>
                      </select>
                    </div>
                    <label className="flex items-center gap-2 text-xs text-slate-300 pt-5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={restoreRopeBands}
                        onChange={(e) => setRestoreRopeBands(e.target.checked)}
                        className="rounded accent-indigo-500"
                      />
                      Restore MOSS `rope.bands` ([16] F32)
                    </label>
                    {quantSubTab === "int8_convrot" && (
                      <label className="flex items-center gap-2 text-xs text-slate-300 pt-5 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={injectMossAttnConfig}
                          onChange={(e) => setInjectMossAttnConfig(e.target.checked)}
                          className="rounded accent-indigo-500"
                        />
                        Inject 42 `comfy_attention.config` ([35] U8)
                      </label>
                    )}
                  </div>
                )}

                <button
                  type="button"
                  onClick={() =>
                    triggerSimulatedRun(
                      quantSubTab === "gguf"
                        ? `Krea 2 to GGUF T4 CUDA Converter (${ggufProfile})`
                        : `Universal ${quantSubTab.toUpperCase()} Streaming Engine`,
                      quantSubTab === "gguf"
                        ? [
                            "🛡️ Running ComfyUI-GGUF Decoder Self-Test (Q6_K, Q5_K, Q4_K, IQ4_XS, Q3_K Cosine > 0.99)...",
                            "📐 Building 32-byte aligned GGUF v3 header (general.architecture = qwen_image)...",
                            "⚡ Streaming T4 GPU quantization in 65,536-block batches..."
                          ]
                        : [
                            "🔍 Auto-detecting model architecture & building exact byte offsets...",
                            "📐 Executing Hadamard 256 group rotation on GPU...",
                            "💾 Streaming 8-byte aligned Safetensors output to disk..."
                          ]
                    )
                  }
                  className="w-full py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center justify-center gap-2 transition"
                >
                  <Zap className="w-4 h-4" />
                  {t("Mulai Proses Konversi / Kuantisasi", "Start Conversion / Quantization")}
                </button>
              </div>
            </div>
          )}

          {/* -----------------------------------------------------------------
              TAB 5: X-RAY & PRE-MERGE ANALYZER (CELL 6 & CELL 13B)
             ----------------------------------------------------------------- */}
          {activeTab === "xray" && (
            <div className="space-y-6">
              <div className="flex gap-2 p-1.5 rounded-xl bg-slate-900 border border-slate-800">
                <button
                  type="button"
                  onClick={() => setXraySubTab("analyzer")}
                  className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold transition ${
                    xraySubTab === "analyzer" ? "bg-indigo-600 text-white" : "text-slate-400"
                  }`}
                >
                  5A. 🔬 Pre-Merge Tensor &amp; Conflict Analyzer v2.0
                </button>
                <button
                  type="button"
                  onClick={() => setXraySubTab("xray")}
                  className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold transition ${
                    xraySubTab === "xray" ? "bg-indigo-600 text-white" : "text-slate-400"
                  }`}
                >
                  5B. 🕵️ The X-Ray (Export Metadata &amp; Tensor List ke JSON/TXT)
                </button>
              </div>

              {xraySubTab === "analyzer" ? (
                <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        {t("Simulasi Pra-Merge: Energi L2 Norm, Anatomi Blok & Konflik TIES", "Pre-Merge Simulation: L2 Norm Energy, Block Anatomy & TIES Conflict")}
                      </h3>
                      <p className="text-xs text-slate-400">
                        {t("Mendukung dekode otomatis donor INT8 ConvRot sebelum dibandingkan dengan Base Official.", "Supports automatic Hadamard decoding of INT8 ConvRot donors against Base Official.")}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        triggerSimulatedRun("Pre-Merge Tensor Conflict Analyzer v2.0", [
                          "🔬 Decoding Base & Donor tensors to FP32 (Hadamard Un-rotate active)...",
                          "📊 Calculating Region L2 Norms & Pairwise Task-Vector Cosine Similarity...",
                          "🎯 Comparing Energy Retention: Old v1.0 vs Linear vs True Disjoint-TIES vs Norm-Preserving..."
                        ])
                      }
                      className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold"
                    >
                      {t("🔬 Analisis Sekarang", "🔬 Run Analysis")}
                    </button>
                  </div>

                  {/* Simulated Report Table */}
                  <div className="overflow-x-auto rounded-xl border border-slate-800">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-2.5 px-3">Bagian Arsitektur (Region)</th>
                          <th className="py-2.5 px-3">Donor #1 (0.45)</th>
                          <th className="py-2.5 px-3">Donor #2 (0.35)</th>
                          <th className="py-2.5 px-3">Donor #3 (0.20)</th>
                          <th className="py-2.5 px-3">TIES Sign Agreement</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/70 bg-slate-900/50 font-mono text-[11px]">
                        <tr>
                          <td className="py-2 px-3 text-slate-200">1. Text Conditioning / TextFusion</td>
                          <td className="py-2 px-3 text-emerald-400">0.4210%</td>
                          <td className="py-2 px-3 text-amber-400">1.8420%</td>
                          <td className="py-2 px-3 text-sky-400">0.3105%</td>
                          <td className="py-2 px-3 text-emerald-300">84.2% Searah</td>
                        </tr>
                        <tr>
                          <td className="py-2 px-3 text-slate-200">2. Early DiT Blocks (Komposisi &amp; Pose)</td>
                          <td className="py-2 px-3 text-emerald-400">1.9240%</td>
                          <td className="py-2 px-3 text-amber-400">4.6110%</td>
                          <td className="py-2 px-3 text-sky-400">1.5400%</td>
                          <td className="py-2 px-3 text-emerald-300">76.8% Searah</td>
                        </tr>
                        <tr>
                          <td className="py-2 px-3 text-slate-200">3. Mid DiT Blocks (Struktur &amp; Anatomi)</td>
                          <td className="py-2 px-3 text-emerald-400">2.8450%</td>
                          <td className="py-2 px-3 text-amber-400">6.1920%</td>
                          <td className="py-2 px-3 text-sky-400">2.1180%</td>
                          <td className="py-2 px-3 text-amber-300">68.4% Searah</td>
                        </tr>
                        <tr>
                          <td className="py-2 px-3 text-slate-200">4. Late DiT Blocks (Pori Kulit &amp; Lighting)</td>
                          <td className="py-2 px-3 text-emerald-400">1.1520%</td>
                          <td className="py-2 px-3 text-amber-400">3.4890%</td>
                          <td className="py-2 px-3 text-sky-400">0.9840%</td>
                          <td className="py-2 px-3 text-emerald-300">79.5% Searah</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-sm font-bold text-white">
                      🕵️ The X-Ray: Export Metadata &amp; Tensor Blueprint (Single &amp; Batch Scan)
                    </h3>
                    <div className="flex items-center gap-2">
                      <select
                        value={xrayFormat}
                        onChange={(e) => setXrayFormat(e.target.value)}
                        className="rounded-lg bg-slate-950 border border-slate-800 px-3 py-1.5 text-xs text-slate-200"
                      >
                        <option value="json">Format .JSON</option>
                        <option value="txt">Format .TXT Tabel</option>
                      </select>
                      <button
                        type="button"
                        onClick={() =>
                          triggerSimulatedRun(`The X-Ray Batch Export (${xrayFormat.toUpperCase()})`, [
                            "🔍 Scanning /content for .safetensors headers...",
                            `📝 Exporting tensor shapes, dtypes & metadata to .${xrayFormat}...`
                          ])
                        }
                        className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold"
                      >
                        Export {xrayFormat.toUpperCase()}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* -----------------------------------------------------------------
              TAB 6: HUGGINGFACE HUB PUBLISHER (CELL 11B & 12)
             ----------------------------------------------------------------- */}
          {activeTab === "hf" && (
            <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <UploadCloud className="w-5 h-5 text-amber-400" />
                {t("6. HuggingFace Hub Login & Multi-File Publisher (Anti-Overwrite Guard)", "6. HuggingFace Hub Login & Multi-File Publisher (Anti-Overwrite Guard)")}
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">HuggingFace WRITE Token</label>
                  <input
                    type="password"
                    value={hfTokenInput}
                    onChange={(e) => setHfTokenInput(e.target.value)}
                    placeholder="hf_..."
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Target Model Repo Name</label>
                  <input
                    type="text"
                    value={hfRepoName}
                    onChange={(e) => setHfRepoName(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-xs font-medium text-slate-400 mb-1">
                    {t("Path File / Folder yang Akan Diupload (Bisa dipisah koma ',')", "Files / Folders to Upload (Comma-separated)")}
                  </label>
                  <input
                    type="text"
                    value={hfUploadPaths}
                    onChange={(e) => setHfUploadPaths(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono text-slate-200"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-xs font-medium text-slate-400 mb-1">Commit Message</label>
                  <input
                    type="text"
                    value={hfCommitMsg}
                    onChange={(e) => setHfCommitMsg(e.target.value)}
                    className="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-xs text-slate-200"
                  />
                </div>
              </div>
              <button
                type="button"
                onClick={() =>
                  triggerSimulatedRun(`Uploading to HuggingFace Hub (${hfRepoName})`, [
                    "🔑 Validating HuggingFace WRITE token & creating repo if needed...",
                    "📦 Uploading file stream with anti-overwrite path protection...",
                    `✅ Published to https://huggingface.co/SEVUNX/${hfRepoName}`
                  ])
                }
                className="w-full py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 transition"
              >
                <UploadCloud className="w-4 h-4" />
                {t("Login & Upload ke HuggingFace Hub", "Login & Upload to HuggingFace Hub")}
              </button>
            </div>
          )}

          {/* -----------------------------------------------------------------
              TAB 7: ZERO-RISK PLUGIN GENERATOR & 1-CELL COLAB LAUNCHER
             ----------------------------------------------------------------- */}
          {activeTab === "plugin" && (
            <div className="space-y-6">
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-3">
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <FileCode className="w-5 h-5 text-emerald-400" />
                  {t("Cara Kerja 1-Cell Colab Launcher & Sistem Plugin Model Baru", "How the 1-Cell Colab Launcher & New Model Plugin System Work")}
                </h2>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {t(
                    "Di repositori GitHub Anda, setiap arsitektur model dipisahkan ke dalam folder `architectures/`. Ketika ada model baru keluar di masa depan, Anda cukup membuat 1 file baru di `architectures/` tanpa pernah menyentuh `moss_image.py`, `krea2.py`, `anima.py`, atau `minimax_h3.py`!",
                    "In your GitHub repository, each model architecture is isolated inside `architectures/`. When a new model is released in the future, you simply add 1 file in `architectures/` without ever touching `moss_image.py`, `krea2.py`, `anima.py`, or `minimax_h3.py`!"
                  )}
                </p>
                <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px] text-emerald-300 overflow-x-auto">
                  <div className="text-slate-400 mb-1"># Isi Sel Tunggal di Colab_Launcher.ipynb:</div>
                  <div>!git clone https://github.com/USERNAME/Merging-Studio-v2.git /content/Merging-Studio-v2</div>
                  <div>!pip install -q gradio fastapi uvicorn safetensors torch numpy tqdm huggingface_hub</div>
                  <div>!python3 /content/Merging-Studio-v2/app.py --share</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT SIDEBAR: ARCHITECTURE INVARIANT GUARD & LIVE TERMINAL (4 COLS) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Architecture Invariant Guard Card */}
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                {t("Pelindung Struktur Layer", "Layer Invariant Guard")}
              </h3>
              <span className="text-[11px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
                100% LOCKED
              </span>
            </div>

            <div className="space-y-2 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-slate-400 text-[11px]">{t("Spesifikasi INT8 ConvRot Aktif:", "Active INT8 ConvRot Blueprint:")}</div>
                <div className="text-slate-200 font-medium mt-0.5 text-[11px] leading-snug">
                  {currentArchSpec.int8Spec}
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-slate-400 text-[11px] mb-1">{t("Keyword Layer yang Dilindungi:", "Protected Layer Keywords:")}</div>
                <div className="flex flex-wrap gap-1">
                  {currentArchSpec.protectedKeywords.map((kw) => (
                    <span
                      key={kw}
                      className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Real-Time Progress & Live Console */}
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Terminal className="w-4 h-4 text-indigo-400" />
                {t("Terminal & Progress Real-Time", "Live Terminal & Progress")}
              </h3>
              <span className="text-xs font-mono text-indigo-400">{progressPct}%</span>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-2.5 rounded-full bg-slate-950 border border-slate-800 overflow-hidden">
              <div
                style={{ width: `${progressPct}%` }}
                className="h-full bg-gradient-to-r from-indigo-500 via-violet-500 to-emerald-500 transition-all duration-300"
              />
            </div>
            {progressLabel && (
              <div className="text-[11px] text-indigo-300 font-mono truncate">{progressLabel}</div>
            )}

            {/* Console Output */}
            <div className="h-64 rounded-xl bg-slate-950 border border-slate-800 p-3 font-mono text-[11px] text-slate-300 overflow-y-auto space-y-1.5">
              {statusLogs.map((log, index) => (
                <div key={index} className="leading-relaxed break-words">
                  {log}
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
