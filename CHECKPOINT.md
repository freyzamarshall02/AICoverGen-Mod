# AICoverGen-Mod — Session Checkpoint
> Last updated: 2026-10-20  
> Latest commit: `ce072d2` on `main` → `freyzamarshall02/AICoverGen-Mod`

---

## ✅ Fixes Already Done & Pushed

### 1. Swift ONNX Download Fix (`src/rvc/lib/predictors/Swift.py`)
- **Problem**: `from rvc.lib.tools.http_download import download_http` — module `rvc.lib.tools` does not exist, crashing whenever Swift pitch extractor was selected and `swift.onnx` was missing.
- **Fix**: Replaced with stdlib `urllib.request.urlretrieve`.
- **Also**: Pre-downloaded `swift.onnx` (135 KB) to `rvc_models/predictors/swift.onnx`.
- **Note**: On Colab the model still re-downloads on first run to `/content/AICoverGen-Mod/rvc_models/predictors/swift.onnx` — this is correct behaviour now.

### 2. Separation Performance Fix (`src/mdx.py` + `requirements.txt`)
- **Problem A**: `requirements.txt` listed **both** `onnxruntime>=1.17.0` (CPU-only) and `audio-separator[gpu]` (which pulls `onnxruntime-gpu`). Whichever pip installed last won — if plain `onnxruntime` won, `CUDAExecutionProvider` disappeared, forcing KARA_2 and Reverb FoxJoy MDX stages onto CPU (~8–14 min each vs ~30 sec on GPU).
- **Fix**: Removed standalone `onnxruntime` line from `requirements.txt`. `audio-separator[gpu]` already brings `onnxruntime-gpu` — no conflict.
- **Result confirmed in log**: `ONNXruntime has CUDAExecutionProvider available, enabling acceleration` ✅  KARA_2: 19s, Reverb: 27s (was 4+ min each).

- **Problem B**: `run_roformer()` called `librosa.load()` on the audio file then never used the result — the subprocess reads the file itself. Pure RAM waste (~95 MB for a 5-min song).
- **Fix**: Removed the `librosa.load` line.

- **Problem C**: BS-Roformer `overlap=4` — doubled processing time with marginal quality gain.
- **Fix**: Lowered to `overlap=2`. Stage 1 time: 3m42s (was 7m29s). ✅

---

## ✅ RVC dtype mismatch — FIXED

### Error (was)
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
```

### Fix Applied
In `src/rvc_module.py` `get_vc()` — replaced the `if is_half: net_g.half()` branch with an unconditional `net_g.float()`, matching how Applio's `rvc/infer/infer.py` line 529 handles it (`self.net_g.to(device).float()`). The `pipeline.py` already passes `feats.float()`, so forcing net_g to fp32 resolves the mismatch. Applio deliberately removed fp16 inference for stability.

---

## Pipeline Overview (for context)

```
Song Input
  │
  ▼ Stage 1 — BS-Roformer (audio-separator CLI, GPU ✅)
  │  model_bs_roformer_ep_317_sdr_12.9755.ckpt  overlap=2
  │  → Vocals.wav + Instrumental.wav
  │
  ▼ Stage 2 — UVR_MDXNET_KARA_2.onnx (ONNX, GPU via CUDAExecutionProvider ✅)
  │  denoise=True (2 passes)
  │  → Backup_Vocals.wav + Main_Vocals.wav
  │
  ▼ Stage 3 — Reverb_HQ_By_FoxJoy.onnx (ONNX, GPU ✅)
  │  denoise=True (2 passes)
  │  → Main_Vocals_DeReverb.wav
  │
  ▼ RVC Voice Conversion  ✅ FIXED
     net_g forced to fp32 (rvc_module.py get_vc)
     feats.float() in pipeline.py → both fp32, no mismatch
```

---

## Key File Locations

| File | Purpose |
|---|---|
| `src/main.py` | Pipeline orchestrator, `preprocess_song()`, `voice_change()` |
| `src/mdx.py` | `run_mdx()` + `run_roformer()` — audio separation |
| `src/rvc_module.py` | `load_hubert()`, `get_vc()`, `rvc_infer()` |
| `src/rvc/infer/pipeline.py` | Main RVC inference pipeline — `VC.pipeline()`, `voice_conversion()` |
| `src/rvc/lib/algorithm/synthesizers.py` | `SynthesizerTrnMs256NSFsid.infer()` |
| `src/rvc/lib/algorithm/encoders.py` | `TextEncoder.forward()` — crash site |
| `src/rvc/configs/config.py` | `Config` class — `is_half`, device detection |
| `src/rvc/lib/predictors/Swift.py` | Swift F0 predictor — fixed ✅ |
| `requirements.txt` | Deps — `onnxruntime` conflict fixed ✅ |
| `rvc_models/predictors/swift.onnx` | Pre-downloaded 135KB model ✅ |

---

## Separation Model Comparison (new vs old)

| Stage | Old (`AICoverGen_Mod_Old_HF`) | New (`AICoverGen-Mod`) |
|---|---|---|
| Vocal/Instrumental | `Kim_Vocal_2.onnx` (MDX-Net) | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (BS-Roformer) |
| Main/Backup Vocals | `UVR_MDXNET_KARA_2.onnx` | `UVR_MDXNET_KARA_2.onnx` (same) |
| DeReverb | `Reverb_HQ_By_FoxJoy.onnx` | `Reverb_HQ_By_FoxJoy.onnx` (same) |
