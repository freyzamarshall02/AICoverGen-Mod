# AICoverGen-Mod — Modernization Plan

> **Target Environment**
> - Python 3.13.x (Google Colab)
> - CUDA 13.0 / Driver 13.0
> - GPU: Tesla T4
> - PyTorch 2.11.0+cu130
> - Running as **no-UI Colab cells only** (no `.ipynb` generation)

---

## 1. Project Overview

AICoverGen-Mod is an AI cover-song pipeline that chains:
1. **Audio download** (yt-dlp / local file)
2. **Source separation** (MDX-Net ONNX + Roformer via `audio-separator`)
3. **Voice conversion** (RVC v2 with HuBERT embedder + FAISS index)
4. **Post-processing** (reverb/effects via pedalboard, mix via pydub, pitch shift via sox)

The project was originally built against older library versions (fairseq HuBERT, old Gradio 3.x API, old RVC infer_pack). It needs a full rework to run cleanly on the 2026 Colab environment.

---

## 2. Root Problems

| Area | Current (broken) | Target |
|---|---|---|
| HuBERT embedder | `fairseq.checkpoint_utils.load_model_ensemble_and_task()` | `transformers.HubertModel` + custom `final_proj` (Applio pattern) |
| RVC model classes | `infer_pack/models.py` with 4 separate Synthesizer classes | Applio's unified `Synthesizer` class from `rvc/lib/algorithm/synthesizers.py` |
| VC pipeline | `vc_infer_pipeline.py` using old `model.extract_features(**inputs)` API | Applio's `rvc/infer/pipeline.py` using `model(feats)["last_hidden_state"]` |
| F0 methods | mangio-crepe + rmvpe (standalone) | Applio RMVPE/FCPE/CREPE predictors from `rvc/lib/predictors/` |
| `my_utils.py` audio loading | ffmpeg-python subprocess | librosa + soxr (Applio pattern in `rvc/lib/utils.py`) |
| `sox` dependency | `import sox` for pitch shift | Replace with `subprocess` calling ffmpeg |
| Gradio UI | Gradio 3.x API (`gr.DataFrame.update`, `gr.Text.update`, `gr.CheckboxGroup.update`) | Gradio 6.x API (components return values directly, no `.update()`) |
| ONNX runtime | bare `onnxruntime_gpu` install | `onnxruntime>=1.17.0` (non-Windows) — Applio's pattern |
| requirements | loose/unpinned with fairseq, deemix | pinned modern deps, fairseq removed, audio-separator retained |

---

## 3. Dependency Changes

### Remove entirely
- `fairseq` / `git+https://github.com/liyaodev/fairseq.git` — replaced by `transformers`
- `deemix` — not used in core pipeline
- `sox` (Python package) — replace pitch-shift with ffmpeg subprocess call
- `onnxruntime_gpu` (old unversioned pin) — use `onnxruntime>=1.17.0`
- `lib` — unknown placeholder, remove

### New `requirements.txt`
```
# Core audio/ML
numpy==2.4.6
torchcrepe==0.0.24
torchfcpe==0.0.4
faiss-cpu==1.14.3
librosa==0.11.0
soundfile==0.14.0
scipy==1.18.0
soxr==1.1.0

# Voice conversion dependencies
transformers==5.13.1
einops==0.8.2
praat-parselmouth
pyworld

# Source separation
onnxruntime>=1.17.0
audio-separator[gpu]

# Audio post-processing
pedalboard
pydub
ffmpeg-python

# Utilities
gradio==6.20.0
yt_dlp
gdown
requests
tqdm
stftpitchshift==2.0

# NOTE: torch==2.11.0+cu130 is pre-installed in Colab — do not reinstall
```

> **Colab install note**: torch/torchaudio are pre-installed. Use `pip install -r requirements.txt` and skip torch re-installation to avoid breaking the CUDA 13.0 build.

---

## 4. Structural Rework Plan

### 4.1 New Directory Layout

```
AICoverGen-Mod/
├── src/
│   ├── rvc/                          # NEW — ported from Applio rvc/
│   │   ├── __init__.py
│   │   ├── configs/
│   │   │   ├── config.py             # Applio Config class (simplified)
│   │   │   ├── 32000.json
│   │   │   ├── 40000.json
│   │   │   ├── 48000.json
│   │   │   └── 24000.json
│   │   ├── lib/
│   │   │   ├── utils.py              # HubertModelWithFinalProj + load_embedding + load_audio
│   │   │   ├── algorithm/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── synthesizers.py   # unified Synthesizer (replaces 4 old model classes)
│   │   │   │   ├── attentions.py
│   │   │   │   ├── commons.py
│   │   │   │   ├── modules.py
│   │   │   │   ├── residuals.py
│   │   │   │   ├── encoders.py
│   │   │   │   └── generators/
│   │   │   └── predictors/
│   │   │       ├── RMVPE.py          # self-contained (no fairseq)
│   │   │       ├── FCPE.py           # optional
│   │   │       └── f0.py
│   │   └── infer/
│   │       ├── pipeline.py           # voice_conversion + get_f0 (Applio version)
│   │       └── infer.py              # VoiceConverter class (simplified for Colab)
│   ├── mdx.py                        # keep, minor updates
│   ├── main.py                       # rework imports + sox removal
│   ├── my_utils.py                   # rewrite with librosa+soxr
│   ├── download_models.py            # update model URLs
│   └── webui.py                      # rewrite for Gradio 6.x
├── mdxnet_models/
│   └── model_data.json
├── rvc_models/
│   ├── embedders/
│   │   └── contentvec/               # NEW — pytorch_model.bin + config.json
│   ├── predictors/
│   │   └── rmvpe.pt                  # F0 model weight
│   └── <voice-model-folders>/        # user's RVC .pth + .index files
├── requirements.txt
├── no_ui.py
└── PLAN.md
```

> **Important**: The `src/infer_pack/` directory (old RVC model classes) will be **fully replaced** by `src/rvc/lib/algorithm/` and `src/rvc/lib/predictors/`. Do not keep the old files alongside the new ones to avoid import confusion. Delete `src/infer_pack/`, `src/rmvpe.py`, and `src/vc_infer_pipeline.py` after new rvc/ is working.

---

### 4.2 Phase-by-Phase Implementation

---

#### Phase 1 — RVC Core Replacement (Highest Priority)

**Goal**: Replace `fairseq`-based HuBERT loading and old `infer_pack` model classes with Applio's modern equivalents.

**Steps**:

1. **Copy Applio's `rvc/` core** into `src/rvc/` (stripping training-only code):
   - `rvc/lib/algorithm/` → `src/rvc/lib/algorithm/` (all files including generators/)
   - `rvc/lib/predictors/RMVPE.py` → `src/rvc/lib/predictors/RMVPE.py`
   - `rvc/lib/predictors/f0.py` → `src/rvc/lib/predictors/f0.py`
   - `rvc/lib/predictors/FCPE.py` → `src/rvc/lib/predictors/FCPE.py` (optional)
   - `rvc/infer/pipeline.py` → `src/rvc/infer/pipeline.py`
   - `rvc/configs/*.json` → `src/rvc/configs/*.json`
   - `rvc/configs/config.py` → `src/rvc/configs/config.py` (simplify, remove training params)

2. **Create `src/rvc/lib/utils.py`** — port Applio's `HubertModelWithFinalProj` + `load_embedding()` function. This is the core replacement for `fairseq`:
   - `HubertModelWithFinalProj` extends `transformers.HubertModel` with a `final_proj` linear layer
   - `load_embedding(embedder_model, custom_embedder)` downloads and loads the model from HuggingFace if not cached

3. **Rewrite `src/rvc.py`** to use the new classes:

   ```python
   # Before (broken):
   from fairseq import checkpoint_utils
   from infer_pack.models import SynthesizerTrnMs256NSFsid, ...

   # After (working):
   from rvc.lib.utils import load_embedding
   from rvc.lib.algorithm.synthesizers import Synthesizer
   from rvc.infer.pipeline import Pipeline as VC
   from rvc.configs.config import Config
   ```

4. **Rewrite `load_hubert()`** in `src/rvc.py`:
   ```python
   def load_hubert(device, is_half, embedder_model="contentvec", custom_path=None):
       from rvc.lib.utils import load_embedding
       model = load_embedding(embedder_model, custom_path)
       model = model.to(device).float().eval()
       return model
   ```

5. **Rewrite `get_vc()`** in `src/rvc.py` to use Applio's unified `Synthesizer`:
   ```python
   def get_vc(device, model_path):
       cpt = torch.load(model_path, map_location="cpu", weights_only=True)
       tgt_sr = cpt["config"][-1]
       cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
       use_f0 = cpt.get("f0", 1)
       version = cpt.get("version", "v1")
       text_enc_hidden_dim = 768 if version == "v2" else 256
       vocoder = cpt.get("vocoder", "HiFi-GAN")
       net_g = Synthesizer(
           *cpt["config"],
           use_f0=use_f0,
           text_enc_hidden_dim=text_enc_hidden_dim,
           vocoder=vocoder,
       )
       del net_g.enc_q
       net_g.load_state_dict(cpt["weight"], strict=False)
       net_g = net_g.to(device).float().eval()
       config = Config()
       vc = VC(tgt_sr, config)
       return cpt, version, net_g, tgt_sr, use_f0, vc
   ```

6. **Rewrite `rvc_infer()`** to use Applio's `Pipeline.pipeline()` signature:
   - Old signature: `vc.pipeline(hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version, protect, crepe_hop_length)`
   - New signature: `vc.pipeline(model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, volume_envelope, version, protect, ...)`
   - Update `src/main.py`'s `voice_change()` function to pass arguments in the new order

7. **Delete old files** after verification:
   - `src/infer_pack/` (entire directory)
   - `src/rmvpe.py`
   - `src/vc_infer_pipeline.py`
   - `src/trainset_preprocess_pipeline_print.py`

---

#### Phase 2 — Audio Utilities Modernization

**Goal**: Remove broken dependencies (`sox`, old `ffmpeg-python` pattern) and modernize audio loading.

**Steps**:

1. **Rewrite `src/my_utils.py`** — use `librosa` + `soxr`:
   ```python
   import librosa
   import soxr
   import numpy as np

   def load_audio(file, sr):
       file = file.strip().strip('"').strip('\n').strip('"').strip()
       try:
           audio, orig_sr = librosa.load(file, sr=None, mono=True)
           if orig_sr != sr:
               audio = soxr.resample(audio, orig_sr, sr, quality='VHQ')
       except Exception as e:
           raise RuntimeError(f"Failed to load audio: {e}")
       return audio
   ```

2. **Replace `sox` pitch-shift in `src/main.py`**:
   ```python
   # Before (broken):
   import sox
   tfm = sox.Transformer()
   tfm.pitch(pitch_change)
   y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)

   # After (ffmpeg-based, no extra dependency):
   def pitch_shift(audio_path, pitch_change):
       output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
       if not os.path.exists(output_path):
           semitone_rate = 44100 * (2 ** (pitch_change / 12))
           cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', audio_path,
                  '-af', f'asetrate={semitone_rate:.2f},aresample=44100',
                  output_path]
           subprocess.run(cmd, check=True)
       return output_path
   ```

3. **Update `src/mdx.py`** — fix ONNX runtime provider selection:
   ```python
   # Current (may fail if onnxruntime-gpu not installed):
   self.provider = ['CUDAExecutionProvider'] if processor >= 0 else ['CPUExecutionProvider']

   # Fix: use available providers with fallback
   import onnxruntime as ort
   available = ort.get_available_providers()
   self.provider = (['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available and processor >= 0
                    else ['CPUExecutionProvider'])
   ```

4. **Remove `import sox`** from `src/main.py` and update `pitch_shift()`.

---

#### Phase 3 — Gradio 6.x UI Update

**Goal**: Fix all Gradio 3.x API calls in `src/webui.py` that are broken in Gradio 6.x.

**Breaking changes to fix**:

| Old (Gradio 3.x) | New (Gradio 6.x) |
|---|---|
| `gr.DataFrame.update(value=...)` | `gr.update(value=...)` |
| `gr.CheckboxGroup.update(choices=...)` | `gr.update(choices=...)` |
| `gr.Text.update(value=...)` | `gr.update(value=...)` |
| `gr.Dropdown(choices=...)` as return | `gr.update(choices=...)` |

**Steps**:

1. Audit all `.update()` calls in `webui.py` — use `gr.update(...)` everywhere.
2. Fix `load_public_models()` return: `gr.DataFrame.update(value=models_table)` → `gr.update(value=models_table)`.
3. Fix `pub_dl_autofill()` return: `gr.Text.update(value=...)` → `gr.update(value=...)`.
4. Fix `update_models_list()` return: `gr.Dropdown(choices=models_l)` → `gr.update(choices=models_l)`.
5. Fix `filter_models()` return similarly.
6. Verify `gr.ClearButton` behavior (API unchanged in 6.x).
7. Test Gradio interface launches cleanly with `--share` flag for Colab tunnel.

> **Note**: Since the primary use is Colab no-UI mode via `no_ui.py`, this phase is lower priority. The `no_ui.py → src/main.py` path does NOT use Gradio (except `gr.Progress` which can be removed or made optional).

---

#### Phase 4 — Model Download Modernization

**Goal**: Update download scripts to fetch the correct model files for the new Applio-style embedder format.

**Key change**: Applio uses contentvec as a HuggingFace model directory (`pytorch_model.bin` + `config.json`) instead of the single fairseq `hubert_base.pt` checkpoint.

**Steps**:

1. **Update `src/download_models.py`** to download contentvec as a directory:
   ```python
   APPLIO_BASE = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
   CONTENTVEC_BIN = f"{APPLIO_BASE}/embedders/contentvec/pytorch_model.bin"
   CONTENTVEC_CFG = f"{APPLIO_BASE}/embedders/contentvec/config.json"
   RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
   ```

2. Create `rvc_models/embedders/contentvec/` directory and download both files there.

3. **Add Roformer model download** — `model_bs_roformer_ep_317_sdr_12.9755.ckpt` is required by `run_roformer()` in `mdx.py`. Download to `mdxnet_models/` directory.
   - Source: audio-separator's model hub or direct URL

4. Update `rvc.py`'s `load_hubert()` call in `main.py`'s `voice_change()` to point to the contentvec directory instead of `hubert_base.pt`.

---

#### Phase 5 — `no_ui.py` Colab Cell Refinement

**Goal**: Make `no_ui.py` fully functional as standalone Colab cells.

**Steps**:

1. Keep the `# @param` comment style for Colab form cells.
2. Add `sys.path` setup at top:
   ```python
   import sys, os
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
   ```
3. Remove the interactive `input()` call for SONG_INPUT — replace with a `# @param` form field.
4. Update all argument names in the subprocess command to match updated `src/main.py` argparse arguments.
5. Optionally add inline progress output instead of subprocess capture for better Colab display.
6. Add a simple `gr.Progress`-free import guard in `main.py` for when called from `no_ui.py` (non-webui context).

---

## 5. File-by-File Change Summary

| File | Action | Notes |
|---|---|---|
| `requirements.txt` | **Full rewrite** | Remove fairseq/deemix/sox/lib, pin modern versions |
| `src/rvc.py` | **Full rewrite** | Use Applio's Synthesizer + pipeline |
| `src/my_utils.py` | **Full rewrite** | librosa + soxr instead of ffmpeg subprocess |
| `src/main.py` | **Partial rewrite** | Remove sox import, fix pitch_shift(), update rvc imports |
| `src/webui.py` | **Partial rewrite** | Gradio 6.x API fixes throughout |
| `src/download_models.py` | **Update** | New model URLs, contentvec directory structure |
| `src/infer_pack/` | **Delete entirely** | Replaced by `src/rvc/lib/algorithm/` |
| `src/rmvpe.py` | **Delete** | Replaced by `src/rvc/lib/predictors/RMVPE.py` |
| `src/vc_infer_pipeline.py` | **Delete** | Replaced by `src/rvc/infer/pipeline.py` |
| `src/trainset_preprocess_pipeline_print.py` | **Delete** | Training utility, not needed |
| `src/rvc/` (new) | **Create** | Port from Applio rvc/ (infer only) |
| `no_ui.py` | **Minor update** | sys.path + Colab cell compatibility |
| `mdxnet_models/model_data.json` | **Keep** | Already correct format |
| `src/mdx.py` | **Keep + minor fix** | ONNX provider fallback |
| `src/configs/` → `src/rvc/configs/` | **Rename + replace** | New naming convention (48000.json not 48k.json) |

---

## 6. Key Architecture Differences: Old vs New

### HuBERT Embedder Loading

```python
# OLD (fairseq — will not install on Python 3.13):
from fairseq import checkpoint_utils
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path], suffix=''
)
hubert = models[0].to(device)
if is_half: hubert = hubert.half()
hubert.eval()

# Usage in VC pipeline:
inputs = {"source": feats, "padding_mask": padding_mask, "output_layer": 9 if v1 else 12}
logits = model.extract_features(**inputs)
feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

# NEW (transformers — works on Python 3.13):
from transformers import HubertModel
import torch.nn as nn

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

model = HubertModelWithFinalProj.from_pretrained(contentvec_dir)
model.to(device).float().eval()

# Usage in VC pipeline:
feats = model(audio_tensor)["last_hidden_state"]
feats = model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
```

### Synthesizer (Voice Model) Loading

```python
# OLD (4 separate classes, must branch on version + f0):
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,      # v1 + f0
    SynthesizerTrnMs256NSFsid_nono, # v1, no f0
    SynthesizerTrnMs768NSFsid,      # v2 + f0
    SynthesizerTrnMs768NSFsid_nono, # v2, no f0
)
if version == "v1":
    if if_f0 == 1: net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
    else: net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
elif version == "v2":
    ...

# NEW (one unified class handles all cases):
from rvc.lib.algorithm.synthesizers import Synthesizer
net_g = Synthesizer(
    *cpt["config"],
    use_f0=cpt.get("f0", 1),
    text_enc_hidden_dim=768 if version == "v2" else 256,
    vocoder=cpt.get("vocoder", "HiFi-GAN"),
)
```

### F0 / RMVPE Loading

```python
# OLD (loads standalone rmvpe.py, requires manual model path):
from rmvpe import RMVPE
self.model_rmvpe = RMVPE(
    os.path.join(BASE_DIR, 'rvc_models', 'rmvpe.pt'),
    is_half=self.is_half,
    device=self.device
)
f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

# NEW (Applio's cleaner RMVPE class, auto-managed in Pipeline):
from rvc.lib.predictors.RMVPE import RMVPE
# Loaded lazily inside Pipeline.get_f0():
if "rmvpe" not in self.f0_models:
    self.f0_models["rmvpe"] = RMVPE(device=pitch_device, sample_rate=16000, hop_size=160)
f0 = self.f0_models["rmvpe"].get_f0(x, filter_radius=0.03)
```

---

## 7. Colab-Specific Notes

- **CUDA 13.0**: PyTorch 2.11.0+cu130 is already installed; avoid reinstalling torch/torchaudio via pip (use conditional install or `--no-deps` for packages that would pull conflicting versions).
- **T4 GPU (16 GB VRAM)**: Config `x_pad=3, x_query=10, x_center=60, x_max=65` (the 6GB+ half-precision config) works fine. T4 supports FP16.
- **Half precision**: Applio's new pipeline loads hubert as `.float()` but net_g can be `.half()` for speed. Keep `is_half=True` as default but set hubert to float to avoid transformer model casting issues.
- **`audio-separator`**: Uses `onnxruntime-gpu` under the hood when CUDA is available. No extra ONNX config needed.
- **Drive mounting**: `no_ui.py` accepts full paths — works with `/content/drive/MyDrive/...` paths as-is.
- **Model path change**: The old code loads `rvc_models/hubert_base.pt` (fairseq format). The new code loads `rvc_models/embedders/contentvec/` (HuggingFace format). Update any references.
- **`weights_only=True`** in `torch.load()` — required for PyTorch 2.x security; already used by Applio. The old `get_vc()` uses plain `torch.load(model_path, map_location='cpu')` which will raise a warning on 2.11. Fix it.

---

## 8. Implementation Order

```
After each phase completed, ask user to continue to next phase

Phase 1: RVC Core (src/rvc/ + rvc.py rewrite)           ← DO FIRST (✅ Completed, next to phase 2)
Phase 2: Audio utilities (my_utils.py, pitch_shift)      ← DO SECOND (✅ Completed, next to phase 4)
Phase 4: Model downloader (download_models.py)           ← DO THIRD (✅ Completed, next to phase 5)
Phase 5: no_ui.py cleanup                                ← DO FOURTH (✅ Completed, next to phase 3)
Phase 3: Gradio WebUI (webui.py)                         ← DO LAST (✅ Completed)
```

---

## 9. Testing Checkpoints

After each phase, verify:

1. **Phase 1**: `python -c "import sys; sys.path.insert(0,'src'); from rvc.lib.utils import load_embedding; print('OK')"` — no ImportError.
2. **Phase 1**: `python src/rvc.py` can import cleanly; `get_vc()` loads a sample `.pth` model without crash.
3. **Phase 2**: `python -c "import sys; sys.path.insert(0,'src'); from my_utils import load_audio; load_audio('test.wav', 16000)"` loads a WAV file.
4. **Phase 2**: `pitch_shift()` in `main.py` produces output WAV without sox dependency.
5. **Phase 4**: `python src/download_models.py` downloads all required model files to correct paths.
6. **Phase 5**: Running `no_ui.py` cells in Colab with a local WAV file runs end-to-end and produces an AI cover in `song_output/`.
7. **Full pipeline**: YouTube URL → AI cover MP3 output.

## 10. Rule

- This project usage for google colab only, never run in this device
- If you need to take a peek at Applio repo, you don't need to check from Applio github anymore, there's already Applio at here (there's a graphify too) /workspaces/codespaces-blank/Applio
- Do not use subagent.
