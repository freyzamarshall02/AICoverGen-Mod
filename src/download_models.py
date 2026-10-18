"""
src/download_models.py  —  AICoverGen-Mod model downloader (Phase 4 rewrite)

Downloads all required model files to their correct locations:
  - MDX-Net ONNX models         → mdxnet_models/
  - Roformer model              → mdxnet_models/
  - contentvec embedder         → rvc_models/embedders/contentvec/
  - RMVPE F0 predictor          → rvc_models/predictors/
"""

import os
import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir   = BASE_DIR / "mdxnet_models"
rvc_models_dir      = BASE_DIR / "rvc_models"
embedders_dir       = rvc_models_dir / "embedders" / "contentvec"
predictors_dir      = rvc_models_dir / "predictors"

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
# Applio HuggingFace resource base
_APPLIO_BASE = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

# MDX-Net ONNX separation models
MDX_DOWNLOAD_LINK = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
MDX_MODELS = [
    "UVR-MDX-NET-Voc_FT.onnx",
    "UVR_MDXNET_KARA_2.onnx",
    "Reverb_HQ_By_FoxJoy.onnx",
    "Kim_Vocal_2.onnx",
]

# Roformer vocal separation model (used by run_roformer in mdx.py)
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_URL   = (
    "https://huggingface.co/Eddycrack864/Music-Source-Separation-Training/resolve/main/"
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
)

# contentvec HuBERT embedder (Applio-style: directory with two files)
CONTENTVEC_BIN_URL = f"{_APPLIO_BASE}/embedders/contentvec/pytorch_model.bin"
CONTENTVEC_CFG_URL = f"{_APPLIO_BASE}/embedders/contentvec/config.json"

# RMVPE F0 predictor
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dl(url: str, dest: Path, desc: str | None = None) -> None:
    """Download *url* to *dest*, skipping if the file already exists."""
    dest = Path(dest)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    print(f"  [dl]   {label}")
    print(f"         {url}")

    if _HAS_TQDM:
        pbar: list[tqdm] = [None]

        def _hook(block_num, block_size, total_size):
            if pbar[0] is None:
                pbar[0] = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {label}",
                    leave=False,
                )
            downloaded = block_num * block_size
            pbar[0].n = min(downloaded, total_size)
            pbar[0].refresh()

        urllib.request.urlretrieve(url, dest, reporthook=_hook)
        if pbar[0]:
            pbar[0].close()
    else:
        urllib.request.urlretrieve(url, dest)

    print(f"  [ok]   {dest.name}")


# ---------------------------------------------------------------------------
# Download routines
# ---------------------------------------------------------------------------

def download_mdx_models() -> None:
    print("\n=== MDX-Net ONNX models ===")
    mdxnet_models_dir.mkdir(parents=True, exist_ok=True)
    for model in MDX_MODELS:
        _dl(MDX_DOWNLOAD_LINK + model, mdxnet_models_dir / model)


def download_roformer_model() -> None:
    print("\n=== Roformer vocal separation model ===")
    mdxnet_models_dir.mkdir(parents=True, exist_ok=True)
    _dl(ROFORMER_URL, mdxnet_models_dir / ROFORMER_MODEL, desc=ROFORMER_MODEL)


def download_contentvec() -> None:
    """Download the contentvec embedder (pytorch_model.bin + config.json)."""
    print("\n=== contentvec HuBERT embedder ===")
    embedders_dir.mkdir(parents=True, exist_ok=True)
    _dl(CONTENTVEC_BIN_URL, embedders_dir / "pytorch_model.bin", desc="contentvec/pytorch_model.bin")
    _dl(CONTENTVEC_CFG_URL, embedders_dir / "config.json",       desc="contentvec/config.json")


def download_rmvpe() -> None:
    """Download the RMVPE F0 predictor weight."""
    print("\n=== RMVPE F0 predictor ===")
    predictors_dir.mkdir(parents=True, exist_ok=True)
    _dl(RMVPE_URL, predictors_dir / "rmvpe.pt", desc="rmvpe.pt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AICoverGen-Mod — downloading all required model files")
    print(f"Base directory : {BASE_DIR}")

    download_mdx_models()
    download_roformer_model()
    download_contentvec()
    download_rmvpe()

    print("\n[+] All models downloaded successfully!")
    print(f"    contentvec  : {embedders_dir}/")
    print(f"    rmvpe.pt    : {predictors_dir}/rmvpe.pt")
    print(f"    MDX/Roformer: {mdxnet_models_dir}/")
