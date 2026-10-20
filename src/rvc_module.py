"""
src/rvc.py  —  AICoverGen-Mod RVC wrapper (Phase 1 rewrite)

Replaces:
  - fairseq.checkpoint_utils  → transformers.HubertModel (via rvc.lib.utils)
  - infer_pack.models.*       → rvc.lib.algorithm.synthesizers.Synthesizer
  - vc_infer_pipeline.VC      → rvc.infer.pipeline.Pipeline
"""

from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from rvc.lib.utils import load_embedding
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.infer.pipeline import Pipeline as VC
from rvc.configs.config import Config
from my_utils import load_audio

BASE_DIR = Path(__file__).resolve().parent.parent


def load_hubert(device, is_half, embedder_model="contentvec", custom_embedder=None):
    """
    Load the HuBERT/contentvec embedder model.

    Args:
        device:          torch device string, e.g. "cuda:0"
        is_half:         unused (kept for API compatibility — model runs in float32)
        embedder_model:  name of the embedder, e.g. "contentvec"
        custom_embedder: path to a custom embedder directory (optional)

    Returns:
        HubertModelWithFinalProj instance, float32, eval mode, on `device`
    """
    model = load_embedding(embedder_model, custom_embedder)
    model = model.to(device).float().eval()
    return model


def get_vc(device, is_half, config, model_path):
    """
    Load an RVC voice model (.pth) and return all required inference objects.

    Args:
        device:     torch device string
        is_half:    whether to run net_g in half precision
        config:     Config instance (provides x_pad, x_query etc.)
        model_path: path to the .pth voice model file

    Returns:
        (cpt, version, net_g, tgt_sr, vc)
    """
    cpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(
            f"Incorrect format for {model_path}. "
            "Use a voice model trained with RVC v2."
        )

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
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g = net_g.to(device)

    # Always run net_g in float32 — matches pipeline.py's feats.float() call.
    # fp16 inference caused a dtype mismatch crash (float != c10::Half) on CUDA.
    # Applio takes the same approach: net_g.to(device).float() unconditionally.
    net_g = net_g.float()

    net_g.eval()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def rvc_infer(
    index_path,
    index_rate,
    input_path,
    output_path,
    pitch_change,
    f0_method,
    cpt,
    version,
    net_g,
    filter_radius,
    tgt_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
    vc,
    hubert_model,
):
    """
    Run the full RVC voice conversion pipeline on a single audio file.

    Args:
        index_path:       path to the FAISS .index file (empty string = no index)
        index_rate:       FAISS blending rate [0..1]
        input_path:       path to input WAV (vocals, 16 kHz mono)
        output_path:      path to write the output WAV
        pitch_change:     semitones to shift pitch (int)
        f0_method:        F0 estimator name: "rmvpe", "crepe", "fcpe"
        cpt:              checkpoint dict (returned by get_vc)
        version:          "v1" or "v2"
        net_g:            Synthesizer model (returned by get_vc)
        filter_radius:    RMVPE threshold / filter radius
        tgt_sr:           target sample rate (int)
        rms_mix_rate:     RMS mixing rate [0..1]  (0 = match input loudness)
        protect:          consonant protection level [0..0.5]
        crepe_hop_length: hop length for crepe (unused by rmvpe/fcpe but kept for API compat)
        vc:               Pipeline instance (returned by get_vc)
        hubert_model:     HubertModelWithFinalProj (returned by load_hubert)
    """
    audio = load_audio(input_path, 16000)
    pitch_guidance = bool(cpt.get("f0", 1))

    audio_opt = vc.pipeline(
        model=hubert_model,
        net_g=net_g,
        sid=0,
        audio=audio,
        pitch=pitch_change,
        f0_method=f0_method,
        file_index=index_path,
        index_rate=index_rate,
        pitch_guidance=pitch_guidance,
        volume_envelope=rms_mix_rate,
        version=version,
        protect=protect,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
    )

    # Normalize to int16 for wavfile.write
    audio_max = np.abs(audio_opt).max() / 0.99
    if audio_max > 1:
        audio_opt = audio_opt / audio_max
    audio_int16 = (audio_opt * 32767).astype(np.int16)
    wavfile.write(output_path, tgt_sr, audio_int16)
