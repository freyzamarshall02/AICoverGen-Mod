"""
AICoverGen-Mod simplified Config — stripped from Applio for inference-only use.
Supports CUDA (T4 / Colab), MPS, and CPU devices.
"""

import os
import sys
import json
from multiprocessing import cpu_count

import torch

# Resolve the directory containing this file so JSON configs can be loaded
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

version_config_paths = [
    os.path.join(_CONFIG_DIR, "48000.json"),
    os.path.join(_CONFIG_DIR, "40000.json"),
    os.path.join(_CONFIG_DIR, "32000.json"),
    os.path.join(_CONFIG_DIR, "24000.json"),
]


class Config:
    def __init__(self, device=None, is_half=True):
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.is_half = is_half
        self.n_cpu = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.json_config = self._load_config_json()
        self.x_pad, self.x_query, self.x_center, self.x_max = self._device_config()

    def _load_config_json(self):
        configs = {}
        for config_file in version_config_paths:
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    configs[os.path.basename(config_file)] = json.load(f)
        return configs

    def _device_config(self) -> tuple:
        if self.device.startswith("cuda"):
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            self.gpu_mem = (
                torch.cuda.get_device_properties(i_device).total_memory // (1024 ** 3)
            )
            # Force FP32 for older 10/16-series cards
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16/10-series or P40 detected — forcing single precision.")
                self.is_half = False
        elif self.device == "mps":
            pass  # MPS: use system memory, no special config
        else:
            self.device = "cpu"
            self.is_half = False

        # Memory-tiered padding/query config
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # 4 GB or less
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)
        elif self.device == "mps":
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)
        else:
            # 6 GB+: standard half-precision config
            x_pad, x_query, x_center, x_max = (3, 10, 60, 65)

        return x_pad, x_query, x_center, x_max


# Alias used by rvc/infer/infer.py
inference_config = Config
