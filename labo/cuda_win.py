"""Rend visibles les DLL CUDA 12 (cuBLAS/cuDNN) fournies par PyTorch — nécessaire à CTranslate2 sous Windows."""
import os
from pathlib import Path

import torch

os.environ["PATH"] = str(Path(torch.__file__).parent / "lib") + os.pathsep + os.environ["PATH"]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
