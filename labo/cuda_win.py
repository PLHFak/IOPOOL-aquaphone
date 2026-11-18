"""Rend visibles les DLL CUDA 12 (cuBLAS/cuDNN) fournies par PyTorch — à importer avant tout le reste.

- CTranslate2 (faster-whisper) les cherche dans le PATH.
- onnxruntime-gpu (HuBERT de Ditto) doit être une version compilée pour CUDA 12 (1.23.2 ; 1.24+ = CUDA 13)
  et les précharger, sinon il bascule en silence sur le processeur.
"""
import os
from pathlib import Path

import torch

os.environ["PATH"] = str(Path(torch.__file__).parent / "lib") + os.pathsep + os.environ["PATH"]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import onnxruntime

    onnxruntime.preload_dlls(directory="")
except ImportError:
    pass
