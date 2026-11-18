"""Brique 1 — Reconnaissance vocale FR (faster-whisper large-v3-turbo) : test isolé + mesure."""
import os
import sys
import time
from pathlib import Path

import torch  # fournit cublas/cudnn (CUDA 12) utilisés par CTranslate2

os.environ["PATH"] = str(Path(torch.__file__).parent / "lib") + os.pathsep + os.environ["PATH"]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from faster_whisper import WhisperModel  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "models" / "voix_pierre_16k.wav"

t0 = time.perf_counter()
modele = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",
    download_root=str(RACINE / "models" / "whisper"),
)
print(f"Chargement du modèle : {time.perf_counter() - t0:.2f} s")


def transcrire(chemin):
    t = time.perf_counter()
    segments, info = modele.transcribe(
        str(chemin), language="fr", beam_size=1, vad_filter=True
    )
    texte = " ".join(s.text.strip() for s in segments)
    return texte, time.perf_counter() - t, info.duration


transcrire(AUDIO)  # préchauffage (initialisation CUDA)
for i in range(3):
    texte, duree, audio = transcrire(AUDIO)
    print(f"Essai {i + 1} : {duree * 1000:.0f} ms pour {audio:.1f} s d'audio")
print("Texte :", texte)
