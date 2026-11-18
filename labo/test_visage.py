"""Brique 4 — Visage (Ditto, PyTorch) : animation de la photo de Pierre sur un audio, test isolé + mesure.

Mode « online » (flux) : l'audio est envoyé par morceaux de 0,2 s, comme en conversation.
Mesure : délai avant la 1re image, images/s, temps total vs durée audio.
Usage : python labo/test_visage.py [audio.wav] [pas_diffusion]
"""
import math
import sys
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import cuda_win  # noqa: F401,E402  (DLL CUDA pour onnxruntime : HuBERT sur GPU)
import os  # noqa: E402

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # pas de pool de processus sous Windows

# Calcul TF32 sur les cœurs tenseurs de la 4090 (cudnn.benchmark : à éviter, tailles variables en flux → 11 s de latence).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RACINE = Path(__file__).resolve().parent.parent
DITTO = RACINE / "third_party" / "ditto"
sys.path.insert(0, str(DITTO))
from stream_pipeline_online import StreamSDK  # noqa: E402

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "models" / "sorties_tts" / "rapide_1.7B_phrase2.wav"
PAS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
SORTIE = RACINE / "models" / "sorties_visage"
SORTIE.mkdir(parents=True, exist_ok=True)

# Photo réduite (720×960) : suffisant pour un écran de téléphone, et bien plus rapide à recoller.
SOURCE = RACINE / "models" / "visage_pierre_960.jpg"
if not SOURCE.exists():
    img = cv2.imread(str(RACINE / "assets" / "visage_pierre.jpg"))
    h, w = img.shape[:2]
    cv2.imwrite(str(SOURCE), cv2.resize(img, (round(w * 960 / h), 960), interpolation=cv2.INTER_AREA))

t0 = time.perf_counter()
sdk = StreamSDK(str(RACINE / "models/ditto/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"),
                str(RACINE / "models/ditto/ditto_pytorch"))
print(f"Chargement des modèles : {time.perf_counter() - t0:.1f} s")

# torch.compile (triton-windows) sur déformation + décodage : 24,5 → 14 ms par image.
# 1re compilation ~50 s, ensuite en cache disque.
if "--sans-compile" not in sys.argv:
    t0 = time.perf_counter()
    sdk.warp_f3d.warp_net.model.compile()
    sdk.decode_f3d.decoder.model.compile()
    # Compilation dans le fil principal (dans les fils de Ditto elle se bloque) avec des entrées factices.
    f3d = np.zeros((1, 32, 16, 64, 64), dtype=np.float32)
    kp = np.zeros((1, 21, 3), dtype=np.float32)
    sdk.decode_f3d(sdk.warp_f3d(f3d, kp, kp))
    print(f"torch.compile : {time.perf_counter() - t0:.1f} s")


def animer(audio_path, sortie, pas):
    t0 = time.perf_counter()
    # Réglages du mode flux repris de v0.4_hubert_cfg_trt_online.pkl (overlap_v2=70 → 10 images par étape).
    sdk.setup(str(SOURCE), str(sortie), online_mode=True, overlap_v2=70, sampling_timesteps=pas)
    t_prep = time.perf_counter() - t0

    # Instrumentation : horodatage de chaque image produite.
    horodatages = []
    ecrire = sdk.writer

    class Espion:
        def __call__(self, img, fmt="rgb"):
            horodatages.append(time.perf_counter())
            ecrire(img, fmt=fmt)

        def close(self):
            ecrire.close()

    sdk.writer = Espion()

    audio, _ = librosa.load(str(audio_path), sr=16000)
    n_images = math.ceil(len(audio) / 16000 * 25)
    sdk.setup_Nd(N_d=n_images)
    chunksize = (3, 5, 2)
    audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio])
    taille = int(sum(chunksize) * 0.04 * 16000) + 80
    t_debut = time.perf_counter()
    for i in range(0, len(audio), chunksize[1] * 640):
        morceau = audio[i:i + taille]
        if len(morceau) < taille:
            morceau = np.pad(morceau, (0, taille - len(morceau)))
        sdk.run_chunk(morceau, chunksize)
    sdk.close()
    t_fin = time.perf_counter()
    duree_audio = n_images / 25
    return {
        "preparation_photo": t_prep,
        "premiere_image": horodatages[0] - t_debut if horodatages else float("nan"),
        "images": len(horodatages),
        "ips": len(horodatages) / (t_fin - t_debut),
        "total": t_fin - t_debut,
        "duree_audio": duree_audio,
    }


animer(AUDIO, SORTIE / "prechauffage.mp4", PAS)  # préchauffage

torch.cuda.reset_peak_memory_stats()
for i in range(2):
    r = animer(AUDIO, SORTIE / f"visage_pas{PAS}_{i}.mp4", PAS)
    print(f"Essai {i + 1} (pas={PAS}) : préparation photo {r['preparation_photo']:.2f} s | "
          f"1re image {r['premiere_image'] * 1000:.0f} ms | {r['images']} images en {r['total']:.2f} s "
          f"pour {r['duree_audio']:.1f} s d'audio → {r['ips']:.1f} images/s (25 nécessaires)")
print(f"VRAM max : {torch.cuda.max_memory_allocated() / 1e9:.1f} Go")

if "--profil" in sys.argv:
    # Temps cumulé par étage (les synchronisations GPU ralentissent un peu l'ensemble).
    from collections import defaultdict

    PROFIL = defaultdict(lambda: [0.0, 0])

    class Chrono:
        def __init__(self, nom, composant):
            self._nom, self._c = nom, composant

        def __getattr__(self, attr):
            return getattr(self._c, attr)

        def __call__(self, *a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            res = self._c(*a, **k)
            torch.cuda.synchronize()
            PROFIL[self._nom][0] += time.perf_counter() - t
            PROFIL[self._nom][1] += 1
            return res

    for nom in ("wav2feat", "audio2motion", "motion_stitch", "warp_f3d", "decode_f3d", "putback"):
        setattr(sdk, nom, Chrono(nom, getattr(sdk, nom)))
    r = animer(AUDIO, SORTIE / "profil.mp4", PAS)
    print("Profil par étage (temps cumulé / appels) :")
    for nom, (t, n) in PROFIL.items():
        print(f"  {nom:14s} {t * 1000:7.0f} ms / {n:3d} appels = {t / max(n, 1) * 1000:6.1f} ms par appel")
