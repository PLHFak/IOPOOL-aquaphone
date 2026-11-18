"""Brique 3 — Voix clonée FR en flux (faster-qwen3-tts, graphes CUDA) : test isolé + mesure.

Mesure le délai avant le premier morceau audio (TTFA) et le facteur temps réel.
Usage : python labo/test_tts_rapide.py [1.7B|0.6B] [chunk_size]
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

RACINE = Path(__file__).resolve().parent.parent
TAILLE = sys.argv[1] if len(sys.argv) > 1 else "1.7B"
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SORTIE = RACINE / "models" / "sorties_tts"
SORTIE.mkdir(parents=True, exist_ok=True)

REF_AUDIO = str(RACINE / "models" / "ref_pierre.wav")
REF_TEXTE = (
    "Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
    "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
    "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
    "discussion et ce sera bien."
)
PHRASES = [
    "Bonjour, je suis l'assistant IA de Marc, votre pisciniste.",
    "Une eau verte malgré le chlore choc, c'est souvent un pH trop élevé.",
    "Pouvez-vous me donner la valeur de votre pH, ou m'envoyer une photo de la bandelette ?",
]

t0 = time.perf_counter()
modele = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / f"qwen3-tts-{TAILLE}"))
print(f"Chargement : {time.perf_counter() - t0:.1f} s")


def dire(phrase):
    t = time.perf_counter()
    premier, morceaux, sr = None, [], 24000
    for audio, sr, _ in modele.generate_voice_clone_streaming(
        text=phrase, language="French", ref_audio=REF_AUDIO, ref_text=REF_TEXTE, chunk_size=CHUNK
    ):
        if premier is None:
            premier = time.perf_counter() - t
        morceaux.append(audio)
    total = time.perf_counter() - t
    return premier, total, np.concatenate(morceaux), sr


t0 = time.perf_counter()
dire("Bonjour.")  # préchauffage : capture des graphes CUDA + préparation de la voix
print(f"Préchauffage (une fois) : {time.perf_counter() - t0:.1f} s")

for i, phrase in enumerate(PHRASES, 1):
    premier, total, audio, sr = dire(phrase)
    duree_audio = len(audio) / sr
    sf.write(SORTIE / f"rapide_{TAILLE}_phrase{i}.wav", audio, sr)
    print(f"Phrase {i} : 1er son {premier * 1000:.0f} ms | total {total * 1000:.0f} ms "
          f"pour {duree_audio:.1f} s d'audio (facteur temps réel {total / duree_audio:.2f})")

print(f"VRAM max : {torch.cuda.max_memory_allocated() / 1e9:.1f} Go")
