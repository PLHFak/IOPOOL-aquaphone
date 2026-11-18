"""Brique 3 — Voix clonée FR (Qwen3-TTS Base) : test isolé + mesure.

Usage : python labo/test_tts.py [1.7B|0.6B]
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

RACINE = Path(__file__).resolve().parent.parent
TAILLE = sys.argv[1] if len(sys.argv) > 1 else "1.7B"
DEPOT = f"Qwen/Qwen3-TTS-12Hz-{TAILLE}-Base"
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
chemin = snapshot_download(DEPOT, local_dir=str(RACINE / "models" / f"qwen3-tts-{TAILLE}"))
print(f"Téléchargement/vérification : {time.perf_counter() - t0:.1f} s")

t0 = time.perf_counter()
modele = Qwen3TTSModel.from_pretrained(
    chemin, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
)
print(f"Chargement du modèle : {time.perf_counter() - t0:.1f} s")

t0 = time.perf_counter()
invite = modele.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXTE)
print(f"Préparation de la voix (une fois) : {time.perf_counter() - t0:.2f} s")

modele.generate_voice_clone(text="Bonjour.", language="French", voice_clone_prompt=invite)  # préchauffage

for i, phrase in enumerate(PHRASES, 1):
    torch.cuda.synchronize()
    t = time.perf_counter()
    wavs, sr = modele.generate_voice_clone(text=phrase, language="French", voice_clone_prompt=invite)
    torch.cuda.synchronize()
    duree = time.perf_counter() - t
    audio = len(wavs[0]) / sr
    sf.write(SORTIE / f"{TAILLE}_phrase{i}.wav", wavs[0], sr)
    print(f"Phrase {i} : {duree * 1000:.0f} ms de calcul pour {audio:.1f} s d'audio "
          f"(facteur temps réel {duree / audio:.2f})")

print(f"VRAM max : {torch.cuda.max_memory_allocated() / 1e9:.1f} Go")
