"""Test : la référence de voix est-elle vraiment prise en compte ?

Mêmes 3 phrases, mêmes réglages que le serveur (flux, morceaux de 4, température 0,6, top_k 30, graine 1234),
avec 3 références :
  A — models/ref_pierre.wav (référence actuelle de Pierre)
  B — personas/pisciniste-fictif/voix.wav (voix d'une autre personne, fictive)
  C — copie de la référence de Pierre remise à bon niveau (crête 0,9), écrite dans labo2/
Sorties : labo2/sorties/<A|B|C>_<n>.wav. Aucun fichier existant n'est modifié.
Usage (sur le pod) : CUDA_VISIBLE_DEVICES=1 .venv/bin/python labo2/test_references.py
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

RACINE = Path(__file__).resolve().parent.parent
ICI = Path(__file__).resolve().parent
SORTIES = ICI / "sorties"
SORTIES.mkdir(exist_ok=True)

TEXTE_PIERRE = ("Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
                "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
                "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
                "discussion et ce sera bien.")
PHRASES = [
    "Une eau verte après un chlore choc, ça veut souvent dire que le pH est trop haut.",
    "Pouvez-vous me dire ce qu'affiche votre sonde iopool en ce moment ?",
    "D'accord, je regarde ça avec vous.",
]

a, sr = sf.read(RACINE / "models" / "ref_pierre.wav", dtype="float32")
ref_c = ICI / "ref_pierre_niveau.wav"
sf.write(ref_c, a * (0.9 / np.abs(a).max()), sr)
fiche_b = json.loads((RACINE / "personas" / "pisciniste-fictif" / "fiche.json").read_text(encoding="utf-8"))
REFS = {
    "A": (str(RACINE / "models" / "ref_pierre.wav"), TEXTE_PIERRE),
    "B": (str(RACINE / "personas" / "pisciniste-fictif" / "voix.wav"), fiche_b["texte_reference"]),
    "C": (str(ref_c), TEXTE_PIERRE),
}

tts = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / "qwen3-tts-1.7B"), device="cuda:0")
for cle, (ref, texte_ref) in REFS.items():
    for n, phrase in enumerate(PHRASES, 1):
        torch.manual_seed(1234)
        morceaux, taux = [], 24000
        for m, taux, _ in tts.generate_voice_clone_streaming(text=phrase, language="French", ref_audio=ref,
                                                             ref_text=texte_ref, chunk_size=4,
                                                             temperature=0.6, top_k=30):
            morceaux.append(np.asarray(m, np.float32).reshape(-1))
        sortie = SORTIES / f"{cle}_{n}.wav"
        sf.write(sortie, np.concatenate(morceaux), taux)
        print(sortie.name, f"{sum(len(x) for x in morceaux) / taux:.1f} s")
