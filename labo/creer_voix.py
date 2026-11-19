"""Crée la voix d'un personnage FICTIF à partir d'une description (Qwen3-TTS VoiceDesign, local).

Usage : python labo/creer_voix.py <persona> [nombre]   → personas/<persona>/candidats/voix_N.wav
La voix choisie devient ensuite l'enregistrement de référence du clonage (voix.wav + texte_reference),
comme l'enregistrement de Pierre : même chaîne de synthèse, même qualité.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

RACINE = Path(__file__).resolve().parent.parent
MODELE = RACINE / "models" / "qwen3-tts-voicedesign"


def main():
    persona = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dossier = RACINE / "personas" / persona
    fiche = json.loads((dossier / "fiche.json").read_text(encoding="utf-8"))
    sortie = dossier / "candidats"
    sortie.mkdir(parents=True, exist_ok=True)
    tts = FasterQwen3TTS.from_pretrained(str(MODELE), device="cuda:0")
    for i in range(n):
        torch.manual_seed(2000 + i)
        audios, sr = tts.generate_voice_design(text=fiche["texte_reference"], instruct=fiche["description_voix"],
                                               language="French")
        audio = np.asarray(audios[0], np.float32).reshape(-1)
        chemin = sortie / f"voix_{i + 1}.wav"
        sf.write(chemin, audio, sr)
        print(f"{chemin}  ({len(audio) / sr:.1f} s)")


if __name__ == "__main__":
    main()
