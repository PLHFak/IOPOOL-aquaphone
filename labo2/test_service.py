"""Test : d'où vient le changement de voix en service ? (aucun fichier existant modifié)

1. Extrait (lecture seule) l'audio des phrases d'attente en cache (models/attente/*.pkl) → labo2/attente/*.wav
2. Génère 20 phrases de réponse typiques avec la référence de Pierre, réglages du serveur → labo2/reponses/*.wav
Usage (sur le pod) : CUDA_VISIBLE_DEVICES=1 .venv/bin/python labo2/test_service.py
"""
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

RACINE = Path(__file__).resolve().parent.parent
ICI = Path(__file__).resolve().parent
TEXTE_PIERRE = ("Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
                "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
                "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
                "discussion et ce sera bien.")

# 1. Phrases d'attente en cache (paquets (pcm int16 40 ms, jpeg))
dossier = ICI / "attente"
dossier.mkdir(exist_ok=True)
for i, f in enumerate(sorted((RACINE / "models" / "attente").glob("*.pkl")), 1):
    paquets = pickle.loads(f.read_bytes())
    pcm = np.concatenate([np.frombuffer(p[0], np.int16) if isinstance(p[0], (bytes, bytearray))
                          else np.asarray(p[0]).reshape(-1) for p in paquets])
    pcm = pcm.astype(np.float32) / (32768.0 if pcm.dtype != np.float32 and np.abs(pcm).max() > 1.5 else 1.0)
    sf.write(dossier / f"attente_{i}_{f.stem[:6]}_{f.stat().st_mtime:.0f}.wav", pcm, 24000)
print("phrases d'attente extraites :", len(list(dossier.glob("*.wav"))))

# 2. Réponses typiques, comme le serveur (graine 1234, température 0,6, top_k 30, flux par 4)
PHRASES = [
    "Bonjour, je suis l'assistant de Marc, que se passe-t-il avec votre piscine ?",
    "D'accord, une eau verte depuis deux jours malgré le chlore choc, c'est fréquent.",
    "Le plus souvent, c'est le pH qui est trop haut et le chlore ne fait plus effet.",
    "Votre sonde iopool affiche quelle valeur de pH en ce moment ?",
    "Très bien.",
    "Alors on va commencer par faire baisser le pH.",
    "L'application iopool vous indique la bonne dose selon le volume de votre bassin.",
    "Laissez tourner la filtration en continu pendant vingt-quatre heures.",
    "Ensuite, brossez bien les parois et le fond.",
    "Si l'eau reste trouble après deux jours, rappelez-nous.",
    "Je peux aussi demander à Marc de passer vous voir.",
    "Est-ce que vous avez un robot de nettoyage ?",
    "Parfait, merci.",
    "Je comprends, ce n'est pas agréable en plein été.",
    "Vérifiez aussi que le filtre n'est pas encrassé.",
    "Un lavage du filtre à sable peut suffire.",
    "Vous avez bien fait de nous appeler.",
    "Je transmets tout ça à Marc, il vous rappelle dans la journée.",
    "Autre chose ?",
    "Bonne journée, et à bientôt !",
]
dossier = ICI / "reponses"
dossier.mkdir(exist_ok=True)
tts = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / "qwen3-tts-1.7B"), device="cuda:0")
for n, phrase in enumerate(PHRASES, 1):
    torch.manual_seed(1234)
    morceaux, taux = [], 24000
    for m, taux, _ in tts.generate_voice_clone_streaming(text=phrase, language="French",
                                                         ref_audio=str(RACINE / "models" / "ref_pierre.wav"),
                                                         ref_text=TEXTE_PIERRE, chunk_size=4,
                                                         temperature=0.6, top_k=30):
        morceaux.append(np.asarray(m, np.float32).reshape(-1))
    sf.write(dossier / f"reponse_{n:02d}.wav", np.concatenate(morceaux), taux)
print("réponses générées :", len(PHRASES))
