"""Test : hasard divisé par 2 sur TOUS les étages de la voix (aucun fichier existant modifié).

Le serveur actuel : talker température 0,6 / top_k 30, mais code predictor (timbre, 15 codes sur 16) figé
à 0,9 / 50 dans son graphe CUDA. Ici : les deux étages à 0,45 / top_k 25 (≈ moitié de 0,9 / 50), le graphe
du code predictor est recapturé avec ces valeurs (en mémoire seulement, la bibliothèque n'est pas modifiée).
Mêmes 20 phrases que test_service.py → labo2/reponses_hasard_reduit/*.wav
"""
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
spec = importlib.util.spec_from_file_location("ts", ICI / "test_service.py")
src = (ICI / "test_service.py").read_text(encoding="utf-8")
PHRASES = eval(src[src.index("PHRASES = [") + len("PHRASES = "):src.index("]\n", src.index("PHRASES = [")) + 1])
TEXTE_PIERRE = ("Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
                "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
                "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
                "discussion et ce sera bien.")
TEMPERATURE, TOP_K = 0.45, 25

tts = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / "qwen3-tts-1.7B"), device="cuda:0")
pg = tts.predictor_graph
pg.temperature, pg.top_k = TEMPERATURE, TOP_K
pg.capture(num_warmup=3)  # graphe du timbre recapturé avec le hasard réduit
print("code predictor :", pg.temperature, pg.top_k)

dossier = ICI / "reponses_hasard_reduit"
dossier.mkdir(exist_ok=True)
for n, phrase in enumerate(PHRASES, 1):
    torch.manual_seed(1234)
    morceaux, taux = [], 24000
    for m, taux, _ in tts.generate_voice_clone_streaming(text=phrase, language="French",
                                                         ref_audio=str(RACINE / "models" / "ref_pierre.wav"),
                                                         ref_text=TEXTE_PIERRE, chunk_size=4,
                                                         temperature=TEMPERATURE, top_k=TOP_K):
        morceaux.append(np.asarray(m, np.float32).reshape(-1))
    sf.write(dossier / f"reponse_{n:02d}.wav", np.concatenate(morceaux), taux)
print("réponses générées :", len(PHRASES))
