"""Rejeu exact du service (aucun fichier existant modifié) : mêmes réglages, même ordre, même contrôle.

Question : les hauteurs mesurées en service (journal du 19/11) se retrouvent-elles hors service ?
- Si oui (mêmes Hz) : la génération est déterministe, c'est le texte + la graine qui décident → pas d'état caché.
- Si non : le processus du service est dans un état différent (bug d'état, mémoire, autre chose chargé).
Usage : python labo2/test_rejeu.py pierre|fictif [xvec]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
QUI = sys.argv[1]
XVEC = len(sys.argv) > 2 and sys.argv[2] == "xvec"
if QUI == "pierre":
    REF = str(RACINE / "models" / "ref_pierre.wav")
    TEXTE = ("Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
             "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
             "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
             "discussion et ce sera bien.")
    HEURES = ["09:35:49", "09:41:43", "10:32:31", "10:32:47", "10:49:15"]  # session « Pierre » du service
    REF_H = 123
else:
    F = RACINE / "personas" / "pisciniste-fictif"
    REF, TEXTE = str(F / "voix.wav"), json.loads((F / "fiche.json").read_text(encoding="utf-8"))["texte_reference"]
    HEURES = ["09:28:01", "09:28:31", "09:29:10", "09:29:43", "09:30:02", "09:30:35"]  # session « fictif »
    REF_H = 194
BORNES = (0.73 * REF_H, 1.34 * REF_H)

PHRASES = []
for ligne in (RACINE / "labo" / "journaux" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines():
    d = json.loads(ligne)
    if d.get("date", "")[-8:] in HEURES:
        PHRASES += [p.strip() for p in re.findall(r".+?[.!?…](?=\s|$)", d["reponse"]) if p.strip()]


def hauteur_mediane(a, sr=24000):  # copie exacte de labo/serveur.py
    v = []
    for i in range(0, len(a) - 2048, sr // 10):
        x = a[i:i + 2048] - a[i:i + 2048].mean()
        if np.sqrt((x ** 2).mean()) < 0.01:
            continue
        c = np.correlate(x, x, "full")[len(x) - 1:]
        lo, hi = sr // 400, sr // 60
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.3 * c[0]:
            v.append(sr / k)
    return float(np.median(v)) if v else 0.0


tts = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / "qwen3-tts-1.7B"), device="cuda:0")
list(tts.generate_voice_clone_streaming(text="Bonjour.", language="French", ref_audio=REF, ref_text=TEXTE,
                                        chunk_size=4))  # préchauffage identique au service (0,9 / 50)
import soundfile as sf  # noqa: E402

SORTIES, n = ICI / "rejeu", 0  # phrase retenue (celle que le service joue) : labo2/rejeu/<qui>[_xvec]_<n>.wav
for phrase in PHRASES:
    res = []
    for essai in range(3):
        torch.manual_seed(1234 + essai)
        flux = tts.generate_voice_clone_streaming(text=phrase, language="French", ref_audio=REF, ref_text=TEXTE,
                                                  chunk_size=4, temperature=0.6, top_k=30, xvec_only=XVEC)
        tete, duree = [], 0.0
        for m, sr, _ in flux:
            m = np.asarray(m, np.float32).reshape(-1)
            tete.append(m)
            duree += len(m) / 24000
            if duree >= 1.2:
                break
        h = hauteur_mediane(np.concatenate(tete))
        bon = h == 0.0 or BORNES[0] <= h <= BORNES[1]
        res.append(f"{h:.0f}{'' if bon else '✗'}")
        if bon or essai == 2:
            for m, sr, _ in flux:  # le service joue la suite
                tete.append(np.asarray(m, np.float32).reshape(-1))
            break
        flux.close()
    SORTIES.mkdir(exist_ok=True)
    n += 1
    sf.write(SORTIES / f"{QUI}{'_xvec' if XVEC else ''}_{n:02d}.wav", np.concatenate(tete), 24000)
    print(QUI, " / ".join(res), "«", phrase[:60], "»", flush=True)
