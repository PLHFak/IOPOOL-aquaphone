"""Test : contrôle de chaque phrase + nouvel essai si la voix s'éloigne de la référence (aucun fichier existant modifié).

Pour chacune des 20 phrases de test_service.py : 4 essais (graines différentes), réglages du serveur actuel.
Chaque essai est noté :
  - empreinte vocale : similarité cosinus entre l'empreinte du locuteur (encodeur du modèle Qwen3-TTS lui-même)
    de l'essai et celle de la référence de Pierre ;
  - hauteur médiane (Hz).
Sorties : labo2/controle/essai<k>_<n>.wav, labo2/controle/choisi_<n>.wav, labo2/controle/notes.json
« essai1 » = ce que le service joue aujourd'hui ; « choisi » = ce que jouerait le service avec contrôle.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
src = (ICI / "test_service.py").read_text(encoding="utf-8")
debut = src.index("PHRASES = [") + len("PHRASES = ")
PHRASES = eval(src[debut:src.index("]\n", debut) + 1])
REF = str(RACINE / "models" / "ref_pierre.wav")
TEXTE_PIERRE = ("Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
                "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
                "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
                "discussion et ce sera bien.")
ESSAIS = 4


def hauteur(a, sr=24000):
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
coeur = tts.model.model


def empreinte(a):
    e = coeur.extract_speaker_embedding(audio=a.astype(np.float32), sr=24000).float().reshape(-1)
    return torch.nn.functional.normalize(e, dim=0)


ref, sr_ref = sf.read(REF, dtype="float32")
e_ref = empreinte(ref)

dossier = ICI / "controle"
dossier.mkdir(exist_ok=True)
notes = []
for n, phrase in enumerate(PHRASES, 1):
    essais = []
    for k in range(ESSAIS):
        torch.manual_seed(1234 + k)
        a = np.concatenate([np.asarray(m, np.float32).reshape(-1) for m, _, _ in tts.generate_voice_clone_streaming(
            text=phrase, language="French", ref_audio=REF, ref_text=TEXTE_PIERRE, chunk_size=4,
            temperature=0.6, top_k=30)])
        sim = float(torch.dot(empreinte(a), e_ref))
        essais.append({"essai": k + 1, "similarite": round(sim, 3), "hauteur": round(hauteur(a)), "duree": round(len(a) / 24000, 1)})
        sf.write(dossier / f"essai{k + 1}_{n:02d}.wav", a, 24000)
    meilleur = max(essais, key=lambda e: e["similarite"])
    (dossier / f"choisi_{n:02d}.wav").write_bytes((dossier / f"essai{meilleur['essai']}_{n:02d}.wav").read_bytes())
    notes.append({"n": n, "phrase": phrase, "essais": essais, "choisi": meilleur["essai"]})
    print(n, " | ".join(f"{e['similarite']:.3f}/{e['hauteur']}Hz" for e in essais), "→ essai", meilleur["essai"], flush=True)
(dossier / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
