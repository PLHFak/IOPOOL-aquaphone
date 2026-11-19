"""Vérification « à tous les étages » de la dérive de voix (aucun fichier existant modifié).

Constat : avec la référence de Pierre, le service sort très souvent à 170-230 Hz (référence 123 Hz) alors que
le test d'hier (labo2/test_controle_voix.py) ne donnait qu'1 tirage sur 8. On rejoue ici les VRAIES phrases du
service (labo/journaux/pipeline.jsonl) en changeant un seul facteur à la fois.

Usage : python labo2/test_etages.py <variante>      (un processus par variante : les graphes CUDA figent les réglages)
  service  : comme le service (TF32 activé, température 0,6, top_k 30, morceaux de 4, graine 1234, ICL)
  sans_tf32: idem sans TF32 (= conditions du test d'hier)
  xvec     : idem service mais clonage par empreinte seule (xvec_only=True, pas de continuation de la référence)
  fictif   : idem service avec la référence du pisciniste fictif (comparaison)
Sorties : labo2/etages/<variante>_<n>.wav et labo2/etages/<variante>.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

VARIANTE = sys.argv[1]
ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
if VARIANTE != "sans_tf32":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

REF_PIERRE = str(RACINE / "models" / "ref_pierre.wav")
TEXTE_PIERRE = ("Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
                "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
                "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
                "discussion et ce sera bien.")
FICTIF = RACINE / "personas" / "pisciniste-fictif"
if VARIANTE == "fictif":
    REF, TEXTE = str(FICTIF / "voix.wav"), json.loads((FICTIF / "fiche.json").read_text(encoding="utf-8"))["texte_reference"]
else:
    REF, TEXTE = REF_PIERRE, TEXTE_PIERRE

# Vraies phrases dites par le service le 19/11 (réponses entières, découpées comme labo/cerveau.py)
PHRASES = []
for ligne in (RACINE / "labo" / "journaux" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()[-12:]:
    rep = json.loads(ligne).get("reponse", "")
    PHRASES += [p.strip() for p in re.findall(r".+?[.!?…](?=\s|$)", rep) if len(p.strip()) > 3]
PHRASES = PHRASES[:24]


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


ref_p, _ = sf.read(REF_PIERRE, dtype="float32")
e_pierre = empreinte(ref_p)
ref_v, _ = sf.read(REF, dtype="float32")
e_ref = empreinte(ref_v)

dossier = ICI / "etages"
dossier.mkdir(exist_ok=True)
notes = []
for n, phrase in enumerate(PHRASES, 1):
    torch.manual_seed(1234)
    a = np.concatenate([np.asarray(m, np.float32).reshape(-1) for m, _, _ in tts.generate_voice_clone_streaming(
        text=phrase, language="French", ref_audio=REF, ref_text=TEXTE, chunk_size=4, temperature=0.6, top_k=30,
        xvec_only=(VARIANTE == "xvec"))])
    e = empreinte(a)
    d = {"n": n, "phrase": phrase, "hauteur_1s": round(hauteur(a[:int(1.2 * 24000)])), "hauteur": round(hauteur(a)),
         "sim_ref": round(float(torch.dot(e, e_ref)), 3), "sim_pierre": round(float(torch.dot(e, e_pierre)), 3),
         "duree": round(len(a) / 24000, 1)}
    notes.append(d)
    sf.write(dossier / f"{VARIANTE}_{n:02d}.wav", a, 24000)
    print(VARIANTE, n, d["hauteur_1s"], d["hauteur"], d["sim_ref"], phrase[:50], flush=True)

h = np.array([d["hauteur_1s"] for d in notes])
s = np.array([d["sim_ref"] for d in notes])
ref_h = hauteur(ref_v if len(ref_v.shape) == 1 else ref_v.mean(1))
bilan = {"variante": VARIANTE, "phrases": len(notes), "hauteur_ref": round(ref_h),
         "hors_plage_1s": int(((h < 0.73 * ref_h) | (h > 1.34 * ref_h)).sum()),
         "hauteur_1s_mediane": float(np.median(h)), "hauteur_1s_min_max": [int(h.min()), int(h.max())],
         "sim_moyenne": round(float(s.mean()), 3), "sim_min": round(float(s.min()), 3)}
print("BILAN", json.dumps(bilan, ensure_ascii=False), flush=True)
(dossier / f"{VARIANTE}.json").write_text(json.dumps({"bilan": bilan, "phrases": notes}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
