"""Essai : contrôle de la voix par EMPREINTE de locuteur (aucun fichier existant modifié).

Brief : labo2/BRIEF_CORRECTION_VOIX.md, étape 1 (et étape 2 avec --predictor).
Pour chacune des 20 phrases de test_service.py : N essais (graines 1234+k), CHAQUE essai est généré en entier
(pas de close() à mi-course → écarte l'hypothèse D « état GPU sale après abandon »), puis noté :
  - similarité cosinus de l'empreinte du locuteur (encodeur de Qwen3-TTS) avec celle de ref_pierre.wav,
    sur la phrase entière (sim) et sur sa 1re seconde et demie seulement (sim_debut) ;
  - hauteur médiane (Hz), durée, temps de génération.
Tous les essais sont toujours générés : on peut ensuite simuler n'importe quel seuil sur notes.json sans relancer.

Usage (sur le pod, HORS service, aucune conversation en cours) :
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python labo2/essai_empreinte.py                    # réglages du service (TF32 actif)
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python labo2/essai_empreinte.py --predictor 0.5,25 # étape 2 : timbre moins aléatoire
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python labo2/essai_empreinte.py --sans-tf32        # étape 3 (variante)
Sorties : labo2/empreinte[_variante]/<k>_<n>.wav, choisi_<n>.wav, notes.json (refuse d'écraser un dossier existant).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ap = argparse.ArgumentParser()
ap.add_argument("--essais", type=int, default=5)
ap.add_argument("--predictor", default="", help="T,K : température et top_k du code predictor (défaut : 0.9,50 de la bibliothèque)")
ap.add_argument("--sans-tf32", action="store_true", help="désactive TF32 (le service l'active)")
args = ap.parse_args()

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
variante = []
if args.predictor:
    variante.append("pred" + args.predictor.replace(",", "_"))
if args.sans_tf32:
    variante.append("sanstf32")
dossier = ICI / ("empreinte" + ("_" + "_".join(variante) if variante else ""))
if (dossier / "notes.json").exists():
    sys.exit(f"{dossier / 'notes.json'} existe déjà : rien n'est écrasé. Renommer ou déplacer ce dossier d'abord.")

torch.backends.cuda.matmul.allow_tf32 = not args.sans_tf32  # comme labo/serveur.py
torch.backends.cudnn.allow_tf32 = not args.sans_tf32
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

src = (ICI / "test_service.py").read_text(encoding="utf-8")
debut = src.index("PHRASES = [") + len("PHRASES = ")
PHRASES = eval(src[debut:src.index("]\n", debut) + 1])
REF = str(RACINE / "models" / "ref_pierre.wav")
TEXTE_PIERRE = ("Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
                "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
                "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
                "discussion et ce sera bien.")
DEBUT_S = 1.5  # durée de tête comparée à la phrase entière (pour savoir si l'on peut décider plus tôt)


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
if args.predictor:
    t_pred, k_pred = args.predictor.split(",")
    pg = tts.predictor_graph
    pg.temperature, pg.top_k = float(t_pred), int(k_pred)
    pg.capture(num_warmup=3)  # graphe du timbre recapturé en mémoire ; la bibliothèque n'est pas modifiée
    print("code predictor :", pg.temperature, pg.top_k, flush=True)
coeur = tts.model.model


def empreinte(a):
    e = coeur.extract_speaker_embedding(audio=a.astype(np.float32), sr=24000).float().reshape(-1)
    return torch.nn.functional.normalize(e, dim=0)


ref, sr_ref = sf.read(REF, dtype="float32")
if ref.ndim > 1:
    ref = ref.mean(1)
if sr_ref != 24000:
    ref = resample_poly(ref, 24000, sr_ref).astype(np.float32)
e_ref, h_ref = empreinte(ref), hauteur(ref)
print(f"référence : {h_ref:.0f} Hz — TF32 {'non' if args.sans_tf32 else 'oui'} — {args.essais} essais × {len(PHRASES)} phrases", flush=True)

dossier.mkdir(exist_ok=True)
notes = []
for n, phrase in enumerate(PHRASES, 1):
    essais = []
    for k in range(args.essais):
        torch.manual_seed(1234 + k)
        t0, t1, morceaux = time.perf_counter(), None, []
        for m, _, _ in tts.generate_voice_clone_streaming(text=phrase, language="French", ref_audio=REF,
                                                          ref_text=TEXTE_PIERRE, chunk_size=4,
                                                          temperature=0.6, top_k=30):
            t1 = t1 or time.perf_counter() - t0
            morceaux.append(np.asarray(m, np.float32).reshape(-1))
        a = np.concatenate(morceaux)  # génération menée jusqu'au bout, quoi qu'il arrive
        t_tot = time.perf_counter() - t0
        tete = a[:int(DEBUT_S * 24000)]
        essais.append({"essai": k + 1, "sim": round(float(torch.dot(empreinte(a), e_ref)), 3),
                       "sim_debut": round(float(torch.dot(empreinte(tete), e_ref)), 3),
                       "hauteur": round(hauteur(a)), "duree": round(len(a) / 24000, 1),
                       "t_premier_s": round(t1 or 0, 2), "t_total_s": round(t_tot, 2)})
        sf.write(dossier / f"{k + 1}_{n:02d}.wav", a, 24000)
    meilleur = max(essais, key=lambda e: e["sim"])
    (dossier / f"choisi_{n:02d}.wav").write_bytes((dossier / f"{meilleur['essai']}_{n:02d}.wav").read_bytes())
    notes.append({"n": n, "phrase": phrase, "essais": essais, "choisi": meilleur["essai"]})
    print(f"{n:2d} " + " | ".join(f"{e['sim']:.3f}/{e['hauteur']}Hz" for e in essais) + f" → essai {meilleur['essai']}", flush=True)

# ─── Bilan : seuil proposé et nombre de refus attendus ─────────────────────────
tous = [e for p in notes for e in p["essais"]]
sim = np.array([e["sim"] for e in tous])
sim_d = np.array([e["sim_debut"] for e in tous])
haut = np.array([e["hauteur"] for e in tous])
bons = sim[(haut >= 0.8 * h_ref) & (haut <= 1.25 * h_ref)]  # PROXY : « bon tirage » = hauteur proche de la référence
seuil_propose = round(float(np.median(bons)) - 0.05, 3) if len(bons) else None
bilan = {"reference_hz": round(h_ref), "tf32": not args.sans_tf32, "predictor": args.predictor or "0.9,50 (bibliothèque)",
         "essais_total": len(tous), "tirages_hauteur_ok": int(len(bons)),
         "sim_mediane": round(float(np.median(sim)), 3), "sim_min_max": [round(float(sim.min()), 3), round(float(sim.max()), 3)],
         "correlation_sim_debut_vs_sim": round(float(np.corrcoef(sim, sim_d)[0, 1]), 3),
         "seuil_propose_proxy": seuil_propose, "simulation_par_seuil": {}}
if seuil_propose is not None:  # combien de tirages « sous le seuil » auraient été repêchés par la hauteur (règle OU du serveur)
    bilan["sous_le_seuil_mais_hauteur_ok"] = int(((sim < seuil_propose) & (haut >= 0.8 * h_ref) & (haut <= 1.25 * h_ref)).sum())
    bilan["sous_le_seuil_total"] = int((sim < seuil_propose).sum())
    bilan["seuil_debut_propose_proxy"] = round(float(np.median(sim_d[(haut >= 0.8 * h_ref) & (haut <= 1.25 * h_ref)])) - 0.05, 3)
for s in sorted({0.0, *(np.round(np.quantile(sim, [0.25, 0.4, 0.5, 0.6]), 3).tolist()), *(([seuil_propose]) if seuil_propose else [])}):
    nb, echec = [], 0
    for p in notes:  # essais joués dans l'ordre jusqu'au 1er ≥ seuil ; sinon le meilleur des N
        rang = next((i + 1 for i, e in enumerate(p["essais"]) if e["sim"] >= s), None)
        echec += rang is None
        nb.append(rang or len(p["essais"]))
    tg = np.mean([p["essais"][min(i, len(p["essais"]) - 1)]["t_total_s"] for p, i in zip(notes, [n - 1 for n in nb])])
    bilan["simulation_par_seuil"][str(s)] = {"essais_moyens_par_phrase": round(float(np.mean(nb)), 2),
                                             "phrases_sans_essai_au_seuil": int(echec),
                                             "t_total_dernier_essai_s_moyen": round(float(tg), 2)}
print("BILAN", json.dumps(bilan, ensure_ascii=False, indent=1), flush=True)
(dossier / "notes.json").write_text(json.dumps({"bilan": bilan, "phrases": notes}, ensure_ascii=False, indent=1), encoding="utf-8")
