# Labo n°1 — Résultats de mesure

Objectif : fin de phrase du client → début de réponse (voix + visage) **< 1,5 s** (acceptable < 2 s).

## Environnement (18/11/2026)

- Windows 11 Famille 10.0.26200 — RTX 4090 24 Go, pilote 610.47 (CUDA 13.3) — 64 Go RAM
- Python 3.11.9, environnement `.venv` — PyTorch 2.11.0+cu128
- Ollama arrêté pendant les tests (il occupait ~17 Go de VRAM)

## Brique 1 — Reconnaissance vocale

- faster-whisper (CTranslate2 4.9.0), modèle `large-v3-turbo`, float16, CUDA, `beam_size=1`, filtre VAD
- Astuce Windows : CTranslate2 utilise les DLL cuBLAS/cuDNN fournies par PyTorch (`torch/lib` ajouté au PATH)
- Test : `assets/voix_pierre.mp4` (25,1 s) — script `labo/test_stt.py`

| Mesure | Valeur |
|---|---|
| Chargement du modèle (à chaud) | 1,8 s (une seule fois au démarrage) |
| Transcription 25,1 s d'audio | 220–252 ms |
| Estimation pour une question de ~5 s | ~100 ms |

Qualité : très bonne. Erreurs : « Pierre L'Hoest » → « Pierre Loës », « iopool » → « Ayupul ».
→ À corriger avec un vocabulaire d'amorce (`initial_prompt` / `hotwords` : iopool, pisciniste, chlore choc, pH…).

## Brique 3 — Voix clonée FR

Modèle retenu : **Qwen3-TTS 12Hz 1.7B Base** (Alibaba, Apache 2.0, FR natif, clonage par référence audio + transcription).
Référence : `models/ref_pierre.wav` = 20 s extraites de `assets/voix_pierre.mp4` (0,5 → 20,5 s, coupé avant « Je propose » tronqué).

Astuce STT : l'amorce `initial_prompt="Pierre L'Hoest, pisciniste, sonde iopool."` corrige les noms propres.

### Essai 1 — paquet officiel `qwen-tts` (non retenu)

Pas de génération en flux ; boucle Python non optimisée → limitée par le processeur, pas par le GPU.

| Modèle | Calcul pour ~5 s d'audio | Facteur temps réel | VRAM |
|---|---|---|---|
| 1.7B | 7,5–8,2 s | 1,64–1,69 (trop lent) | 4,8 Go |
| 0.6B | 7,4–8,0 s | 1,58–1,66 (idem) | 2,7 Go |

### Essai 2 — `faster-qwen3-tts` 0.5.4 (graphes CUDA, sortie en flux) ✅

- Nécessite `transformers==5.15.1` (la 5.23 casse le tokeniseur audio : `property 'upsample' ... has no setter`).
- Ne pas installer en même temps le paquet officiel `qwen-tts` (même module `qwen_tts`).
- Script `labo/test_tts_rapide.py`, modèle 1.7B, `chunk_size=4` (~0,33 s d'audio par morceau).

| Mesure | Valeur |
|---|---|
| Préchauffage au démarrage (capture graphes CUDA) | 1,7 s, une fois |
| **Délai avant le 1er son** | **192–198 ms** |
| Phrase complète (3,4–4,3 s d'audio) | 1,5–1,7 s de calcul |
| Facteur temps réel | 0,40–0,46 (2,3× plus rapide que la parole) |
| VRAM max | 5,1 Go |

Qualité / ressemblance : **à écouter par Pierre** — `models/sorties_tts/rapide_1.7B_phrase*.wav`.
