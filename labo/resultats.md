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
