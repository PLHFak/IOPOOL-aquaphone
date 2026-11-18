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

## Brique 2 — Cerveau (Claude, streaming)

- SDK `anthropic` 2.10.0, `labo/cerveau.py` : streaming + découpage en phrases (chaque phrase complète part aussitôt vers la voix).
- Prompt système : assistant IA de Marc, réponses parlées, 2–4 phrases, une question à la fois, reformulation seulement si ambigu, pas de dosage sans mesures, escalade vers Marc.
- Cas de test : « Mon eau est verte depuis deux jours, j'ai mis du chlore choc et ça ne change rien. » — script `labo/test_cerveau.py`, médiane de 3 essais.

| Réglage | 1er mot | **1re phrase complète** | Réponse totale |
|---|---|---|---|
| Opus 5.5, effort `low` (réflexion toujours active) | 1 454 ms | **1 827 ms** | 2 659 ms |
| Sonnet 5.5, sans réflexion (`between_tools`), effort `low` | 517 ms | **781 ms** | 1 350 ms |
| Haiku 5.5, sans réflexion, effort `low` | 453 ms | **699 ms** | 944 ms |

Qualité (1er essai) :
- Opus : pH trop haut (> 7,6), filtration en continu, demande le pH. Très bon, 4 phrases.
- Sonnet : empathie + pH trop haut + demande la valeur du pH. Très bon, 3 phrases, le plus naturel à l'oral.
- Haiku : reste vague (« équilibre ou filtration »), pose deux questions à la fois. Correct mais moins utile.

→ Opus fait à lui seul exploser le budget (1,8 s avant la 1re phrase). **Sonnet 5.5 sans réflexion** : meilleur compromis qualité/délai.

## Brique 4 — Visage (Ditto, PyTorch)

Modèle retenu : **Ditto** (Ant Group, Apache 2.0, `antgroup/ditto-talkinghead`) — anime toute la tête (mouvements, clignements, lèvres) à partir d'**une seule photo**, en flux. MuseTalk écarté : synchronisation des lèvres seule (visage figé sur un selfie).

- Code cloné dans `third_party/ditto` (ignoré par Git), poids PyTorch (2,2 Go) dans `models/ditto`.
- Audit avant exécution : `.pkl` de config = tableaux numpy uniquement (vérifié par `pickletools`, sans exécution) ; `.pth` chargés par PyTorch 2.11 en `weights_only` ; pas d'accès réseau ni de commande système dans `core/`.
- Photo source réduite à 720×960 (`models/visage_pierre_960.jpg`).
- Script `labo/test_visage.py` : mode flux (audio par morceaux de 0,2 s), phrase de 3,6 s en voix clonée.

### Adaptations Windows (sans rien installer hors du dépôt)

| Problème | Correctif |
|---|---|
| Module Cython `blend` → exige un compilateur C | Remplacé par numpy, calcul identique (`labo/patches/ditto_blend_numpy.patch`) |
| Mode flux : 35 images au lieu de 90 | `overlap_v2=70` (valeur de la config « online » officielle) |
| onnxruntime-gpu 1.32 compilé pour CUDA 13 → HuBERT tournait **sur le processeur** (52 ms) | onnxruntime-gpu **1.23.2** (CUDA 12) + préchargement des DLL de PyTorch (`labo/cuda_win.py`) → 8,6 ms |
| Déformation + décodage : 24,5 ms/image | `torch.compile` (triton-windows, dans le `.venv`) → 14 ms ; compilation dans le fil principal (dans les fils de Ditto elle se bloque) |
| `cudnn.benchmark` | **À éviter** : tailles variables en flux → 11 s avant la 1re image |
| TensorRT | Non retenu : plugin `grid_sample_3d` fourni pour Linux seulement |

### Mesures (phrase de 3,6 s, 2e essai)

| Réglage | 1re image | Images/s (25 requis) | VRAM |
|---|---|---|---|
| Départ (10 pas, HuBERT sur CPU) | 1 155 ms | 15,7 ❌ | 1,6 Go |
| 5 pas | 764 ms | 20,1 ❌ | 1,6 Go |
| 5 pas + HuBERT GPU | 570 ms | 20,3 ❌ | 1,6 Go |
| 10 pas + HuBERT GPU + compile | 758 ms | 21,5 ❌ | 1,4 Go |
| **5 pas + HuBERT GPU + compile** ✅ | **530 ms** | **29,0** | 1,4 Go |

Préparation de la photo : 0,16 s (une fois par conversation). torch.compile au démarrage : ~50 s la 1re fois, puis ~5–12 s (cache).

Qualité / ressemblance : **à regarder par Pierre** — `models/sorties_visage/DEMO_visage_voix_pierre.mp4`.

## Bilan provisoire du délai (fin de phrase → début de réponse)

| Étape | Délai |
|---|---|
| Détection fin de phrase (VAD) | à mesurer (brique 5) |
| Reconnaissance vocale | ~0,1 s |
| Claude Sonnet 5.5 → 1re phrase | ~0,8 s |
| Voix → 1er son | ~0,2 s |
| Visage → 1re image | ~0,5 s |
| **Total estimé (hors VAD et réseau)** | **~1,6 s** |

Pistes pour passer sous 1,5 s : lancer la voix sur le **début** de la 1re phrase (premiers mots), réduire le préchauffage de Ditto (`chunksize`), envoyer le son et l'image ensemble dès la 1re image.

## Briques 5 et 6 — Pipeline de bout en bout + page web (18/11/2026)

Serveur `labo/serveur.py` (aiohttp, 127.0.0.1:8765) + page `labo/web/index.html`, publiée en HTTPS
sur le tailnet par `tailscale serve`. Micro du navigateur en PCM 16 kHz par WebSocket ; retour en
paquets « 1 image JPEG 540×720 + ses 40 ms d'audio 24 kHz » (synchro lèvres/son par construction).
Fin de parole : Silero VAD en flux (0,5 s de silence). Semi-duplex : le micro est ignoré pendant la réponse.

Test sans navigateur (`labo/test_pipeline.py`, cas de test dit par la voix clonée, joué en temps réel) :

| Étape (ms) | Essai 1 | Essai 2 | Essai 3 |
|---|---|---|---|
| Attente du silence (VAD) | 454 | 472 | 500 |
| Transcription | 277 | 412 | 275 |
| Claude Sonnet, 1re phrase | 898 | 942 | 757 |
| Voix, 1er son | 217 | 216 | 217 |
| Visage, 1re image | 1240 | 1164 | 1242 |
| Tampon de lecture | 153 | 152 | 143 |
| **Total fin de parole → réponse** | **3240** | **3358** | **3133** |
| Coupures pendant la réponse | 7 (1,2 s) | 7 (1,5 s) | 7 (1,1 s) |

Constats :
- Le visage coûte 1,2 s en flux réel (contre 0,5 s en test isolé, où tout l'audio était déjà là) :
  Ditto attend ~0,6 s d'audio (fenêtre HuBERT + lot de 10 images) avant la 1re image.
- Voix et visage en même temps sur la carte : Ditto tombe à ~20 images/s → coupures.
- Corrigé : recompilation `torch.compile` dans les fils de Ditto (garde « num_threads ») qui bloquait tout.
- Essayé sans succès : flux CUDA prioritaire pour Ditto (plus aucune image produite).

### Correctifs « ça bloque » (18/11/2026)

Constat sur les essais de Pierre (page web via Tailscale) : son qui se fige par moments, et une réponse
donnée à une phrase coupée (« Bonjour, je voudrais savoir comment je peux... »).

| Réglage (cas de test, 3 essais) | Coupures | Durée des coupures | 1re image | Total perçu |
|---|---|---|---|---|
| Avant : visage 5 étapes, tampon 0,15 s | 7 à 13 | 1,2 à 1,9 s | 1,1 à 1,4 s | 3,1 à 3,4 s |
| Tampon 0,5 s seul | 6 | 2,2 s | 1,2 s | 3,5 s |
| Tampon 1,0 s seul | 3 à 6 | 1,1 à 1,8 s | 1,2 à 1,4 s | 4,0 s |
| **Visage 3 étapes + tampon 0,5 s** | **0 à 5 (moy. 2)** | **0 à 1,8 s (moy. 0,7)** | **1,0 à 1,2 s** | **3,2 à 3,4 s** |

- Le tampon seul ne suffit pas : le déficit vient de la carte graphique partagée voix + visage pendant
  toute la réponse. Visage allégé (`AQUAPHONE_PAS=3`, étapes du modèle de mouvement Ditto) : seul,
  ~38 images/s au lieu de ~30 → assez de marge la plupart du temps.
- Phrase inachevée : si la transcription finit par « ... », on attend la suite (1,2 s max) et on recolle
  les deux morceaux avant de répondre ; le micro reçu pendant la transcription est rejoué.
- À surveiller : naturel du visage à 3 étapes (avis de Pierre), légère variation du timbre d'une phrase
  à l'autre (voix générée phrase par phrase).
