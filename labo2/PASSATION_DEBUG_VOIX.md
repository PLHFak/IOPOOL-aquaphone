# Passation debug — la voix clonée change d'une phrase à l'autre

Pour : un développeur / debugger qui n'a pas suivi le labo. Rédigé le 19/11/2026 (Claude Code).
Lire aussi : `CLAUDE.md` (règles), `labo/resultats.md` (mesures), `runpod/README.md` (déploiement).

## 1. Symptôme (rapporté par Pierre)

- Dans une même réponse, la voix change **complètement** d'une phrase à l'autre : timbre, débit, parfois voix
  de femme. « On dirait qu'elle change volontairement », « le texte semble influencer la voix ».
- La toute première expérience sur le PC (même moteur, réglages par défaut) lui a paru stable.
- Le contrôle de hauteur ajouté le 19/11 à 03h15 (commit `70f6180`) n'a pas réglé le ressenti.

## 2. Chaîne audio (ce qu'il faut savoir)

```
micro page → WebSocket → Silero VAD → faster-whisper → Claude (streaming, découpé en phrases, labo/cerveau.py)
  → pour CHAQUE phrase : Qwen3-TTS 1.7B Base, clonage (ref audio + transcription), faster-qwen3-tts 0.5.4
  → contrôle 1re seconde (hauteur) → Ditto (visage) → page
```

- Code voix : `labo/serveur.py`, fonctions `parler()` et `parler_controle()`, chargement dans `charger()`.
- Chaque phrase = **un appel indépendant** `generate_voice_clone_streaming(text=phrase, ref_audio, ref_text,
  temperature=0.6, top_k=30, chunk_size=4)`, graine `torch.manual_seed(1234 + essai)`.
- Aucune continuité entre phrases : le modèle ne « sait » pas comment il a prononcé la phrase précédente.
- Un seul fil GPU pour la voix (`EXEC`, ThreadPoolExecutor max_workers=1) : pas de générations simultanées.
- Les phrases d'attente (« Bonne question, je vérifie… ») sont des clips pré-calculés au démarrage
  (`labo/attente.py`, cache disque) : même moteur, autre moment.

## 3. Ce qui a été vérifié (faits)

| Vérification | Résultat |
|---|---|
| Code du pod = code local | oui (md5 identiques de `serveur.py`, `index.html`, `cerveau.py` le 19/11 08h15) |
| Référence de Pierre `models/ref_pierre.wav` | 20 s, 24 kHz, sa voix seule ; transcription Whisper = `REF_TEXTE` (aux noms propres près) |
| Référence du pisciniste fictif `personas/pisciniste-fictif/voix.wav` | 8,5 s, voix **synthétique** (Qwen3-TTS VoiceDesign), transcription = `texte_reference` |
| Hauteur médiane des références | Pierre 123 Hz ; pisciniste fictif 194 Hz |
| Températures | `temperature`/`top_k` ne s'appliquent qu'au 1er codebook ; le « code predictor » (15 codebooks sur 16 = timbre) tire à **température 0,9 / top_k 50 codés en dur** et capturés dans un graphe CUDA (`faster_qwen3_tts/model.py` ~l.215-223, `predictor_graph.py`) |
| Test 20 phrases × 4 graines (`labo2/test_controle_voix.py`, sorties `labo2/controle/`) | ≈ 1 tirage sur 8 part à 180-230 Hz (référence 123 Hz) ; la 1re seconde suffit à le détecter (10/10, 3 fausses alertes / 80) |
| Halving du hasard (`labo2/test_hasard_reduit.py`) | pas d'amélioration nette |

## 4. Ce que montre le journal du service (pod, 19/11 matin)

Personnage **Pierre** (08h08-08h09) — beaucoup plus de rejets qu'en test :
```
essai 1 écarté (172 Hz) / essai 2 écarté (88 Hz) / essai 3 retenu (132 Hz)  « Quel est le problème exactement… »
essai 1 écarté (192 Hz) / essai 2 retenu (114 Hz)                           « D'accord, c'est cohérent… »
essai 1 écarté (188 Hz) / essai 2 écarté (227 Hz) / essai 3 retenu (142 Hz) « Pour ne pas me tromper… »
```
Personnage **pisciniste fictif** (depuis 08h10 — Pierre écoutait donc PAS sa voix à ce moment) :
```
essai 1 écarté (107 Hz) / essai 2 retenu (198 Hz)  « Je regarde ça. »
essai 1 écarté (128 Hz) / essai 2 retenu (167 Hz)  « D'accord, voyons… »
essai 1 écarté (106 Hz) / essai 2 retenu (153 Hz)  « Je comprends, laissez-moi regarder. »
```
Lecture : pour le fictif, des phrases retenues à 153 Hz et 198 Hz coexistent → la plage acceptée (142-260 Hz)
laisse passer deux « personnes » différentes. Et le 1er tirage est très souvent hors plage en service.

Le menu ☰ affiche maintenant « Version en service » : empreinte du code, personnage actif, référence,
nombre de phrases refaites depuis le démarrage (route `GET /version`).

## 5. Hypothèses, de la plus probable à la moins probable

1. **Variabilité propre au clonage « en contexte » (ICL) de Qwen3-TTS** : la voix n'est pas un modèle de
   locuteur figé, c'est une *continuation* de la référence. Le texte cible (question, exclamation, phrase très
   courte, ponctuation « : » « … ») change la prosodie et peut faire « glisser » le timbre. → Explique « le texte
   influence la voix ». À tester : même phrase × 10 graines vs 10 phrases × même graine.
2. **Le code predictor à température 0,9 en dur** (bibliothèque faster-qwen3-tts) : le timbre est tiré au
   hasard bien plus librement que nos réglages ne le laissent croire. → Bug/limite *hors de notre code*.
   À tester : patcher une copie de la bibliothèque (labo2) avec 0,5-0,7 et recapturer les graphes.
3. **faster-qwen3-tts ≠ paquet officiel `qwen-tts`** : ré-implémentation (graphes CUDA, cache de prompt
   vocal clé = (chemin, texte, xvec, silence), décodage en flux par morceaux de 4). Une différence de
   comportement (masque, position, contexte du décodeur audio) est possible. → Comparer à l'officiel
   (plus lent, mais sert de référence de qualité ; **ne pas l'installer dans le même venv** : même module).
4. **`flux.close()` en cours de génération** (contrôle de la 1re seconde) : abandonner le générateur à mi-
   chemin pourrait laisser un état (cache KV statique, graphe) sale pour l'appel suivant. Le rejet à 88 Hz
   juste après un rejet à 172 Hz est suspect. À tester : comparer les tirages après un `close()` et après une
   génération complète.
5. **Mesure de hauteur** (autocorrélation simple) : erreurs d'octave possibles (88 Hz ≈ moitié de 176 Hz).
   Le contrôle peut rejeter de bons tirages et en garder de mauvais. Remplacer par l'empreinte du locuteur
   (`tts.model.model.extract_speaker_embedding(audio, sr=24000)`, similarité cosinus, déjà utilisée dans
   `labo2/test_controle_voix.py`) ou par pYIN/CREPE.
6. **Référence synthétique et courte (fictif, 8,5 s)** : cloner une voix elle-même générée amplifie
   l'instabilité. Hors sujet pour la voix de Pierre, mais explique une partie du ressenti depuis 08h10.

## 6. Expériences proposées (dans `labo2/`, sans toucher au service)

| # | Expérience | Ce qui tranche |
|---|---|---|
| A | Même phrase × 10 graines, puis 10 phrases × graine fixe ; mesurer similarité locuteur + hauteur | hasard (A1 varie) vs texte (A2 varie) |
| B | Paquet officiel `qwen-tts` dans un venv séparé, mêmes 20 phrases, même référence | bug de faster-qwen3-tts ou propriété du modèle |
| C | Copie patchée de faster-qwen3-tts : température du code predictor 0,5 / 0,7 / 0,9 | rôle du 0,9 en dur |
| D | Générer 20 phrases avec et sans `close()` prématuré intercalé | état sale après abandon |
| E | Contrôle par empreinte du locuteur (seuil calé sur les tirages jugés bons par Pierre) au lieu de la hauteur | meilleur filtre |
| F | Mode « x-vector seul » (sans transcription de référence, si exposé par la bibliothèque) | ICL responsable de la dérive |
| G | Générer la réponse entière en un seul appel (au lieu de phrase par phrase) | coût en délai vs stabilité |

Écoute : `labo2/ecoute.html` (lecteur des sorties), copies sur le Bureau « Aquaphone - test voix ».
Le jugement final est **à l'oreille de Pierre** : les chiffres ne suffisent pas.

## 7. Accès et règles (impératif)

- Pod RunPod `hn5obumrqbyns1` (2 × RTX 4090) ; service : `bash runpod/lancer.sh` (boucle de redémarrage
  `runpod/boucle.sh`), journal `labo/journaux/serveur.log`. Accès SSH et mot de passe de la page : **demander
  à Pierre** (jamais dans un fichier commité).
- **Ne supprimer ni écraser aucun fichier existant sans l'accord de Pierre** ; travailler dans `labo2/`.
- Ne pas lancer de test sur le GPU du service pendant que Pierre ou des testeurs sont connectés
  (une conversation à la fois) : regarder « Navigateur connecté » dans le journal.
- Voix de Pierre : jamais dégradée (pas de passage au 0.6B pour gagner du temps).
- Voix et visage : uniquement ceux de Pierre ou de personnes ayant donné leur accord écrit.
- Ne jamais commiter `.env`, `assets/`, `models/`, `connaissances/`, ni les WAV.
