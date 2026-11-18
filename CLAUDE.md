# Instructions pour Claude Code — PC labo Aquaphone

Tu tournes en terminal sur le **PC labo de Pierre (RTX 4090, 24 Go)**. Lis d'abord `README.md` : il contient la vision, le parcours client, l'architecture et le plan du labo.

## Ta mission actuelle : le kit de test (labo n°1)

Construire un pipeline de bout en bout, en local, et mesurer les délais :

1. **Entrée vocale** — micro du PC → reconnaissance vocale FR (faster-whisper, modèle large-v3-turbo ou équivalent récent).
2. **Cerveau** — texte → Claude via l'API Anthropic (clé dans `.env`, variable `ANTHROPIC_API_KEY`), réponse en streaming. Prompt système : assistant du pisciniste « Marc », se présente comme IA, reformule en une phrase seulement si ambigu, réponses courtes et parlées.
3. **Voix** — synthèse FR avec clonage à partir de `assets/voix_pierre.*` (choisir le meilleur modèle open source FR disponible au moment de l'installation, compatible CUDA).
4. **Visage** — animation de `assets/visage_pierre.jpg` synchronisée sur l'audio (type LivePortrait / MuseTalk ; choisir ce qui tient le temps réel sur 4090).
5. **Mesures** — journaliser le délai de chaque étape et le délai total « fin de phrase → début de réponse ». Objectif < 1,5 s, acceptable < 2 s.

6. **Interface de test** — Pierre pilote le PC à distance (Bureau à distance Chrome) : il n'est pas devant le PC, et son micro n'est pas transmis au PC. Sers une petite page web sur le PC, accessible depuis son Mac ou son iPhone via Tailscale, qui capte le micro du navigateur et affiche le visage animé avec le son. C'est aussi la préfiguration de l'application iPhone.

Cas de test : « Mon eau est verte depuis deux jours, j'ai mis du chlore choc et ça ne change rien. »

## Méthode

- Commence par détecter l'environnement (OS, version CUDA, pilotes, Python, espace disque) et annonce ce que tu as trouvé avant d'installer quoi que ce soit.
- Une brique à la fois : installer → tester seule → mesurer → passer à la suivante.
- Environnement Python isolé (venv ou conda) dans ce dépôt ; ne rien installer globalement sans le demander.
- Avant de télécharger un modèle de plusieurs Go, annonce sa taille.
- Commit Git après chaque brique qui fonctionne (message en français). Ne jamais commiter `.env`, les clés, ni les modèles téléchargés (les mettre dans `.gitignore`).
- Consigne les résultats de mesure dans `labo/resultats.md`.

## Règles

- **Visage et voix : uniquement ceux de Pierre** (fichiers dans `assets/`). Ne jamais utiliser le visage ou la voix d'un tiers sans accord écrit.
- Le PC sert aussi à d'autres usages (conférences, jeux de caméra) : ne pas modifier les pilotes GPU ni les logiciels existants sans demander.
- Réponds à Pierre en français, de façon concise et structurée ; il dicte souvent à la voix, interprète les erreurs de transcription.

## Fichiers attendus de Pierre (copiés à la main, jamais commités : le dépôt est public)

- `assets/visage_pierre.jpg` — selfie de face, bouche fermée
- `assets/voix_pierre.mp4` (ou .wav) — vidéo/enregistrement de sa voix

## État du labo n°1 — passation (18/11/2026)

Le pipeline de bout en bout fonctionne (détails et mesures : `labo/resultats.md`).

**Où ça tourne** : service sur le PC labo (`python labo/serveur.py`, page privée
`https://duplex-pc.taila4e728.ts.net/` via `tailscale serve --bg 8765`). Cible : **RunPod** (pod 2 × RTX 4090,
Europe, Secure Cloud), le PC devient serveur de secours. Développement repris depuis le Mac de Pierre.
Préparation RunPod prête mais **jamais testée sous Linux** : `runpod/README.md`, `installer.sh`, `lancer.sh`.

**Décisions de Pierre (à respecter)**
- **Voix : jamais dégradée** (Qwen3-TTS 1.7B). En cas de manque de puissance, on dégrade le visage
  (`AQUAPHONE_PAS`, `AQUAPHONE_HAUTEUR`, `AQUAPHONE_LARGEUR`), pas la voix.
- Délai : accepter quelques dixièmes de seconde de plus pour une réponse mieux informée.
- **Documentation iopool** : zip interne du support, extrait dans `connaissances/` (gitignoré, jamais commité,
  dépôt public). `labo/contexte.py` n'en garde que la partie autorisée pour un client (voir son README) et
  la met en cache côté API (1 h). Jamais de dose chiffrée : l'application iopool la calcule.
- **Mode labo** : phrase commençant par « Aquaphone » → ni voix, ni visage, ni Claude ; la demande est écrite
  dans `labo/journaux/demandes_vocales.md`. Claude Code surveille ce fichier (Monitor, `tail -F`) et répond
  **par écrit** dans le terminal et sur la page (`python labo/note.py "texte"`). « iopool » → retour client.
- **Anti-écho** (haut-parleur → micro) : la page coupe le micro pendant la lecture + 0,6 s, le serveur aussi,
  et jette une phrase qui répète ce que l'assistant vient de dire. Pas encore de coupure de parole (barge-in).
- Ne pas lancer de test automatique (`labo/test_pipeline.py`) pendant que Pierre utilise la page :
  le serveur ne gère qu'une conversation à la fois.

**Problèmes ouverts**
1. Coupures du son : la voix et le visage se partagent la carte graphique. Le serveur journalise par réponse
   (« Diagnostic … ») les trous vus par la page, la vitesse de la voix et les images/s du visage → RunPod.
2. Délai fin de phrase → réponse ≈ 3 à 5 s (objectif < 2 s) : 1re image du visage ≈ 1 s, Claude 0,7 à 2 s,
   réserve de lecture 1 s.
3. Timbre de la voix qui varie un peu d'une phrase à l'autre (voix générée phrase par phrase).

**Fichiers privés à copier à la main** (Mac ou RunPod, jamais dans Git) : `.env`, `assets/`,
`models/ref_pierre.wav`, `models/question_test_16k.wav`, `connaissances/`.
Après le labo : faire tourner la clé API Anthropic.
