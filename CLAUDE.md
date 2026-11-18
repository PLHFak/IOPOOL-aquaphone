# Instructions pour Claude Code — PC labo Aquaphone

Tu tournes en terminal sur le **PC labo de Pierre (RTX 4090, 24 Go)**. Lis d'abord `README.md` : il contient la vision, le parcours client, l'architecture et le plan du labo.

## Ta mission actuelle : le kit de test (labo n°1)

Construire un pipeline de bout en bout, en local, et mesurer les délais :

1. **Entrée vocale** — micro du PC → reconnaissance vocale FR (faster-whisper, modèle large-v3-turbo ou équivalent récent).
2. **Cerveau** — texte → Claude via l'API Anthropic (clé dans `.env`, variable `ANTHROPIC_API_KEY`), réponse en streaming. Prompt système : assistant du pisciniste « Marc », se présente comme IA, reformule en une phrase seulement si ambigu, réponses courtes et parlées.
3. **Voix** — synthèse FR avec clonage à partir de `assets/voix_pierre.*` (choisir le meilleur modèle open source FR disponible au moment de l'installation, compatible CUDA).
4. **Visage** — animation de `assets/visage_pierre.jpg` synchronisée sur l'audio (type LivePortrait / MuseTalk ; choisir ce qui tient le temps réel sur 4090).
5. **Mesures** — journaliser le délai de chaque étape et le délai total « fin de phrase → début de réponse ». Objectif < 1,5 s, acceptable < 2 s.

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
