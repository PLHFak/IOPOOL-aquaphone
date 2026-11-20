# Brief de correction — stabiliser la voix clonée d'une phrase à l'autre

Écrit le 20/11/2026 pour être exécuté tel quel (par Claude Code sur le pod ou un développeur).
À lire avant tout : `labo2/PASSATION_DEBUG_VOIX.md` (contexte complet) et
`labo2/ANALYSE_BUG_VOIX.md` (diagnostic). Rappel des règles : travailler dans `labo2/`,
ne rien écraser, pas de test GPU pendant qu'une conversation est en service, voix de
Pierre jamais dégradée, ne jamais Commiter `.env`/`assets`/`models`/WAV.

## Rappel du bug

Chaque phrase est un appel indépendant de clonage Qwen3-TTS. Le timbre (« code predictor »,
15 codebooks sur 16) y est tiré à température 0.9 / top_k 50 codés en dur dans le graphe
CUDA de faster-qwen3-tts : nos réglages 0.6/30 ne le contrôlent pas. D'où des phrases à
167-230 Hz chez une référence à 123 Hz, parfois une « voix de femme », sans continuité
entre phrases. Le contrôle actuel par hauteur (serveur.py `parler_controle`) se trompe
d'octave, ses bornes sont larges, et son 3e essai est joué quoi qu'il arrive.

## Objectif

Une même réponse de plusieurs phrases doit sonner comme **une seule et même personne**,
en l'occurrence Pierre. Critère final : l'oreille de Pierre. Chiffres d'appui :
similarité cosinus d'empreinte de locuteur ≥ au meilleur tiers des tirages de
`labo2/controle/notes.json`, hauteur médiane dans 110-140 Hz.

## Corrections à implémenter (dans l'ordre)

### 1. `labo2/essai_empreinte.py` — démontrer le contrôle par empreinte (nouveau fichier)

Reprendre la structure de `test_controle_voix.py` (mêmes 20 phrases, même référence,
empreinte via `coeur.extract_speaker_embedding`, similarité cosinus vs `ref_pierre.wav`) :

- pour chaque phrase : jusqu'à **5 essais** (graines 1234+k), on **génère l'essai en
  entier** puis on décide (pas de `close()` à mi-course, élimine l'hypothèse d'état GPU
  sale, hypothèse D de la passation) ;
- on retient l'essai de plus forte similarité cosinus ;
- journaliser par phrase : similarité et hauteur de chaque essai, essai retenu.

Sorties : `labo2/empreinte/<k>_<n>.wav`, `labo2/empreinte/choisi_<n>.wav`,
`labo2/empreinte/notes.json`. **Lancer sur le pod** :
`CUDA_VISIBLE_DEVICES=1 .venv/bin/python labo2/essai_empreinte.py` (hors service).

Tranche : si le cosinus sépare nettement les tirages « Pierre » des tirages dérivés,
le contrôle par empreinte est le bon remplacement. Caler le seuil sur ces notes
(prendre la valeur médiane des bons tirages moins ~0,05).

### 2. Recapturer le graphe du predictor avec un hasard réduit (en mémoire, sans toucher la bibliothèque)

Comme `test_hasard_reduit.py` : `pg = tts.predictor_graph ; pg.temperature, pg.top_k = 0.5, 25 ;
pg.capture(num_warmup=3)`. Le halving seul a paru décevant à l'oreille, mais combiné au
contrôle par empreinte il réduit le nombre de refus (donc le délai). À mesurer :
refus par phrase avec predictor 0.9/50 vs 0.5/25, sur les mêmes 20 phrases.

### 3. Comparer TF32 on/off (explique test vs service)

`labo2/test_etages.py` existe déjà avec les variantes `service` (TF32) et `sans_tf32`.
Les lancer toutes les deux hors service et comparer `hors_plage_1s` et `sim_moyenne`
des bilans JSON. Si TF32 dégrade nettement, l'essayer en désactivant
`torch.backends.cuda.matmul.allow_tf32` dans le brief serveur (secondaire).

### 4. Intégrer `parler_controle` v2 dans `labo/serveur.py` (seule modification du serveur)

Remplacer, dans `parler_controle()` l.140-170 :

- la mesure de hauteur par la similarité d'empreinte (seuil de l'étape 1) ;
- garder la hauteur en garde-fou secondaire (écarter seulement si les deux signaux
  dérivent), bornes resserrées à (0.80-1.25)×référence ;
- `CONTROLE_ESSAIS` passé de 3 à 5, et le **dernier essai n'est plus joué aveuglément** :
  on joue le meilleur des 5 même si sous le seuil (on ne peut pas ne pas répondre) ;
- éviter le `flux.close()` à mi-course : consommer l'essai en entier dans le fil EXEC
  avant de décider. Coût : 1-2 s de délai en plus par phrase abandonnée, acceptable une
  exception à la règle « flux » uniquement pour les phrases courtes (< 8 s) ;
- versionner `pg.temperature/pg.top_k` recapturés au chargement (étape 2).

Mise à jour des stats menu ☰ : ajouter `similarite_moyenne` et `meilleur_essai` dans
`VOIX_STATS`. Après déploiement, **régénérer le cache des phrases d'attente**
(réglages clé de cache : changer la valeur de réglages utilisée dans
`attente.charger` ou purger `models/attente/*.pkl` avec l'accord de Pierre, les
anciens clips portent les anciens tirages).

### 5. Si la dérive persiste à l'oreille (repli)

- **G** : générer la réponse entière en un seul appel (la dérive ne peut plus survenir
  qu'entre réponses). Mesurer le délai fin de question → début de réponse, cible < 2 s.
- **F** : mode x-vector seul (`xvec_only=True`, variante de `test_etages.py`).
- **B** : paquet officiel `qwen-tts` dans un venv séparé comme référence qualité.

## Critères de succès

- Réponse de 5+ phrases : Pierre reconnaît sa voix du début à la fin.
- Journal service : plus aucun retenu en dessous du seuil cosinus ; taux de refus
  par phrase < 1,5 en moyenne.
- Délai fin de question → début de réponse toujours < 2 s.
- Écoute finale : `labo2/ecoute.html`, jugement à l'oreille de Pierre, sans chiffre
  suffisant seul.

## Accès

Pod RunPod `hn5obumrqbyns1` (2 × RTX 4090), service `bash runpod/lancer.sh`, journal
`labo/journaux/serveur.log` (vérifier « Navigateur connecté » avant tout test GPU).
Accès SSH et mot de passe de la page : demander à Pierre.

---

## Suivi d'exécution (20/11/2026, Claude)

| Étape | État |
|---|---|
| 1. `labo2/essai_empreinte.py` | écrit, **non exécuté** (pas d'accès au pod). Toujours N essais complets ; mesure aussi la 1re seconde et demie (`sim_debut`) et le temps de génération. Le bilan propose un seuil (proxy : tirages dont la hauteur est proche de la référence) à confirmer à l'oreille. |
| 2. Predictor 0.5/25 | `essai_empreinte.py --predictor 0.5,25` (mesure) ; `AQUAPHONE_PREDICTOR=0.5,25` (serveur). |
| 3. TF32 | rien à écrire : `test_etages.py service` puis `test_etages.py sans_tf32`. `essai_empreinte.py --sans-tf32` aussi. |
| 4. `serveur.py` v2 | écrit, **non testé**. Inactif tant que `AQUAPHONE_SEUIL_EMPREINTE` n'est pas fixé (le contrôle v1 reste alors en place). Pierre seulement. Phrases ≤ 120 caractères ; au-delà v1 en flux. |
| 5. Repli | non commencé. |

Points ouverts :
- **Délai** : en v2 la phrase n'est rendue qu'une fois générée en entier, même si le 1er tirage est bon. À mesurer ; si `correlation_sim_debut_vs_sim` est forte, décider sur la 1re seconde et demie reste possible.
- **Règle hauteur / empreinte** : « écarter seulement si les deux dérivent » repêche un tirage dont l'empreinte est mauvaise mais la hauteur correcte. `sous_le_seuil_mais_hauteur_ok` (bilan) et `sauvees_par_hauteur` (menu ☰) disent si cela arrive.
- **Hypothèse D** (`close()` à mi-course) non testée par l'étape 1, qui génère tout.
- `labo2/ANALYSE_BUG_VOIX.md`, cité par le brief, n'est pas dans le dépôt.
