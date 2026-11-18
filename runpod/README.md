# Aquaphone sur RunPod

But : supprimer les coupures en donnant **une carte graphique à la voix et une au visage**.
Sur le PC labo (une seule RTX 4090), la voix et le visage se partagent la carte et le visage prend du retard.
Claude (le cerveau, avec la documentation iopool en cache) ne change pas : il tourne chez Anthropic.

## Machine à louer

| Choix | Valeur |
|---|---|
| Type | **Pod** (pas Serverless : le serveur garde les modèles chargés et une connexion WebSocket ouverte) |
| Cartes | **2 × RTX 4090** (24 Go chacune ; la voix ≈ 5 Go, le visage ≈ 3 Go) |
| Cloud | **Secure Cloud** (centres de données certifiés : la photo et la voix de Pierre y sont copiées) |
| Région | Europe (EU-…), pour limiter le délai réseau (+20 à 50 ms) |
| Version CUDA du pilote | **12.8 ou plus** (filtre « CUDA Version » de RunPod ; torch 2.11 cu128) |
| Image | `runpod/pytorch` (Ubuntu, Python 3.11) |
| Disque | **Network Volume ≈ 30 Go** monté sur `/workspace` (modèles + code, conservé quand le pod est arrêté) |
| Ports | **HTTP port 8765** (proxy HTTPS public de RunPod) |

Prix indicatifs (à vérifier sur runpod.io au moment de louer) : RTX 4090 ≈ 0,69 à 0,89 $/h en Secure Cloud,
0,34 $/h en Community Cloud → **≈ 1,4 à 1,8 $/h pour 2 cartes** en Secure Cloud, plus le volume (quelques $/mois).
**Arrêter le pod après chaque session** : seul le volume reste facturé.

## Mise en route (≈ 30 min la 1re fois)

1. RunPod : créer le volume puis le pod (tableau ci-dessus), ouvrir un terminal sur le pod.
2. Sur le pod :
   ```bash
   git clone https://github.com/PLHFak/IOPOOL-aquaphone.git /workspace/iopool-aquaphone
   bash /workspace/iopool-aquaphone/runpod/installer.sh
   ```
3. Copier les fichiers privés du PC vers le pod (jamais via Git) :
   `.env` (avec `AQUAPHONE_MOT_DE_PASSE`), `assets/visage_pierre.jpg`, `models/ref_pierre.wav`,
   `models/question_test_16k.wav`, `connaissances/`.
4. `bash /workspace/iopool-aquaphone/runpod/lancer.sh` → ouvrir `https://<id-du-pod>-8765.proxy.runpod.net/`
   et saisir le mot de passe (une fois par appareil, retenu 30 jours).

Sessions suivantes : démarrer le pod, puis seulement l'étape 4.

## Réglages

- `AQUAPHONE_GPU_VOIX=1` (fixé par `lancer.sh`) : voix + transcription sur la carte 1, visage sur la carte 0.
- Avec deux cartes, on pourra revenir à la qualité maximale du visage (`AQUAPHONE_PAS=5`, `AQUAPHONE_HAUTEUR=960`,
  `AQUAPHONE_LARGEUR=540`) et réduire la réserve de lecture de la page.

## Confidentialité et accès

- Adresse publique protégée par **mot de passe** (`labo/acces.py`) : page, photo et WebSocket. Majuscules
  indifférentes ; 5 erreurs depuis une adresse → blocage 15 min ; 30 erreurs en 1 h → tout bloqué 1 h.
  Le flux (voix, visage, texte) est chiffré en HTTPS. Changer le mot de passe après chaque série de tests.
- Tailscale reste possible en plus (`TS_AUTHKEY=… bash installer.sh`), mais n'est plus nécessaire.
- La photo et la voix de Pierre ne sont copiées que sur le volume du pod ; supprimer le volume en fin de labo.
- `.env` (clés, mot de passe) et `connaissances/` (documents internes iopool) : jamais dans Git, copiés à la main.
