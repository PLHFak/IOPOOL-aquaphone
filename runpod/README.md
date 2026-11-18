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

Prix indicatifs (à vérifier sur runpod.io au moment de louer) : RTX 4090 ≈ 0,69 à 0,89 $/h en Secure Cloud,
0,34 $/h en Community Cloud → **≈ 1,4 à 1,8 $/h pour 2 cartes** en Secure Cloud, plus le volume (quelques $/mois).
**Arrêter le pod après chaque session** : seul le volume reste facturé.

## Mise en route (≈ 30 min la 1re fois)

1. Console Tailscale → Settings → Keys : créer une clé d'authentification (usage unique, 1 jour).
2. RunPod : créer le volume puis le pod (tableau ci-dessus), ouvrir un terminal sur le pod.
3. Sur le pod :
   ```bash
   git clone https://github.com/PLHFak/IOPOOL-aquaphone.git /workspace/iopool-aquaphone
   TS_AUTHKEY=tskey-auth-… bash /workspace/iopool-aquaphone/runpod/installer.sh
   ```
4. Copier les fichiers privés du PC vers le pod (via Tailscale, jamais via Git) :
   `.env`, `assets/visage_pierre.jpg`, `models/ref_pierre.wav`, `models/question_test_16k.wav`, `connaissances/`.
5. `bash /workspace/iopool-aquaphone/runpod/lancer.sh` → ouvrir `https://aquaphone-runpod.<tailnet>.ts.net/`.

Sessions suivantes : démarrer le pod, puis seulement l'étape 5.

## Réglages

- `AQUAPHONE_GPU_VOIX=1` (fixé par `lancer.sh`) : voix + transcription sur la carte 1, visage sur la carte 0.
- Avec deux cartes, on pourra revenir à la qualité maximale du visage (`AQUAPHONE_PAS=5`, `AQUAPHONE_HAUTEUR=960`,
  `AQUAPHONE_LARGEUR=540`) et réduire la réserve de lecture de la page.

## Confidentialité

- La photo et la voix de Pierre ne sont copiées que sur le volume du pod ; supprimer le volume en fin de labo.
- `.env` (clé Anthropic) et `connaissances/` (documents internes iopool) : jamais dans Git, copiés à la main.
- Le serveur n'écoute que sur 127.0.0.1 ; l'accès passe uniquement par Tailscale (pas de port public RunPod).
