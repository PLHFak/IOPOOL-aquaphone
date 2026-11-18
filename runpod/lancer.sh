#!/usr/bin/env bash
# Démarre le serveur Aquaphone sur le pod : visage sur la carte 0, voix + transcription sur la carte 1.
# Accès : proxy HTTPS public de RunPod (port 8765 à déclarer en « HTTP port » du pod), protégé par
# le mot de passe AQUAPHONE_MOT_DE_PASSE du fichier .env (le serveur refuse de démarrer sans).
set -euo pipefail
cd /workspace/iopool-aquaphone
grep -q '^AQUAPHONE_MOT_DE_PASSE=.' .env || { echo "Manque AQUAPHONE_MOT_DE_PASSE dans .env"; exit 1; }
mkdir -p labo/journaux
export AQUAPHONE_GPU_VOIX=${AQUAPHONE_GPU_VOIX:-1} AQUAPHONE_HOTE=0.0.0.0 PYTHONIOENCODING=utf-8        HF_HOME=/workspace/.cache/huggingface
pkill -f labo/serveur.py || true
nohup .venv/bin/python -u labo/serveur.py > labo/journaux/serveur.log 2>&1 &
echo "Serveur lancé (journal : labo/journaux/serveur.log, prêt en ≈ 1 à 2 min)."
echo "Adresse : https://${RUNPOD_POD_ID:-<id-du-pod>}-8765.proxy.runpod.net/"
