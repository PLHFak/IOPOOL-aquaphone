#!/usr/bin/env bash
# Démarrage d'un pod neuf sur le disque réseau (appelé par runpod/a_la_demande.py, par SSH).
# Le disque /workspace garde code, venv et modèles ; le conteneur, lui, est neuf : on remet les paquets système
# puis on lance le service. Une carte → voix et visage sur la carte 0 ; deux cartes → voix sur la 1.
set -euo pipefail
cd /workspace/iopool-aquaphone
command -v ffmpeg >/dev/null && dpkg -s libegl1 >/dev/null 2>&1 || {
  apt-get update -qq && apt-get install -y -qq ffmpeg python3-venv libegl1 libgl1 libgles2 sox >/dev/null; }
CARTES=$(nvidia-smi -L | wc -l)
if [ "$CARTES" -ge 2 ]; then export AQUAPHONE_GPU_VOIX=1; else export AQUAPHONE_GPU_VOIX=0; fi
echo "$CARTES carte(s), voix sur la carte $AQUAPHONE_GPU_VOIX"
bash runpod/lancer.sh
