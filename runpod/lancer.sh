#!/usr/bin/env bash
# Démarre le serveur Aquaphone sur le pod : visage sur la carte 0, voix + transcription sur la carte 1.
set -euo pipefail
cd /workspace/iopool-aquaphone
pgrep -x tailscaled >/dev/null || nohup tailscaled --tun=userspace-networking \
  --state=/workspace/tailscale.state >/workspace/tailscaled.log 2>&1 &
sleep 2; tailscale serve --bg 8765 >/dev/null
mkdir -p labo/journaux
export AQUAPHONE_GPU_VOIX=${AQUAPHONE_GPU_VOIX:-1} PYTHONIOENCODING=utf-8 HF_HOME=/workspace/.cache/huggingface
nohup .venv/bin/python -u labo/serveur.py > labo/journaux/serveur.log 2>&1 &
echo "Serveur lancé (journal : labo/journaux/serveur.log)."
echo "Adresse : https://$(tailscale status --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')/"
