#!/usr/bin/env bash
# Installation d'Aquaphone sur un pod RunPod (Linux, 2 cartes graphiques).
# À lancer sur le pod :  bash /workspace/iopool-aquaphone/runpod/installer.sh
# Idempotent : tout est rangé dans /workspace (volume persistant) → la 2e fois, quelques secondes.
set -euo pipefail
W=/workspace
D=$W/iopool-aquaphone
export HF_HOME=$W/.cache/huggingface PIP_CACHE_DIR=$W/.cache/pip

echo "== Paquets système"
apt-get update -qq && apt-get install -y -qq ffmpeg git curl python3-venv >/dev/null

echo "== Code"
[ -d "$D/.git" ] || git clone https://github.com/PLHFak/IOPOOL-aquaphone.git "$D"
cd "$D" && git pull --ff-only || true

echo "== Python (venv dans le dépôt)"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "== Ditto (visage) — même version que sur le PC + correctif de recollage OpenCV"
if [ ! -d third_party/ditto ]; then
  git clone -q https://github.com/antgroup/ditto-talkinghead third_party/ditto
  git -C third_party/ditto checkout -q c3e47ee
  git -C third_party/ditto apply "$D/labo/patches/ditto_blend_numpy.patch"
fi

echo "== Modèles (≈ 8 Go au 1er lancement : Ditto 2,2 Go, voix 1.7B 4,3 Go, Whisper 1,6 Go au démarrage)"
.venv/bin/hf download digital-avatar/ditto-talkinghead --include "ditto_cfg/*" "ditto_pytorch/*" \
  --local-dir models/ditto >/dev/null
.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/qwen3-tts-1.7B >/dev/null

if [ -n "${TS_AUTHKEY:-}" ]; then
  echo "== Tailscale (facultatif : accès privé en plus du mot de passe)"
  command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
  if ! pgrep -x tailscaled >/dev/null; then
    # Pas de périphérique TUN dans un conteneur RunPod : mode « userspace »
    nohup tailscaled --tun=userspace-networking --state="$W/tailscale.state" >"$W/tailscaled.log" 2>&1 &
    sleep 3
  fi
  tailscale status >/dev/null 2>&1 || tailscale up --authkey="$TS_AUTHKEY" --hostname=aquaphone-runpod
  tailscale serve --bg 8765
fi

echo "== Fichiers privés à copier depuis le PC (jamais dans Git) :"
for f in .env assets/visage_pierre.jpg models/ref_pierre.wav models/question_test_16k.wav \
         connaissances/iopool-agent-reference-docs/README.md; do
  [ -e "$f" ] && echo "   ok      $f" || echo "   MANQUE  $f"
done
echo "Ensuite :  bash runpod/lancer.sh"
