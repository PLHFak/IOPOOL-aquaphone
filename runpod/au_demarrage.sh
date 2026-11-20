#!/usr/bin/env bash
# Lancé par le conteneur à sa création (dockerArgs du pod, voir labo-demarrage/ sur Vercel) :
#   bash /workspace/iopool-aquaphone/runpod/au_demarrage.sh; exec /start.sh
# Rend la main tout de suite (SSH de RunPod démarre ensuite) ; en arrière-plan : paquets, service, mise en veille.
J=/workspace/iopool-aquaphone/labo/journaux
mkdir -p "$J"
nohup bash -c "bash /workspace/iopool-aquaphone/runpod/demarrage.sh; bash /workspace/iopool-aquaphone/runpod/veille.sh" \
  >> "$J/au_demarrage.log" 2>&1 &
