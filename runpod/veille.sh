#!/usr/bin/env bash
# Mise en veille automatique, DANS le pod (aucun PC nécessaire) : le pod se supprime lui-même quand personne ne
# s'en sert. Le disque réseau (code, modèles, fichiers) reste. Clé : celle que RunPod donne à chaque pod
# (variable RUNPOD_API_KEY du conteneur), jamais la clé de Pierre.
#   - 15 min pour la 1re connexion après le démarrage ;
#   - ensuite 5 min sans connexion ni question (labo/journaux/activite.json, écrit par le serveur).
ARRET_MIN=${AQUAPHONE_ARRET_MIN:-5}
ATTENTE_1RE_MIN=${AQUAPHONE_ATTENTE_1RE_MIN:-15}
ACT=/workspace/iopool-aquaphone/labo/journaux/activite.json
DEBUT=$(date +%s)
ENV1=$(tr '\0' '\n' < /proc/1/environ)
CLE=$(sed -n 's/^RUNPOD_API_KEY=//p' <<<"$ENV1")
POD=$(sed -n 's/^RUNPOD_POD_ID=//p' <<<"$ENV1")
while sleep 30; do
  MAINT=$(date +%s)
  DERN=$(python3 -c "import json;print(int(json.load(open('$ACT'))['derniere']))" 2>/dev/null || echo 0)
  if [ "$DERN" -gt "$DEBUT" ]; then LIMITE=$((ARRET_MIN * 60)); REF=$DERN; else LIMITE=$((ATTENTE_1RE_MIN * 60)); REF=$DEBUT; fi
  if [ $((MAINT - REF)) -gt "$LIMITE" ]; then
    echo "$(date +%H:%M:%S) aucune utilisation depuis $(((MAINT - REF) / 60)) min → suppression du pod $POD"
    curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $CLE" https://api.runpod.io/graphql \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD\\\"}) }\"}"
    exit 0
  fi
done
