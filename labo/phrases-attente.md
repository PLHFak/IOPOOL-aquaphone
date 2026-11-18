# Phrases d'attente — masquer le délai de réponse

## Objectif

Le client ne doit jamais rester plus d'environ **0,8 s** devant un visage muet après avoir fini de parler.
Pendant que Claude réfléchit et que l'animation se prépare, l'assistant dit une courte phrase naturelle, **préparée à l'avance**.

## Principe technique

1. **Pré-calcul au démarrage du serveur** : chaque phrase ci-dessous est synthétisée une fois avec la voix clonée (Qwen3-TTS) et animée avec le visage (Ditto). Audio et images sont gardés en mémoire ou en cache disque, prêts à être joués instantanément.
2. **Déclenchement** : dès que la fin de parole est détectée, on lance un minuteur.
   - Si la première phrase de la vraie réponse est prête avant **0,8 s** → pas de phrase d'attente.
   - Sinon → jouer une phrase d'attente pré-calculée.
3. **Enchaînement** : la vraie réponse démarre dès la fin de la phrase d'attente, sans couper un mot ni laisser de blanc.
4. **Pendant les silences** : le visage garde une animation de repos (respiration légère, clignements) ; jamais une photo figée.

## Règles de choix

- Choisir la catégorie selon le contexte (voir ci-dessous) ; à défaut, la catégorie « Générale ».
- Tirage au hasard **sans répétition** sur les 3 derniers tours.
- **Une seule** phrase d'attente par tour, jamais deux d'affilée.
- Pas de phrase d'attente pour une salutation ou un simple « merci ».
- Ton : calme, aimable, naturel ; phrases de **1 à 1,5 s** maximum.

## Phrases

### Générale
- « Je regarde ça. »
- « D'accord, voyons… »
- « Bonne question, je vérifie. »
- « Un instant, je réfléchis. »
- « Je comprends, laissez-moi regarder. »
- « Hmm, voyons voir. »

### Problème d'eau (eau verte, trouble, odeur…)
- « Ah, je vois le genre de souci. »
- « D'accord, on va trouver d'où ça vient. »
- « Je regarde ce qui peut causer ça. »

### Consultation des données de la piscine (sonde iopool)
- « Je consulte les mesures de votre piscine. »
- « Je jette un œil à vos dernières valeurs. »

### Analyse d'une photo (bandelette, eau, équipement)
- « Je regarde votre photo. »
- « Merci pour la photo, je l'analyse. »

### Escalade vers le pisciniste
- « Je préviens Marc tout de suite. »

## Variable « Marc »

Le prénom du pisciniste est une variable de configuration (`PISCINISTE_PRENOM`).
Les phrases qui le contiennent sont pré-calculées pour le prénom configuré.

## Mesures à ajouter dans `labo/resultats.md`

- Fréquence de déclenchement des phrases d'attente (% des tours).
- Délai perçu : fin de parole du client → **premier son** (avec et sans phrase d'attente).
- Ressenti de Pierre : naturel / répétitif / gênant.
