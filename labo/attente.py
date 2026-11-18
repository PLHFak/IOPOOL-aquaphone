"""Phrases d'attente (spécification : labo/phrases-attente.md).

Chaque phrase est synthétisée (voix clonée) et animée (visage) une seule fois, puis gardée en mémoire et en
cache disque (`models/attente/`). Pendant que Claude réfléchit, le serveur envoie le clip tout prêt : aucun
calcul, donc il part juste après la transcription (≈ 0,8 à 0,9 s après la fin de parole).

Catégories gardées pour l'instant : « Générale » et « Problème d'eau ». Les catégories « données de la
sonde », « photo » et « escalade » de la spécification attendent les fonctions correspondantes
(sinon l'assistant annoncerait une action qu'il ne fait pas).
"""
import collections
import hashlib
import os
import pickle
import random
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CACHE = RACINE / "models" / "attente"
PISCINISTE_PRENOM = os.environ.get("PISCINISTE_PRENOM", "Marc")

PHRASES = {
    "generale": [
        "Je regarde ça.",
        "D'accord, voyons…",
        "Bonne question, je vérifie.",
        "Un instant, je réfléchis.",
        "Je comprends, laissez-moi regarder.",
        "Hmm, voyons voir.",
    ],
    "eau": [
        "Ah, je vois le genre de souci.",
        "D'accord, on va trouver d'où ça vient.",
        "Je regarde ce qui peut causer ça.",
    ],
}

MOTS_EAU = re.compile(r"\b(vert|verte|verdâtre|trouble|troubles|laiteuse|odeur|sent|algue|algues|mousse|"
                      r"marron|brune|jaune|noire|taches?|dépôts?|sale|opaque)\b", re.I)
POLITESSE = re.compile(r"^\W*(bonjour|bonsoir|salut|merci|au revoir|ok|okay|d'accord|oui|non|parfait|super)\b", re.I)


def toutes():
    return [p for liste in PHRASES.values() for p in liste]


def choisir(texte, recents):
    """Phrase d'attente pour ce tour, ou None (salutation, simple merci…). `recents` : deque(maxlen=3)."""
    if POLITESSE.match(texte) and len(texte.split()) <= 4:
        return None
    categorie = "eau" if MOTS_EAU.search(texte) else "generale"
    candidates = [p for p in PHRASES[categorie] if p not in recents] or PHRASES[categorie]
    phrase = random.choice(candidates)
    recents.append(phrase)
    return phrase


def nouveaux_recents():
    return collections.deque(maxlen=3)


def charger(rendre, reglages):
    """Rend {phrase: [(pcm, jpeg), …]} ; `rendre(phrase)` calcule un clip absent du cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    clips = {}
    for phrase in toutes():
        cle = hashlib.sha1(repr((phrase, reglages)).encode()).hexdigest()[:16]
        fichier = CACHE / f"{cle}.pkl"
        if fichier.exists():
            clips[phrase] = pickle.loads(fichier.read_bytes())
        else:
            clips[phrase] = rendre(phrase)
            fichier.write_bytes(pickle.dumps(clips[phrase]))
    return clips
