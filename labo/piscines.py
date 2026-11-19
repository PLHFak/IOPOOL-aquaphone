"""Piscine du client, ajoutée au contexte de Claude (choix dans le menu ☰ de la page).

Deux choix :
- « standard » : piscines/standard.md, piscine de démonstration (fictive, sans sonde iopool, eau un peu trouble).
- « dictee »   : piscines/dictee.md, description dictée à la voix sur la page (transcrite par Whisper),
                 aussi détaillée que voulu, complétable par du texte libre. Jamais commitée.
Le choix est dans piscines/choix.txt et s'applique dès la question suivante (pas de redémarrage).
"""
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "piscines"
FICHIERS = {"standard": DOSSIER / "standard.md", "dictee": DOSSIER / "dictee.md"}
CHOIX = DOSSIER / "choix.txt"
EN_TETE_DICTEE = ("# Piscine du client (décrite par lui-même)\n\nDescription donnée par le client, à la voix ou par écrit. "
                  "Sers-t'en pour personnaliser tes réponses, sans la réciter ; ne donne jamais de dose chiffrée.\n")


def choix():
    try:
        c = CHOIX.read_text(encoding="utf-8").strip()
    except OSError:
        c = "standard"
    return c if c in FICHIERS else "standard"


def choisir(c):
    if c not in FICHIERS:
        raise ValueError("choix inconnu")
    DOSSIER.mkdir(exist_ok=True)
    CHOIX.write_text(c, encoding="utf-8")


def texte(c=None):
    try:
        return FICHIERS[c or choix()].read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def ajouter_dictee(contenu, source):
    """Ajoute un passage (dicté ou écrit) à la description de la piscine du client."""
    contenu = contenu.strip()
    if not contenu:
        raise ValueError("description vide")
    DOSSIER.mkdir(exist_ok=True)
    f = FICHIERS["dictee"]
    debut = f.read_text(encoding="utf-8") if f.exists() else EN_TETE_DICTEE
    f.write_text(debut.rstrip() + f"\n\n## {source} ({time.strftime('%d/%m %H:%M')})\n{contenu}\n", encoding="utf-8")


def effacer_dictee():
    """Vide la description dictée (demandé explicitement par l'utilisateur depuis la page)."""
    FICHIERS["dictee"].unlink(missing_ok=True)


def etat():
    return {"choix": choix(), "standard": texte("standard"), "dictee": texte("dictee")}
