"""Mode « labo » de la conversation vocale : Pierre parle au labo (Claude Code) au lieu de jouer le client.

- Phrase qui commence par « Aquaphone » → mode labo : la demande est notée dans
  `labo/journaux/demandes_vocales.md` ; la voix confirme seulement, Claude Code (qui surveille ce fichier)
  répond par écrit dans le terminal.
- Phrase qui commence par « iopool » (en mode labo) → retour en mode client (assistant du pisciniste).
"""
import re
import time
import unicodedata
from pathlib import Path

JOURNAUX = Path(__file__).resolve().parent / "journaux"
DEMANDES = JOURNAUX / "demandes_vocales.md"

# Variantes de transcription Whisper : « Aquaphone », « Aqua phone », « Aquafone », « iopool », « Io pool », « Yo pool »…
MOT_LABO = re.compile(r"^\W*(?:ok\W+|allo\W+|alors\W+)?a\W*q?\W*[ck]?ua\W*(?:ph|f)one?s?\b\W*", re.I)
MOT_CLIENT = re.compile(r"^\W*(?:ok\W+|alors\W+)?(?:i|y|ya|yi)\W*o\W*pool(?:e|es|s)?\b\W*", re.I)


def _simple(texte):
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def detecter(texte, mode):
    """Rend (nouveau_mode, texte_sans_mot_cle)."""
    s = _simple(texte)
    if (m := MOT_LABO.match(s)):
        return "labo", texte[m.end():].strip()
    if mode == "labo" and (m := MOT_CLIENT.match(s)):
        return "client", texte[m.end():].strip()
    return mode, texte


def noter_demande(texte):
    DEMANDES.parent.mkdir(exist_ok=True)
    nouveau = not DEMANDES.exists()
    with open(DEMANDES, "a", encoding="utf-8") as f:
        if nouveau:
            f.write("# Demandes vocales de Pierre (mode labo)\n\n")
        f.write(f"- [{time.strftime('%Y-%m-%d %H:%M:%S')}] {texte}\n")


def reponse_fixe(ancien, mode, texte):
    """Phrases dites sans passer par Claude (changement de mode, accusé de réception) ; None sinon."""
    if mode == "labo":
        if texte:
            noter_demande(texte)
            return ["C'est noté.", "Claude Code te répond par écrit."]
        return ["Mode labo, je t'écoute."]
    if ancien == "labo" and not texte:
        return ["D'accord, je redeviens l'assistant du pisciniste."]
    return None
