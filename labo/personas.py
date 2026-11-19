"""Personnages de l'assistant (visage + voix), choisis depuis le menu de la page.

- « pierre » : photo et voix de Pierre (models/visage_pierre_*.jpg, models/ref_pierre.wav).
- « pisciniste-fictif » : personne qui n'existe pas (visage RealVisXL, voix Qwen3-TTS VoiceDesign).
- « place-1 » à « place-5 » : emplacements où une personne dépose SA photo et SA voix, avec son accord
  explicite (case à cocher obligatoire, date enregistrée dans fiche.json).

Fichiers dans personas/<nom>/ : fiche.json, visage.jpg, voix.wav (photos et voix jamais commitées).
Le personnage actif est écrit dans personas/actif.txt ; en changer redémarre le serveur (≈ 2 min).
"""
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "personas"
ACTIF = DOSSIER / "actif.txt"
PLACES = [f"place-{i}" for i in range(1, 6)]
TEXTE_A_LIRE = ("Bonjour, je suis l'assistant de votre pisciniste. Dites-moi ce qui se passe avec votre piscine : "
                "l'eau est-elle claire, verte ou trouble ? On va regarder ça ensemble, tranquillement.")
CONSENTEMENT = ("Je certifie être la personne de la photo et de l'enregistrement, et j'accepte qu'ils servent "
                "à animer un assistant IA de démonstration pendant le labo Aquaphone (suppression à la fin du labo).")


def actif(defaut="pierre"):
    try:
        nom = ACTIF.read_text(encoding="utf-8").strip()
    except OSError:
        return defaut
    return nom if nom and pret(nom) else defaut


def choisir(nom):
    DOSSIER.mkdir(exist_ok=True)
    ACTIF.write_text(nom, encoding="utf-8")


def fiche(nom):
    try:
        return json.loads((DOSSIER / nom / "fiche.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def photo(nom):
    if nom == "pierre":
        return RACINE / "models" / "visage_pierre_720.jpg"
    return DOSSIER / nom / "visage.jpg"


def pret(nom):
    if nom == "pierre":
        return (RACINE / "models" / "ref_pierre.wav").exists() and photo(nom).exists()
    d = DOSSIER / nom
    return (d / "visage.jpg").exists() and (d / "voix.wav").exists() and bool(fiche(nom).get("texte_reference"))


def liste(courant):
    noms = ["pierre", "pisciniste-fictif"] + PLACES
    res = []
    for nom in noms:
        f = fiche(nom)
        libelle = {"pierre": "Pierre", "pisciniste-fictif": "Pisciniste imaginaire"}.get(nom) \
            or f.get("prenom") or f"Place libre {nom[-1]}"
        res.append({"nom": nom, "libelle": libelle, "pret": pret(nom), "actif": nom == courant,
                    "place": nom in PLACES, "fictif": bool(f.get("fictif"))})
    return res


def enregistrer(nom, prenom, image, audio, transcrire):
    """Dépôt d'une personne dans une place : photo (octets), voix (octets, tout format), accord déjà vérifié.

    `transcrire(chemin_wav_16k)` rend le texte réellement prononcé (référence du clonage de voix).
    """
    if nom not in PLACES:
        raise ValueError("emplacement inconnu")
    d = DOSSIER / nom
    d.mkdir(parents=True, exist_ok=True)
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("photo illisible")
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        k = 1600 / max(h, w)
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    for ancien in d.glob("visage_*.jpg"):  # versions réduites de l'ancienne photo
        ancien.unlink()
    cv2.imwrite(str(d / "visage.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    brut = d / "voix_depot.bin"
    brut.write_bytes(audio)
    for sortie, sr in ((d / "voix.wav", 24000), (d / "voix_16k.wav", 16000)):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(brut), "-ac", "1", "-ar", str(sr),
                        "-t", "20", "-af", "silenceremove=start_periods=1:start_threshold=-45dB", str(sortie)],
                       check=True)
    brut.unlink()
    texte = transcrire(str(d / "voix_16k.wav"))
    if len(texte.split()) < 5:
        raise ValueError("enregistrement trop court ou inaudible : lisez le texte proposé (10 à 15 s)")
    (d / "fiche.json").write_text(json.dumps({
        "nom": nom, "prenom": prenom.strip()[:40] or nom, "fictif": False, "texte_reference": texte,
        "consentement": CONSENTEMENT, "consentement_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return texte
