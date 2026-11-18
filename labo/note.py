"""Affiche une réponse écrite de Claude Code sur la page web (mode labo).

Usage : python labo/note.py "texte"      (ou le texte sur l'entrée standard)
"""
import sys
import urllib.request
from pathlib import Path

JETON = (Path(__file__).resolve().parent / "journaux" / "jeton_note.txt").read_text(encoding="utf-8").strip()
texte = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
req = urllib.request.Request("http://127.0.0.1:8765/note", data=texte.encode("utf-8"),
                             headers={"X-Jeton": JETON, "Content-Type": "text/plain; charset=utf-8"})
print(urllib.request.urlopen(req, timeout=5).read().decode())
