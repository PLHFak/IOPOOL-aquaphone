"""Contexte de Marc : base documentaire iopool (dossier `connaissances/`, jamais commité).

La base est la copie interne du support iopool (zip de Pierre, 18/11/2026). Son README impose
d'exclure certains fichiers d'un assistant qui parle directement aux clients :
- `operations/` (procédures internes), contacts de partenaires et prénoms d'employés ;
- doses internes : `internal-dosing-reference.md`, `domain-knowledge.md`, `salt-pools.md` ;
- `eco-sensor-electrochemistry.md` (marqué INTERNE, procédures non validées) ;
- `_reference/` (prix historiques et fiches internes), `orders-delivery/`, `KNOWLEDGE_MAP.md`.
"""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
BASE = RACINE / "connaissances" / "iopool-agent-reference-docs"

INCLUS = [
    "water-chemistry/ranges-by-treatment-type.md",
    "products/eco-probe-master.md",
    "products/ecotab-master.md",
    "products/gateway.md",
    "products/chemistry-packs.md",
    "products/vip-5-year-card.md",
    "products/chemicals-and-consumables-shelf-life.md",
    "troubleshooting/",
    "app-user-guide/",
    "warranty-returns/warranty-rma-runbook.md",
]

CONSIGNES = """
Tu as ci-dessous la documentation de référence d'iopool (en anglais), la sonde connectée et l'application que Marc installe chez ses clients. Sers-t'en pour répondre juste, mais réponds toujours en français parlé, court, sans citer la documentation ni ses noms de fichiers.

Règles impératives sur cette documentation :
- Ne donne jamais de dose de produit chiffrée : c'est l'application iopool qui calcule la dose à partir des mesures et du volume ; renvoie vers sa recommandation.
- Tout passage marqué interne (« INTERNAL », « never quote », « agent knowledge », « not shown in-app ») sert à comprendre, jamais à être répété au client : procédures de support, références internes, scripts, codes, fiches de remplacement, prix internes.
- Ne mentionne aucun prénom d'employé, aucune adresse e-mail autre que hello@iopool.com, aucun code promo.
- Ne promets ni remplacement, ni remboursement, ni geste commercial : propose que Marc ou le support iopool vérifie et rappelle.
- Si la documentation ne couvre pas la question, dis-le simplement et propose de transmettre à Marc.
"""


def fichiers():
    for motif in INCLUS:
        chemin = BASE / motif
        if motif.endswith("/"):
            yield from sorted(chemin.rglob("*.md"))
        elif chemin.exists():
            yield chemin


def documents():
    """Texte de la base à placer (en cache) dans le prompt système ; chaîne vide si absente."""
    if not BASE.exists():
        return ""
    morceaux = [CONSIGNES.strip(), "<documentation_iopool>"]
    for f in fichiers():
        rel = f.relative_to(BASE).as_posix()
        morceaux.append(f'<document chemin="{rel}">\n{f.read_text(encoding="utf-8").strip()}\n</document>')
    morceaux.append("</documentation_iopool>")
    return "\n\n".join(morceaux)


if __name__ == "__main__":
    t = documents()
    print(f"{len(list(fichiers()))} fichiers, {len(t)} caractères (~{len(t) // 4} tokens)")
