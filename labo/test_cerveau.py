"""Brique 2 — test isolé du cerveau + mesure (premier mot, première phrase, total).

Compare plusieurs réglages sur le cas de test ; 3 essais chacun.
"""
import statistics
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from cerveau import phrases_en_flux  # noqa: E402

QUESTION = "Mon eau est verte depuis deux jours, j'ai mis du chlore choc et ça ne change rien."

REGLAGES = {
    "Opus 5.5 (effort low)": dict(modele="claude-opus-5-5", output_config={"effort": "low"}),
    "Sonnet 5.5 (sans réflexion)": dict(modele="claude-sonnet-5-5", thinking={"type": "between_tools"},
                                        output_config={"effort": "low"}),
    "Haiku 5.5 (sans réflexion)": dict(modele="claude-haiku-5-5", thinking={"type": "disabled"},
                                       output_config={"effort": "low"}),
}

client = anthropic.Anthropic()
for nom, reglage in REGLAGES.items():
    essais = []
    for i in range(3):
        mesures = {}
        try:
            phrases = list(phrases_en_flux(client, [{"role": "user", "content": QUESTION}],
                                           mesures=mesures, **reglage))
        except anthropic.APIStatusError as e:
            print(f"{nom} : erreur {e.status_code} — {e.message}")
            break
        essais.append(mesures)
        if i == 0:
            print(f"\n=== {nom} ===\n" + "\n".join(f"  « {p} »" for p in phrases))
    if essais:
        med = {k: statistics.median(e[k] for e in essais) for k in ("premier_mot", "premiere_phrase", "total")}
        print(f"  médiane sur {len(essais)} : 1er mot {med['premier_mot']*1000:.0f} ms | "
              f"1re phrase {med['premiere_phrase']*1000:.0f} ms | total {med['total']*1000:.0f} ms")
