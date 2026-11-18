"""Brique 2 — Cerveau : Claude en streaming, découpé en phrases pour la synthèse vocale."""
import re
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODELE_PAR_DEFAUT = "claude-sonnet-5-5"
# Sans réflexion + effort bas : 1re phrase en ~0,8 s (voir labo/resultats.md)
OPTIONS_PAR_DEFAUT = {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}}

PROMPT_SYSTEME = """Tu es l'assistant vocal de Marc, pisciniste. Tu parles à un client de Marc, au téléphone.

- Tu es une intelligence artificielle : dis-le simplement si on te le demande, et ne prétends jamais être Marc.
- Tes réponses sont prononcées à voix haute : phrases courtes, langage parlé, pas de liste, pas de markdown, pas d'émoticône. Écris les nombres et unités comme on les dit (« sept virgule deux », « deux cents grammes »).
- Deux à quatre phrases au maximum, puis laisse le client répondre. Pose une seule question à la fois.
- Reformule la demande en une phrase seulement si elle est ambiguë ; sinon réponds directement.
- Sécurité : ne donne jamais de dosage chimique précis sans connaître le volume du bassin et les mesures de l'eau ; ne propose jamais de mélanger des produits.
- Si tu ne sais pas ou si la situation dépasse un conseil simple, propose de transmettre à Marc, qui rappellera le client."""

FIN_DE_PHRASE = re.compile(r"(.+?[.!?…])(\s+|$)", re.S)


def systeme(documents=""):
    """Prompt système ; la documentation (longue, fixe) est mise en cache 1 h côté API."""
    if not documents:
        return PROMPT_SYSTEME
    return [{"type": "text", "text": PROMPT_SYSTEME},
            {"type": "text", "text": documents, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def phrases_en_flux(client, historique, modele=MODELE_PAR_DEFAUT, mesures=None, documents="", **options):
    """Génère la réponse phrase par phrase dès que chacune est complète.

    `mesures` (dict) reçoit : premier_mot, premiere_phrase, total (secondes depuis l'appel).
    """
    options = options or OPTIONS_PAR_DEFAUT
    t0 = time.perf_counter()
    tampon = ""
    with client.messages.stream(
        model=modele, max_tokens=1024, system=systeme(documents), messages=historique, **options
    ) as flux:
        for morceau in flux.text_stream:
            if mesures is not None and "premier_mot" not in mesures:
                mesures["premier_mot"] = time.perf_counter() - t0
            tampon += morceau
            while (m := FIN_DE_PHRASE.match(tampon)) and m.end(2) > m.end(1):
                if mesures is not None and "premiere_phrase" not in mesures:
                    mesures["premiere_phrase"] = time.perf_counter() - t0
                yield m.group(1).strip()
                tampon = tampon[m.end():]
        reponse = flux.get_final_message()
    if tampon.strip():
        if mesures is not None and "premiere_phrase" not in mesures:
            mesures["premiere_phrase"] = time.perf_counter() - t0
        yield tampon.strip()
    if mesures is not None:
        mesures["total"] = time.perf_counter() - t0
        mesures["stop_reason"] = reponse.stop_reason
        mesures["usage"] = reponse.usage
