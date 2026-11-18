"""Accès à la page par mot de passe (pour une adresse publique, ex. proxy HTTPS de RunPod).

- Mot de passe dans `.env` : AQUAPHONE_MOT_DE_PASSE (jamais dans Git). Vide → pas de mot de passe
  (accès réservé à Tailscale, le serveur reste alors sur 127.0.0.1).
- Majuscules et minuscules indifférentes, espaces autour ignorés.
- Une fois connecté, le navigateur garde un cookie 30 jours (signé ; changer le mot de passe le révoque).
- La page, la photo et la WebSocket (micro, voix, visage) sont toutes protégées.
- Anti-robots : 5 erreurs depuis une même adresse → blocage 15 min ; 30 erreurs en tout sur 1 h → blocage
  de toutes les connexions pendant 1 h (journalisé).
"""
import collections
import hashlib
import hmac
import html
import os
import time

from aiohttp import web

MOT_DE_PASSE = os.environ.get("AQUAPHONE_MOT_DE_PASSE", "")
COOKIE = "aquaphone"
DUREE_COOKIE = 30 * 24 * 3600
LIBRES = {"/connexion", "/note"}  # /note a son propre jeton (labo/note.py, 127.0.0.1)

_SECRET = hashlib.sha256(("aquaphone-acces|" + MOT_DE_PASSE).encode()).digest()
_JETON_COOKIE = hmac.new(_SECRET, b"connecte", hashlib.sha256).hexdigest()

ERREURS_PAR_IP = collections.defaultdict(collections.deque)  # ip -> instants des échecs
ERREURS_TOTALES = collections.deque()
MAX_PAR_IP, FENETRE_IP = 5, 15 * 60
MAX_TOTAL, FENETRE_TOTALE = 30, 3600

PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Aquaphone</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;display:flex;
min-height:100vh;align-items:center;justify-content:center;margin:0}}form{{background:#1e293b;padding:28px;
border-radius:14px;width:min(320px,90vw)}}h1{{font-size:20px;margin:0 0 16px}}input,button{{width:100%;
box-sizing:border-box;font-size:17px;padding:12px;border-radius:8px;border:0;margin-top:10px}}
button{{background:#0ea5e9;color:#fff;font-weight:600}}p{{color:#fca5a5;margin:10px 0 0}}</style></head>
<body><form method="post" action="/connexion"><h1>Aquaphone</h1>
<input type="password" name="mot_de_passe" placeholder="Mot de passe" autocomplete="current-password" autofocus>
<button type="submit">Entrer</button>{message}</form></body></html>"""


def actif():
    return bool(MOT_DE_PASSE)


def _ip(request):
    return (request.headers.get("CF-Connecting-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.remote or "?")


def _purger(d, fenetre, maintenant):
    while d and d[0] < maintenant - fenetre:
        d.popleft()


def _bloque(ip, maintenant):
    _purger(ERREURS_TOTALES, FENETRE_TOTALE, maintenant)
    d = ERREURS_PAR_IP[ip]
    _purger(d, FENETRE_IP, maintenant)
    return len(d) >= MAX_PAR_IP or len(ERREURS_TOTALES) >= MAX_TOTAL


def _connecte(request):
    return hmac.compare_digest(request.cookies.get(COOKIE, ""), _JETON_COOKIE)


def _page(message="", statut=200):
    m = f"<p>{html.escape(message)}</p>" if message else ""
    return web.Response(text=PAGE.format(message=m), content_type="text/html", status=statut,
                        headers={"Cache-Control": "no-store"})


async def connexion(request, log=print):
    maintenant = time.time()
    ip = _ip(request)
    if request.method == "GET":
        return _page()
    if _bloque(ip, maintenant):
        return _page("Trop d'essais. Réessayez plus tard.", 429)
    donnees = await request.post()
    essai = str(donnees.get("mot_de_passe", "")).strip()
    # Majuscules/minuscules indifférentes (saisie sur téléphone) ; le blocage des essais compense.
    if hmac.compare_digest(essai.casefold().encode(), MOT_DE_PASSE.strip().casefold().encode()):
        ERREURS_PAR_IP.pop(ip, None)
        log(f"Connexion réussie depuis {ip}")
        r = web.HTTPFound("/")
        r.set_cookie(COOKIE, _JETON_COOKIE, max_age=DUREE_COOKIE, httponly=True, samesite="Lax",
                     secure=request.secure or request.headers.get("X-Forwarded-Proto") == "https")
        raise r
    ERREURS_PAR_IP[ip].append(maintenant)
    ERREURS_TOTALES.append(maintenant)
    log(f"Mot de passe refusé depuis {ip} ({len(ERREURS_PAR_IP[ip])} échec(s))")
    return _page("Mot de passe incorrect.", 401)


@web.middleware
async def garde(request, handler):
    if not actif() or request.path in LIBRES or _connecte(request):
        return await handler(request)
    if request.path == "/":
        raise web.HTTPFound("/connexion")
    raise web.HTTPUnauthorized()
