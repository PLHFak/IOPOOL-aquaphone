"""RunPod à la demande, piloté depuis le PC labo (la clé RunPod ne quitte pas le PC).

- Page d'accueil (port 8770, même mot de passe que la page Aquaphone) : état du pod + bouton « Démarrer ».
  Démarrer = reprendre le pod arrêté (le disque /workspace est conservé), lancer le service (runpod/lancer.sh
  par SSH), puis rediriger vers https://<pod>-8765.proxy.runpod.net/ quand il répond (≈ 3 min).
- Surveillance : toutes les 30 s, si le pod tourne, lecture de labo/journaux/activite.json (écrit par le serveur
  à chaque connexion et chaque question). Aucune question ni connexion depuis ARRET_MIN minutes → arrêt du pod
  (on ne paie plus que le stockage). Un onglet resté ouvert sans parler ne le garde pas allumé.

Usage (PC) : .venv\\Scripts\\python runpod\\a_la_demande.py      [AQUAPHONE_ARRET_MIN=5 par défaut]
Accès public : tailscale funnel --bg --https=8443 http://127.0.0.1:8770
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

RACINE = Path(__file__).resolve().parent.parent
load_dotenv(RACINE / ".env")
sys.path.insert(0, str(RACINE / "labo"))
import acces  # noqa: E402  (lit AQUAPHONE_MOT_DE_PASSE, donc après load_dotenv)
from aiohttp import web  # noqa: E402

POD = os.environ.get("AQUAPHONE_POD", "hn5obumrqbyns1")
CLE = os.environ.get("RUNPOD_API_KEY", "")
ARRET_MIN = float(os.environ.get("AQUAPHONE_ARRET_MIN", "5"))
CLE_SSH = str(Path.home() / ".ssh" / "aquaphone_runpod")
PORT = 8770
ADRESSE_POD = f"https://{POD}-8765.proxy.runpod.net/"
ETAT = {"phase": "inconnu", "message": "", "depuis": time.time(), "lance_a": None}


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def graphql(requete):
    r = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": requete}).encode(),
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {CLE}"})
    with urllib.request.urlopen(r, timeout=30) as f:
        d = json.load(f)
    if d.get("errors"):
        raise RuntimeError(d["errors"][0].get("message", "erreur RunPod"))
    return d["data"]


def pod():
    return graphql(f'query {{ pod(input: {{podId: "{POD}"}}) {{ desiredStatus runtime {{ uptimeInSeconds '
                   'ports { ip publicPort privatePort type } } } }')["pod"]


def ssh_adresse(p):
    for x in ((p or {}).get("runtime") or {}).get("ports") or []:
        if x["privatePort"] == 22 and x["type"] == "tcp":
            return x["ip"], x["publicPort"]
    return None


def ssh(adresse, commande, delai=60):
    ip, port = adresse
    return subprocess.run(["ssh", "-i", CLE_SSH, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
                           "-o", "ConnectTimeout=10", f"root@{ip}", commande],
                          capture_output=True, text=True, timeout=delai)


def service_repond():
    try:
        with urllib.request.urlopen(ADRESSE_POD + "connexion", timeout=10) as f:
            return f.status == 200
    except Exception:
        return False


def phase(p, message=""):
    if ETAT["phase"] != p:
        journal(f"{p} {message}")
    ETAT.update(phase=p, message=message, depuis=time.time())


def demarrer():
    """Reprend le pod, lance le service, attend qu'il réponde (fil séparé)."""
    try:
        phase("demarrage", "réveil du pod (cartes graphiques)…")
        graphql(f'mutation {{ podResume(input: {{podId: "{POD}", gpuCount: 2}}) {{ id desiredStatus }} }}')
        for _ in range(60):  # ≤ 5 min pour obtenir la machine et le SSH
            adr = ssh_adresse(pod())
            if adr and ssh(adr, "true", 20).returncode == 0:
                break
            time.sleep(5)
        else:
            raise RuntimeError("le pod ne répond pas (cartes graphiques indisponibles ?)")
        phase("demarrage", "chargement de la voix et du visage (≈ 2 min)…")
        r = ssh(adr, "cd /workspace/iopool-aquaphone && bash runpod/lancer.sh", 60)
        if r.returncode:
            raise RuntimeError("lancement du service : " + (r.stderr or r.stdout)[-300:])
        ETAT["lance_a"] = time.time()
        for _ in range(72):  # ≤ 6 min
            if service_repond() and "Serveur prêt" in ssh(adr, "grep -a 'Serveur prêt' "
                                                           "/workspace/iopool-aquaphone/labo/journaux/serveur.log | tail -1",
                                                           20).stdout:
                phase("pret")
                return
            time.sleep(5)
        raise RuntimeError("le service ne répond pas après 6 min")
    except Exception as e:
        phase("erreur", str(e))


async def surveiller():
    """Arrêt du pod après ARRET_MIN minutes sans connexion ni question."""
    while True:
        try:
            p = await asyncio.to_thread(pod)
            if p["desiredStatus"] != "RUNNING":
                if ETAT["phase"] != "demarrage":
                    phase("arrete")
            elif ETAT["phase"] not in ("demarrage",):
                if ETAT["lance_a"] is None:  # déjà allumé au lancement de ce programme : délai compté d'ici
                    ETAT["lance_a"] = time.time()
                    phase("pret" if await asyncio.to_thread(service_repond) else "allume")
                adr = ssh_adresse(p)
                derniere = ETAT["lance_a"]
                if adr:
                    r = await asyncio.to_thread(ssh, adr, "cat /workspace/iopool-aquaphone/labo/journaux/activite.json", 20)
                    try:
                        derniere = max(derniere, json.loads(r.stdout)["derniere"])
                    except (ValueError, KeyError):
                        pass
                inactif = time.time() - derniere
                ETAT["inactif_s"] = round(inactif)
                if inactif > ARRET_MIN * 60:
                    journal(f"aucune utilisation depuis {inactif / 60:.1f} min → arrêt du pod")
                    await asyncio.to_thread(graphql, f'mutation {{ podStop(input: {{podId: "{POD}"}}) {{ id desiredStatus }} }}')
                    ETAT["lance_a"] = None
                    phase("arrete", f"arrêté après {ARRET_MIN:g} min sans utilisation")
        except Exception as e:
            journal("surveillance :", e)
        await asyncio.sleep(30)


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Aquaphone</title>
<style>body{font-family:-apple-system,system-ui,sans-serif;background:#0f172a;color:#e2e8f0;display:flex;
min-height:100vh;align-items:center;justify-content:center;margin:0}main{background:#1e293b;padding:28px;
border-radius:14px;width:min(340px,90vw);text-align:center}button{width:100%;font-size:18px;padding:14px;
border-radius:10px;border:0;background:#0ea5e9;color:#fff;font-weight:600;margin-top:14px}
button:disabled{opacity:.4}p{color:#94a3b8}</style></head><body><main><h1>Aquaphone</h1>
<p id="etat">…</p><button id="go" disabled>Démarrer l'assistant</button></main><script>
const go = document.getElementById('go'), etat = document.getElementById('etat');
const TEXTES = {arrete: "L'assistant est en veille.", demarrage: 'Démarrage en cours', pret: 'Prêt !',
                allume: 'Allumé, service en cours de lancement…', erreur: 'Erreur : ', inconnu: 'Vérification…'};
async function suivre() {
  const d = await (await fetch('etat', {cache: 'no-store'})).json();
  etat.textContent = TEXTES[d.phase] + (d.message ? ' — ' + d.message : '');
  go.disabled = !(d.phase === 'arrete' || d.phase === 'erreur');
  if (d.phase === 'pret') { location.href = d.adresse; return; }
  setTimeout(suivre, 3000);
}
go.onclick = async () => { go.disabled = true; await fetch('demarrer', {method: 'POST'}); };
suivre();
</script></body></html>"""


async def accueil(request):
    return web.Response(text=PAGE, content_type="text/html")


async def etat(request):
    return web.json_response({**ETAT, "adresse": ADRESSE_POD, "arret_min": ARRET_MIN})


async def bouton(request):
    if ETAT["phase"] in ("arrete", "erreur"):
        ETAT["phase"] = "demarrage"
        asyncio.get_running_loop().run_in_executor(None, demarrer)
    return web.json_response(ETAT)


async def connexion(request):
    return await acces.connexion(request, journal)


async def lancer(app):
    app["surveillance"] = asyncio.create_task(surveiller())


def main():
    if not CLE:
        sys.exit("RUNPOD_API_KEY manquante dans .env")
    if not acces.actif():
        sys.exit("AQUAPHONE_MOT_DE_PASSE manquant dans .env : page publique refusée")
    app = web.Application(middlewares=[acces.garde])
    app.add_routes([web.get("/", accueil), web.get("/etat", etat), web.post("/demarrer", bouton),
                    web.get("/connexion", connexion), web.post("/connexion", connexion)])
    app.on_startup.append(lancer)
    journal(f"RunPod à la demande : pod {POD}, arrêt après {ARRET_MIN:g} min sans utilisation — http://127.0.0.1:{PORT}/")
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)


if __name__ == "__main__":
    main()
