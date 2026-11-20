"""RunPod à la demande, piloté depuis le PC labo (la clé RunPod ne quitte pas le PC).

Disque réseau RunPod « aquaphone » (EU-RO-1, 60 Go) : code, venv, modèles, fichiers privés. Il reste ; les pods,
eux, sont jetables : créés au démarrage sur la 1re carte libre du centre, supprimés après inactivité.

- Page d'accueil (port 8770, même mot de passe que la page Aquaphone) : état + bouton « Démarrer ».
  Démarrer = créer un pod sur le disque (2 × RTX 4090, sinon 1 × 4090, sinon 1 × A100), lancer
  runpod/demarrage.sh par SSH, puis rediriger vers https://<pod>-8765.proxy.runpod.net/ (≈ 3-4 min).
- Surveillance : toutes les 30 s, lecture de labo/journaux/activite.json (écrit par le serveur à chaque connexion
  et chaque question). Rien depuis ARRET_MIN minutes → pod supprimé (le disque réseau reste, seul facturé).
  Un onglet resté ouvert sans parler ne garde pas le pod.

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

CLE = os.environ.get("RUNPOD_API_KEY", "")
ARRET_MIN = float(os.environ.get("AQUAPHONE_ARRET_MIN", "5"))
ATTENTE_1RE_MIN = 15  # après le démarrage, délai laissé pour la 1re connexion (ensuite : ARRET_MIN)
VOLUME, CENTRE, NOM = "hm9zp32n6a", "EU-RO-1", "aquaphone"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# Par ordre de préférence (prix/h Secure Cloud) ; la 1re libre est prise. Blackwell (PRO 4500, 5090, PRO 6000) :
# torch cu128 du venv les gère ; à surveiller dans le journal du serveur à leur 1re utilisation.
CARTES = [("NVIDIA GeForce RTX 4090", 2), ("NVIDIA GeForce RTX 4090", 1), ("NVIDIA RTX PRO 4500 Blackwell", 1),
          ("NVIDIA GeForce RTX 5090", 1), ("NVIDIA A100 80GB PCIe", 1), ("NVIDIA A100-SXM4-80GB", 1),
          ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 1)]
CLE_SSH = Path.home() / ".ssh" / "aquaphone_runpod"
PORT = 8770
ETAT = {"phase": "inconnu", "message": "", "depuis": time.time(), "lance_a": None, "adresse": None, "pod": None}


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def graphql(requete, variables=None):
    r = urllib.request.Request("https://api.runpod.io/graphql",
                               data=json.dumps({"query": requete, "variables": variables or {}}).encode(),
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {CLE}"})
    with urllib.request.urlopen(r, timeout=30) as f:
        d = json.load(f)
    if d.get("errors"):
        raise RuntimeError(d["errors"][0].get("message", "erreur RunPod"))
    return d["data"]


def pod_actuel():
    """Le pod « aquaphone » en service (ou None)."""
    pods = graphql("query { myself { pods { id name desiredStatus machine { gpuDisplayName } "
                   "runtime { uptimeInSeconds gpus { id } ports { ip publicPort privatePort type } } } } }")["myself"]["pods"]
    return next((p for p in pods if p["name"].startswith(NOM) and p["desiredStatus"] == "RUNNING"
                 and p["id"] != "hn5obumrqbyns1"), None)  # ancien pod sans disque réseau : jamais piloté ici


def creer_pod():
    q = ("mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) "
         "{ id machine { gpuDisplayName } } }")
    erreurs = []
    for carte, n in CARTES:
        i = {"name": NOM, "cloudType": "SECURE", "gpuCount": n, "gpuTypeId": carte, "dataCenterId": CENTRE,
             "networkVolumeId": VOLUME, "volumeMountPath": "/workspace", "containerDiskInGb": 20, "imageName": IMAGE,
             "ports": "22/tcp,8765/http", "startSsh": True,
             "env": [{"key": "PUBLIC_KEY", "value": CLE_SSH.with_suffix(".pub").read_text().strip()}]}
        try:
            p = graphql(q, {"i": i})["podFindAndDeployOnDemand"]
            journal(f"pod {p['id']} créé : {n} × {carte}")
            return p["id"], f"{n} × {p['machine']['gpuDisplayName'] if p.get('machine') else carte}"
        except RuntimeError as e:
            erreurs.append(f"{n} × {carte} : {e}")
    raise RuntimeError("aucune carte libre à " + CENTRE + " (" + " ; ".join(erreurs)[:300] + ")")


def ssh_adresse(p):
    for x in ((p or {}).get("runtime") or {}).get("ports") or []:
        if x["privatePort"] == 22 and x["type"] == "tcp":
            return x["ip"], x["publicPort"]
    return None


def ssh(adresse, commande, delai=60):
    ip, port = adresse
    return subprocess.run(["ssh", "-i", str(CLE_SSH), "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
                           "-o", "ConnectTimeout=10", f"root@{ip}", commande],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=delai)


def service_repond(adresse):
    try:
        r = urllib.request.Request(adresse + "connexion", headers={"User-Agent": "Mozilla/5.0 aquaphone"})
        with urllib.request.urlopen(r, timeout=10) as f:  # le proxy RunPod refuse l'agent Python par défaut
            return f.status == 200
    except Exception:
        return False


def phase(p, message=""):
    if (ETAT["phase"], ETAT["message"]) != (p, message):
        journal(f"{p} {message}")
    ETAT.update(phase=p, message=message, depuis=time.time())


def demarrer():
    """Crée le pod (ou reprend celui qui tourne), lance le service, attend qu'il réponde (fil séparé)."""
    try:
        phase("demarrage", "recherche d'une carte graphique libre…")
        p = pod_actuel()
        if p is None:
            pid, cartes = creer_pod()
            phase("demarrage", f"machine obtenue ({cartes}), mise en route…")
        else:
            pid = p["id"]
        ETAT.update(pod=pid, adresse=f"https://{pid}-8765.proxy.runpod.net/")
        for _ in range(72):  # ≤ 6 min pour le conteneur et le SSH
            adr = ssh_adresse(pod_actuel())
            if adr and ssh(adr, "true", 20).returncode == 0:
                break
            time.sleep(5)
        else:
            raise RuntimeError("le pod ne répond pas en SSH")
        phase("demarrage", "chargement de la voix et du visage (≈ 2 min)…")
        r = ssh(adr, "bash /workspace/iopool-aquaphone/runpod/demarrage.sh", 300)
        if r.returncode:
            raise RuntimeError("lancement du service : " + (r.stderr or r.stdout)[-300:])
        ETAT["lance_a"] = time.time()
        for _ in range(120):  # ≤ 10 min ; « Serveur pr » en ASCII (les accents ne passent pas tels quels par SSH)
            pret = "Serveur pr" in ssh(adr, "grep -a 'Serveur pr' /workspace/iopool-aquaphone/labo/journaux/"
                                              "serveur.log | tail -1", 20).stdout
            if pret and service_repond(ETAT["adresse"]):
                phase("pret")
                return
            time.sleep(5)
        fin = ssh(adr, "tail -c 1500 /workspace/iopool-aquaphone/labo/journaux/serveur.log", 20).stdout
        journal("fin du journal du serveur :", fin[-1500:])
        raise RuntimeError("le service ne répond pas après 10 min")
    except Exception as e:
        phase("erreur", str(e))


async def surveiller():
    """Suppression du pod après ARRET_MIN minutes sans connexion ni question (le disque réseau reste)."""
    while True:
        try:
            p = await asyncio.to_thread(pod_actuel)
            if p is None:
                if ETAT["phase"] not in ("demarrage", "erreur"):
                    phase("arrete", ETAT["message"] if ETAT["phase"] == "arrete" else "")
            elif ETAT["phase"] != "demarrage":
                if ETAT["lance_a"] is None:  # déjà allumé au lancement de ce programme : délai compté d'ici
                    ETAT.update(lance_a=time.time(), pod=p["id"], adresse=f"https://{p['id']}-8765.proxy.runpod.net/")
                    phase("pret" if await asyncio.to_thread(service_repond, ETAT["adresse"]) else "allume")
                derniere = ETAT["lance_a"]
                adr = ssh_adresse(p)
                if adr:
                    r = await asyncio.to_thread(ssh, adr, "cat /workspace/iopool-aquaphone/labo/journaux/activite.json", 20)
                    try:
                        derniere = max(derniere, json.loads(r.stdout)["derniere"])
                    except (ValueError, KeyError):
                        pass
                inactif = time.time() - derniere
                ETAT["inactif_s"] = round(inactif)
                connu = adr and r.stdout.strip().startswith("{")  # quelqu'un s'est déjà connecté depuis ce démarrage ?
                limite = ARRET_MIN if connu and derniere > ETAT["lance_a"] else ATTENTE_1RE_MIN
                if inactif > limite * 60:
                    journal(f"aucune utilisation depuis {inactif / 60:.1f} min → pod {p['id']} supprimé (disque conservé)")
                    await asyncio.to_thread(graphql, f'mutation {{ podTerminate(input: {{podId: "{p["id"]}"}}) }}')
                    ETAT.update(lance_a=None, pod=None, adresse=None)
                    phase("arrete", f"mis en veille après {ARRET_MIN:g} min sans utilisation")
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
const TEXTES = {arrete: "L'assistant est en veille.", demarrage: 'Démarrage en cours (≈ 4 min)', pret: 'Prêt !',
                allume: 'Allumé, service en cours de lancement…', erreur: 'Erreur', inconnu: 'Vérification…'};
async function suivre() {
  try {
    const d = await (await fetch('etat', {cache: 'no-store'})).json();
    etat.textContent = TEXTES[d.phase] + (d.message ? ' — ' + d.message : '');
    go.disabled = !(d.phase === 'arrete' || d.phase === 'erreur');
    if (d.phase === 'pret' && d.adresse) { location.href = d.adresse; return; }
  } catch (e) {}
  setTimeout(suivre, 3000);
}
go.onclick = async () => { go.disabled = true; await fetch('demarrer', {method: 'POST'}); };
suivre();
</script></body></html>"""


async def accueil(request):
    return web.Response(text=PAGE, content_type="text/html")


async def etat(request):
    return web.json_response({**ETAT, "arret_min": ARRET_MIN})


async def bouton(request):
    if ETAT["phase"] in ("arrete", "erreur", "inconnu"):
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
    journal(f"RunPod à la demande : disque {VOLUME} ({CENTRE}), arrêt après {ARRET_MIN:g} min sans utilisation "
            f"— http://127.0.0.1:{PORT}/")
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)


if __name__ == "__main__":
    main()
