"""Test de bout en bout sans navigateur : joue le cas de test au serveur comme un micro (temps réel),
reçoit le visage + la voix, simule la lecture, affiche les délais et enregistre la vidéo reçue.

Usage : python labo/test_pipeline.py [nombre_d_essais]   (serveur lancé : python labo/serveur.py)
"""
import asyncio
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

RACINE = Path(__file__).resolve().parent.parent
URL = "ws://127.0.0.1:8765/ws"
TAMPON = 0.15
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SORTIE = RACINE / "models" / "sorties_pipeline"


async def essai(i):
    audio, _ = sf.read(RACINE / "models" / "question_test_16k.wav", dtype="float32")
    pcm = (np.concatenate([np.zeros(8000, np.float32), audio]) * 32767).astype("<i2")
    images, sons, mesures = [], [], None
    async with aiohttp.ClientSession() as s, s.ws_connect(URL, max_msg_size=0) as ws:
        fin_reponse = asyncio.Event()

        async def micro():  # 20 ms toutes les 20 ms, puis silence tant que la réponse n'est pas finie
            t0 = time.perf_counter()
            k = 0
            while not fin_reponse.is_set():
                bloc = pcm[k * 320:(k + 1) * 320]
                if len(bloc) < 320:
                    bloc = np.zeros(320, "<i2")
                await ws.send_bytes(bloc.tobytes())
                k += 1
                await asyncio.sleep(max(0, t0 + k * 0.02 - time.perf_counter()))

        tache = asyncio.create_task(micro())
        t_lecture = None
        lecture, coupures, trou = 0.0, 0, 0.0  # même logique de lecture que la page web
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                n, rid = struct.unpack_from("<HI", msg.data, 2)
                if t_lecture is None:
                    t_lecture = time.perf_counter() + TAMPON
                    asyncio.get_running_loop().call_later(
                        TAMPON, lambda r=rid: asyncio.ensure_future(ws.send_str(json.dumps({"type": "lecture", "rid": r}))))
                now = time.perf_counter()
                if lecture < now + 0.02:
                    if lecture:
                        coupures += 1
                        trou += now + TAMPON - lecture
                    lecture = now + TAMPON
                lecture += n / 24000
                sons.append(np.frombuffer(msg.data, "<i2", n, 12))
                images.append(msg.data[12 + 2 * n:])
            else:
                d = json.loads(msg.data)
                if d["type"] in ("transcription", "phrase", "erreur"):
                    print(f"  [{d['type']}] {d.get('texte')}")
                elif d["type"] == "fin":
                    duree = sum(len(x) for x in sons) / 24000
                    await asyncio.sleep(max(0, t_lecture + duree - time.perf_counter()) + 0.3)
                    await ws.send_str(json.dumps({"type": "pret", "rid": d["rid"]}))
                elif d["type"] == "mesures":
                    mesures = {**d, "coupures": coupures, "duree_coupures_ms": round(trou * 1000)}
                    fin_reponse.set()
                    break
        tache.cancel()

    # vidéo reçue (pour vérifier la synchro lèvres/son)
    dossier = SORTIE / f"essai{i}"
    dossier.mkdir(parents=True, exist_ok=True)
    for k, jpeg in enumerate(images):
        (dossier / f"{k:05d}.jpg").write_bytes(jpeg)
    sf.write(dossier / "son.wav", np.concatenate(sons), 24000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", str(dossier / "%05d.jpg"),
                    "-i", str(dossier / "son.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(SORTIE / f"essai{i}.mp4")], check=False)
    return mesures, len(images)


async def main():
    for i in range(1, N + 1):
        print(f"Essai {i} :")
        m, n = await essai(i)
        print(f"  {n} images reçues")
        print("  " + " | ".join(f"{k} {v}" for k, v in m.items() if k not in ("type", "question", "reponse", "date")))


asyncio.run(main())
