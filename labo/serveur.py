"""Briques 5 et 6 — Pipeline de bout en bout + page web de test.

Navigateur ──(micro, PCM 16 kHz, WebSocket)──▶ VAD Silero ▶ faster-whisper ▶ Sonnet (flux, par phrase)
▶ Qwen3-TTS voix clonée (flux, 24 kHz) ▶ Ditto (flux, 25 images/s)
──(paquets « 1 image JPEG + ses 40 ms d'audio »)──▶ navigateur (son et image synchronisés)

Le serveur n'écoute que sur 127.0.0.1 ; l'accès depuis le Mac/iPhone passe par `tailscale serve`
(HTTPS, obligatoire pour le micro dans Safari), sans rien ouvrir dans le pare-feu.

Usage : python labo/serveur.py      puis ouvrir https://<nom-du-pc>.<tailnet>.ts.net/
Journal des délais : labo/journaux/pipeline.jsonl
"""
import asyncio
import collections
import itertools
import json
import os
import queue
import struct
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_win  # noqa: F401,E402  (DLL CUDA 12 pour CTranslate2 et onnxruntime)

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import anthropic  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402
import torch  # noqa: E402
from aiohttp import WSMsgType, web  # noqa: E402
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.vad import get_assets_path  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import cerveau  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "third_party" / "ditto"))
from stream_pipeline_online import StreamSDK  # noqa: E402

# ─── Réglages ────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("AQUAPHONE_PORT", "8765"))
WEB = Path(__file__).resolve().parent / "web"
JOURNAL = Path(__file__).resolve().parent / "journaux" / "pipeline.jsonl"

# Détection de fin de parole (Silero, morceaux de 32 ms)
VAD_SEUIL = 0.5
SILENCE_FIN = float(os.environ.get("AQUAPHONE_SILENCE", "0.5"))  # silence qui clôt la phrase (s)
PAROLE_MIN = 0.3  # en dessous : bruit, ignoré (s)
PRE_ROLL = 0.3  # audio gardé avant le début détecté (s)
PROMPT_STT = "Un client parle à l'assistant de son pisciniste : piscine, chlore choc, pH, iopool."

# Voix
REF_AUDIO = str(RACINE / "models" / "ref_pierre.wav")
REF_TEXTE = (
    "Bonjour, je m'appelle Pierre L'Hoest, je suis votre pisciniste, ici c'est mon clone "
    "qui vous répond, mais je pense que je peux vous répondre correctement, surtout si vous "
    "avez une sonde iopool dans votre piscine, je peux intégrer ces données dans notre "
    "discussion et ce sera bien."
)
TTS_CHUNK = 4

# Visage : 1 image = 40 ms = 960 échantillons à 24 kHz = 640 à 16 kHz
IPS = 25
ECH_IMAGE = 24000 // IPS
BLOC_IMAGES = 5
BLOC_24 = BLOC_IMAGES * ECH_IMAGE
FENETRE_16 = (3 + 5 + 2) * 640 + 80  # fenêtre HuBERT : 3 images passées, 5 courantes, 2 futures
PAS_16 = BLOC_IMAGES * 640
BLOCS_VIDANGE = 4  # silence ajouté en fin de réponse pour sortir les dernières images
AVANCE_MIN = 0.8  # s d'audio d'avance sous laquelle on insère un silence (évite les coupures)
LARGEUR_VIDEO = 540
QUALITE_JPEG = 80
SOURCE = RACINE / "models" / "visage_pierre_960.jpg"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ─── Écoute : VAD Silero en flux ─────────────────────────────────────────────
class Vad:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
            providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.contexte = np.zeros(64, np.float32)

    def __call__(self, morceau):
        x = np.concatenate([self.contexte, morceau])[None].astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.contexte = morceau[-64:]
        return float(np.asarray(out).reshape(-1)[0])


class Ecoute:
    """Reçoit le micro par petits morceaux ; rend (audio de la phrase, instant de fin de parole)."""

    def __init__(self):
        self.vad = Vad()
        self.reset()

    def reset(self):
        self.vad.reset()
        self.reste = np.zeros(0, np.float32)
        self.avant = collections.deque(maxlen=int(PRE_ROLL * 16000 / 512))
        self.phrase = []
        self.n_parole = self.n_silence = 0
        self.t_fin = None

    @property
    def en_parole(self):
        return bool(self.phrase)

    def ajouter(self, pcm):
        self.reste = np.concatenate([self.reste, pcm])
        resultat = None
        while len(self.reste) >= 512:
            m, self.reste = self.reste[:512], self.reste[512:]
            p = self.vad(m)
            if not self.phrase:
                self.avant.append(m)
                if p >= VAD_SEUIL:
                    self.phrase, self.n_parole, self.n_silence = list(self.avant), 1, 0
                continue
            self.phrase.append(m)
            if p >= VAD_SEUIL - 0.15:
                self.n_parole += p >= VAD_SEUIL
                self.n_silence = 0
            else:
                if self.n_silence == 0:
                    self.t_fin = time.perf_counter()  # fin de parole (à confirmer par le silence)
                self.n_silence += 1
                if self.n_silence * 512 / 16000 >= SILENCE_FIN:
                    if self.n_parole * 512 / 16000 >= PAROLE_MIN:
                        resultat = (np.concatenate(self.phrase), self.t_fin)
                    self.phrase, self.n_parole, self.n_silence = [], 0, 0
                    self.avant.clear()
        return resultat


# ─── Visage : Ditto alimenté en continu ──────────────────────────────────────
class Visage:
    """Alimente Ditto bloc par bloc (0,2 s) et associe chaque image produite à ses 40 ms d'audio.

    Ditto produit les images dans l'ordre où l'audio lui est donné : une file (audio, réponse)
    suffit à les apparier. Entre deux réponses, on n'alimente plus Ditto : le visage reste figé.
    """

    def __init__(self, sdk):
        self.sdk = sdk
        self.cond = threading.Condition()
        self.attente = np.zeros(0, np.float32)
        self.rid, self.actif, self.fini = None, False, False
        self.n_reponse = 0
        self.silences = 0
        self.t_premiere = {}
        self.fins = set()
        self.images = collections.deque()
        self.buf16 = np.zeros(3 * 640, np.float32)  # contexte initial (comme dans le test isolé)
        self.sortie = None
        self.n_nourries = self.n_emises = 0
        sdk.writer = self  # Ditto appelle self(image, fmt) depuis son fil « writer »
        threading.Thread(target=self._boucle, daemon=True, name="visage").start()

    # appelé par le fil de la réponse
    def commencer(self, rid):
        with self.cond:
            self.rid, self.actif, self.fini = rid, True, False
            self.n_reponse = self.silences = 0
            self.attente = np.zeros(0, np.float32)

    def pousser(self, audio24):
        with self.cond:
            self.attente = np.concatenate([self.attente, audio24])
            self.cond.notify()

    def terminer(self):
        with self.cond:
            self.fini = True
            self.cond.notify()

    def _avance(self):
        t0 = self.t_premiere.get(self.rid)
        if t0 is None:
            return float("inf")
        return self.n_reponse / IPS - (time.perf_counter() - t0)

    def _boucle(self):
        zeros = np.zeros(BLOC_24, np.float32)
        while True:
            with self.cond:
                while True:
                    if len(self.attente) >= BLOC_24:
                        bloc, self.attente, dernier = self.attente[:BLOC_24], self.attente[BLOC_24:], False
                        break
                    if self.actif and self.fini:
                        bloc = np.pad(self.attente, (0, BLOC_24 - len(self.attente)))
                        self.attente, self.actif, dernier = np.zeros(0, np.float32), False, True
                        break
                    if self.actif and self._avance() < AVANCE_MIN:
                        bloc, dernier = zeros, False  # la voix prend du retard : on comble
                        self.silences += 1
                        break
                    self.cond.wait(timeout=0.02)
                rid = self.rid
            try:
                self._nourrir(bloc, rid, False)
                if dernier:
                    for _ in range(BLOCS_VIDANGE):
                        self._nourrir(zeros, rid, True)
            except Exception:
                traceback.print_exc()

    def _nourrir(self, bloc24, rid, vidange):
        for f in range(BLOC_IMAGES):
            self.images.append((bloc24[f * ECH_IMAGE:(f + 1) * ECH_IMAGE], rid, vidange))
        if not vidange:
            self.n_reponse += BLOC_IMAGES
        self.n_nourries += BLOC_IMAGES
        self.buf16 = np.concatenate([self.buf16, resample_poly(bloc24, 2, 3).astype(np.float32)])
        while len(self.buf16) >= FENETRE_16:
            self.sdk.run_chunk(self.buf16[:FENETRE_16], (3, 5, 2))
            self.buf16 = self.buf16[PAS_16:]

    # appelé par Ditto (fil « writer »)
    def __call__(self, img, fmt="rgb"):
        self.n_emises += 1
        try:
            audio, rid, vidange = self.images.popleft()
        except IndexError:
            return
        if rid is None:
            return
        if vidange:
            if rid not in self.fins:
                self.fins.add(rid)
                if self.sortie:
                    self.sortie.fin(rid)
            return
        self.t_premiere.setdefault(rid, time.perf_counter())
        h, w = img.shape[:2]
        petite = cv2.resize(img, (LARGEUR_VIDEO, round(h * LARGEUR_VIDEO / w)), interpolation=cv2.INTER_AREA)
        _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(petite, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, QUALITE_JPEG])
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        if self.sortie:
            self.sortie.image(rid, pcm, jpeg.tobytes())

    def close(self):
        pass


# ─── Session : une connexion WebSocket ───────────────────────────────────────
COMPTEUR = itertools.count(1)
EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")  # STT + voix : un seul fil
WHISPER = TTS = VISAGE = CLIENT = None


class Session:
    def __init__(self, ws, loop):
        self.ws, self.loop = ws, loop
        self.file = asyncio.Queue()
        self.historique = []
        self.ecoute = Ecoute()
        self.occupe = False
        self.parole_signalee = False
        self.mesures = {}

    def envoyer(self, msg):  # depuis n'importe quel fil
        self.loop.call_soon_threadsafe(self.file.put_nowait, msg)

    async def expediteur(self):
        while (msg := await self.file.get()) is not None:
            try:
                if isinstance(msg, bytes):
                    await self.ws.send_bytes(msg)
                else:
                    await self.ws.send_str(json.dumps(msg, ensure_ascii=False))
            except (ConnectionResetError, RuntimeError):
                break

    # sortie du visage
    def image(self, rid, pcm, jpeg):
        self.envoyer(struct.pack("<BBHII", 1, 0, len(pcm) // 2, rid, 0) + pcm + jpeg)

    def fin(self, rid):
        self.envoyer({"type": "fin", "rid": rid})

    def traiter(self, audio, t_fin, texte=None):
        """Fil « gpu » : transcription → Claude (fil séparé) → voix → visage."""
        rid = next(COMPTEUR)
        m = {"rid": rid, "t_fin": t_fin, "t_decision": time.perf_counter()}
        self.mesures[rid] = m
        demarre = False
        try:
            if texte is None:
                segments, _ = WHISPER.transcribe(audio, language="fr", beam_size=1, vad_filter=False,
                                                 initial_prompt=PROMPT_STT, condition_on_previous_text=False)
                texte = " ".join(s.text.strip() for s in segments).strip()
                m["duree_parole"] = len(audio) / 16000
            m["t_stt"] = time.perf_counter()
            m["question"] = texte
            if not texte:
                self.envoyer({"type": "etat", "etat": "ecoute"})
                self.reprendre()
                return
            self.envoyer({"type": "transcription", "texte": texte})
            self.historique.append({"role": "user", "content": texte})
            VISAGE.sortie = self
            VISAGE.commencer(rid)
            demarre = True

            phrases, mc = queue.Queue(), {}

            def penser():
                try:
                    for p in cerveau.phrases_en_flux(CLIENT, list(self.historique), mesures=mc):
                        m.setdefault("t_phrase1", time.perf_counter())
                        phrases.put(p)
                except Exception as e:  # transmis au fil gpu
                    phrases.put(e)
                finally:
                    phrases.put(None)

            threading.Thread(target=penser, daemon=True, name="claude").start()
            reponse = []
            while (p := phrases.get()) is not None:
                if isinstance(p, Exception):
                    raise p
                reponse.append(p)
                self.envoyer({"type": "phrase", "texte": p})
                for morceau, sr, _ in TTS.generate_voice_clone_streaming(
                        text=p, language="French", ref_audio=REF_AUDIO, ref_text=REF_TEXTE, chunk_size=TTS_CHUNK):
                    m.setdefault("t_son1", time.perf_counter())
                    morceau = np.asarray(morceau, np.float32).reshape(-1)
                    if sr != 24000:
                        morceau = resample_poly(morceau, 24000, sr).astype(np.float32)
                    VISAGE.pousser(morceau)
            m["t_voix_finie"] = time.perf_counter()
            m["reponse"] = " ".join(reponse)
            self.historique.append({"role": "assistant", "content": m["reponse"]})
        except Exception as e:
            traceback.print_exc()
            if self.historique and self.historique[-1]["role"] == "user":
                self.historique.pop()
            self.envoyer({"type": "erreur", "texte": f"{type(e).__name__} : {e}"})
            if not demarre:
                self.reprendre()
        finally:
            if demarre:
                VISAGE.terminer()  # → silence de vidange → message « fin » → le navigateur répond « pret »

    def reprendre(self):
        self.ecoute.reset()
        self.parole_signalee = False
        self.occupe = False

    def cloturer(self, rid):
        m = self.mesures.pop(rid, None)
        if not m or "t_stt" not in m:
            return
        t_image = VISAGE.t_premiere.pop(rid, None)

        def ms(a, b):
            return round((m[b] - m[a]) * 1000) if a in m and b in m else None

        if t_image:
            m["t_image1"] = t_image
        res = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": m.get("question"),
            "reponse": m.get("reponse"),
            "duree_parole_s": round(m.get("duree_parole", 0), 2),
            "attente_silence_vad": ms("t_fin", "t_decision"),
            "transcription": ms("t_decision", "t_stt"),
            "claude_1re_phrase": ms("t_stt", "t_phrase1"),
            "voix_1er_son": ms("t_phrase1", "t_son1"),
            "visage_1re_image": ms("t_son1", "t_image1"),
            "reseau_et_tampon": ms("t_image1", "t_lecture"),
            "total_serveur": ms("t_fin", "t_image1"),
            "total_percu": ms("t_fin", "t_lecture"),
            "silences_inseres": VISAGE.silences,
        }
        log("Délais (ms) :", {k: v for k, v in res.items() if k not in ("question", "reponse", "date")})
        JOURNAL.parent.mkdir(exist_ok=True)
        with open(JOURNAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        self.envoyer({"type": "mesures", **res})


def rechauffer_claude():
    try:
        CLIENT.models.list(limit=1)
    except Exception:
        pass


async def page(request):
    return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


async def photo(request):
    return web.FileResponse(SOURCE)


async def websocket(request):
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    s = Session(ws, loop)
    VISAGE.sortie = s  # la dernière page connectée reçoit le visage
    tache = asyncio.create_task(s.expediteur())
    log("Navigateur connecté :", request.headers.get("Tailscale-User-Login") or request.remote)
    s.envoyer({"type": "etat", "etat": "ecoute"})
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            if s.occupe:
                continue  # semi-duplex : on n'écoute pas pendant que l'assistant parle
            res = s.ecoute.ajouter(np.frombuffer(msg.data, "<i2").astype(np.float32) / 32768)
            if s.ecoute.en_parole and not s.parole_signalee:
                s.parole_signalee = True
                s.envoyer({"type": "etat", "etat": "entendu"})
                # pendant que le client parle : ouvrir la connexion HTTPS vers l'API (−0,5 s au 1er appel)
                loop.run_in_executor(None, rechauffer_claude)
            if res:
                s.occupe = True
                s.envoyer({"type": "etat", "etat": "reflexion"})
                loop.run_in_executor(EXEC, s.traiter, *res)
        elif msg.type == WSMsgType.TEXT:
            d = json.loads(msg.data)
            if d["type"] == "lecture":
                m = s.mesures.get(d.get("rid"))
                if m is not None:
                    m.setdefault("t_lecture", time.perf_counter())
                s.envoyer({"type": "etat", "etat": "parle"})
            elif d["type"] == "pret":
                s.cloturer(d.get("rid"))
                s.reprendre()
                s.envoyer({"type": "etat", "etat": "ecoute"})
            elif d["type"] == "texte" and not s.occupe and d.get("texte", "").strip():
                s.occupe = True
                s.envoyer({"type": "etat", "etat": "reflexion"})
                loop.run_in_executor(EXEC, s.traiter, None, time.perf_counter(), d["texte"].strip())
    s.file.put_nowait(None)
    await tache
    if VISAGE.sortie is s:
        VISAGE.sortie = None
    log("Navigateur déconnecté")
    return ws


# ─── Démarrage ───────────────────────────────────────────────────────────────
def charger():
    global WHISPER, TTS, VISAGE, CLIENT
    t = time.perf_counter()
    WHISPER = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16",
                           download_root=str(RACINE / "models" / "whisper"))
    list(WHISPER.transcribe(np.zeros(16000, np.float32), language="fr")[0])
    log(f"Reconnaissance vocale prête ({time.perf_counter() - t:.0f} s)")

    t = time.perf_counter()
    TTS = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / "qwen3-tts-1.7B"))
    # graphes CUDA capturés dans le fil qui servira ensuite
    EXEC.submit(lambda: list(TTS.generate_voice_clone_streaming(
        text="Bonjour.", language="French", ref_audio=REF_AUDIO, ref_text=REF_TEXTE,
        chunk_size=TTS_CHUNK))).result()
    log(f"Voix prête ({time.perf_counter() - t:.0f} s)")

    t = time.perf_counter()
    if not SOURCE.exists():
        img = cv2.imread(str(RACINE / "assets" / "visage_pierre.jpg"))
        h, w = img.shape[:2]
        cv2.imwrite(str(SOURCE), cv2.resize(img, (round(w * 960 / h), 960), interpolation=cv2.INTER_AREA))
    sdk = StreamSDK(str(RACINE / "models/ditto/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"),
                    str(RACINE / "models/ditto/ditto_pytorch"))
    # torch.compile dans le fil principal (dans les fils de Ditto, la compilation se bloque).
    # Le nombre de fils CPU de torch fait partie des gardes de compilation : on aligne le fil
    # principal sur la valeur d'un fil neuf (celle des fils de Ditto), sinon tout recompile.
    n_fils = []
    fil = threading.Thread(target=lambda: n_fils.append(torch.get_num_threads()))
    fil.start()
    fil.join()
    torch.set_num_threads(n_fils[0])
    sdk.warp_f3d.warp_net.model.compile()
    sdk.decode_f3d.decoder.model.compile()
    f3d = np.zeros((1, 32, 16, 64, 64), dtype=np.float32)
    kp = np.zeros((1, 21, 3), dtype=np.float32)
    sdk.decode_f3d(sdk.warp_f3d(f3d, kp, kp))
    sortie = RACINE / "models" / "sorties_visage"
    sortie.mkdir(parents=True, exist_ok=True)
    sdk.setup(str(SOURCE), str(sortie / "serveur.mp4"), online_mode=True, overlap_v2=70, sampling_timesteps=5)
    try:
        sdk.writer.close() if hasattr(sdk.writer, "close") else None
    except Exception:
        pass
    VISAGE = Visage(sdk)
    # préchauffage : 2 s de silence
    VISAGE.commencer(None)
    VISAGE.pousser(np.zeros(2 * 24000, np.float32))
    VISAGE.terminer()
    limite = time.perf_counter() + 60
    while ((VISAGE.n_nourries < 50 or VISAGE.n_emises < VISAGE.n_nourries - 25)
           and time.perf_counter() < limite):
        time.sleep(0.1)
    time.sleep(0.5)
    log(f"Visage prêt ({time.perf_counter() - t:.0f} s) — images données {VISAGE.n_nourries}, "
        f"produites {VISAGE.n_emises}")

    CLIENT = anthropic.Anthropic()
    log(f"VRAM utilisée : {torch.cuda.memory_allocated() / 1e9:.1f} Go (PyTorch)")


def main():
    charger()
    app = web.Application()
    app.add_routes([web.get("/", page), web.get("/photo.jpg", photo), web.get("/ws", websocket)])
    log(f"Serveur prêt : http://127.0.0.1:{PORT}/  (via Tailscale : tailscale serve --bg {PORT})")
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)


if __name__ == "__main__":
    main()
