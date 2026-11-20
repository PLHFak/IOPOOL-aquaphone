"""Accélération de la voix sans changer sa hauteur (choix de Pierre le 20/11 : ×1,15 à l'écoute).

Le modèle de voix n'a pas de réglage de débit : on accélère le son en sortie avec Rubber Band (filtre ffmpeg
`rubberband`, le même que pour les échantillons écoutés), EN FLUX : un ffmpeg par phrase, les morceaux entrent
dès qu'ils sont générés et ressortent accélérés au fil de l'eau (pas d'attente de la fin de la phrase).
"""
import queue
import shutil
import subprocess
import threading

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
SR = 24000


def accelerer(morceaux, vitesse):
    """Générateur : morceaux float32 mono 24 kHz → mêmes morceaux accélérés (vitesse > 1 = plus rapide)."""
    if abs(vitesse - 1.0) < 1e-3:
        yield from morceaux
        return
    p = subprocess.Popen([FFMPEG, "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                          "-af", f"rubberband=tempo={vitesse}:pitchq=quality", "-flush_packets", "1",
                          "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    sortie = queue.Queue()

    def lire():
        reste = b""
        while True:
            b = p.stdout.read(4096)
            if not b:
                break
            reste += b
            n = len(reste) // 4 * 4
            if n:
                sortie.put(np.frombuffer(reste[:n], np.float32).copy())
                reste = reste[n:]
        sortie.put(None)

    fil = threading.Thread(target=lire, daemon=True)
    fil.start()
    fini = False
    try:
        for m in morceaux:
            p.stdin.write(np.asarray(m, np.float32).reshape(-1).tobytes())
            while True:  # ce qui est déjà sorti part tout de suite
                try:
                    x = sortie.get_nowait()
                except queue.Empty:
                    break
                if x is None:
                    fini = True
                    break
                yield x
        p.stdin.close()
        while not fini:
            x = sortie.get()
            if x is None:
                break
            yield x
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
