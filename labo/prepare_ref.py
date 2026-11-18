"""Affiche les segments horodatés de la voix de Pierre pour choisir l'extrait de référence du clonage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cuda_win  # noqa: F401,E402
from faster_whisper import WhisperModel  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16",
                 download_root=str(RACINE / "models" / "whisper"))
segs, _ = m.transcribe(str(RACINE / "models" / "voix_pierre_16k.wav"), language="fr",
                       word_timestamps=True, initial_prompt="Pierre L'Hoest, pisciniste, sonde iopool.")
for s in segs:
    print(f"[{s.start:6.2f} → {s.end:6.2f}] {s.text.strip()}")
