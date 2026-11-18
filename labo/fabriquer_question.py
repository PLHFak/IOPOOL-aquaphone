"""Fabrique l'audio du cas de test (voix clonée de Pierre) pour tester le pipeline sans micro."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serveur import REF_AUDIO, REF_TEXTE, RACINE  # noqa: E402

QUESTION = "Mon eau est verte depuis deux jours, j'ai mis du chlore choc et ça ne change rien."
modele = FasterQwen3TTS.from_pretrained(str(RACINE / "models" / "qwen3-tts-1.7B"))
morceaux = [a for a, sr, _ in modele.generate_voice_clone_streaming(
    text=QUESTION, language="French", ref_audio=REF_AUDIO, ref_text=REF_TEXTE, chunk_size=8)]
audio = resample_poly(np.concatenate(morceaux), 2, 3).astype(np.float32)
sf.write(RACINE / "models" / "question_test_16k.wav", audio, 16000)
print(f"question_test_16k.wav : {len(audio) / 16000:.1f} s")
