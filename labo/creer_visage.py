"""Génère le visage d'un personnage FICTIF (personne qui n'existe pas) avec RealVisXL (SDXL, local).

Usage : python labo/creer_visage.py <persona> [nombre]   → personas/<persona>/candidats/*.jpg
Contraintes pour l'animation (Ditto) : visage de face, bouche fermée, épaules visibles, regard caméra,
lumière douce. Le décor et la tenue viennent de personas/<persona>/fiche.json (« description_visage »).
"""
import json
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

RACINE = Path(__file__).resolve().parent.parent
MODELE = RACINE / "models" / "realvisxl"

CADRAGE = ("head and shoulders portrait photo, facing the camera, looking straight into the lens, "
           "mouth closed, gentle friendly smile with closed lips, eyes open, centered, sharp focus on the face, "
           "soft natural daylight, photorealistic, 35mm photo, high detail skin")
NEGATIF = ("open mouth, teeth, sunglasses, hat, hands near face, profile, looking away, tilted head, "
           "blurry, cartoon, illustration, 3d render, text, watermark, logo, deformed, extra fingers")


def main():
    persona = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dossier = RACINE / "personas" / persona
    fiche = json.loads((dossier / "fiche.json").read_text(encoding="utf-8"))
    sortie = dossier / "candidats"
    sortie.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODELE, torch_dtype=torch.float16, variant="fp16",
                                                     use_safetensors=True).to("cuda")
    invite = f"{fiche['description_visage']}, {CADRAGE}"
    premier = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for graine in range(premier, premier + n):
        g = torch.Generator("cuda").manual_seed(1000 + graine)
        image = pipe(invite, negative_prompt=NEGATIF, width=832, height=1216, num_inference_steps=30,
                     guidance_scale=5.0, generator=g).images[0]
        chemin = sortie / f"candidat_{graine + 1}.jpg"
        image.save(chemin, quality=95)
        print(chemin)


if __name__ == "__main__":
    main()
