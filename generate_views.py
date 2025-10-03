import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dotenv import load_dotenv
import os
import yaml
from pathlib import Path
from model import NeRF, render_image, pose_spherical
import numpy as np
from tqdm import tqdm 
from PIL import Image

if __name__ == "__main__":
    # --- Load config ---
    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- Load model ---
    device = config["train"]["device"]
    model = NeRF().to(device)

    model_path = str(config["render"]["model_path"])
    checkpoint = torch.load(model_path + ".pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Modèle chargé depuis {model_path}")

    # --- Render params ---
    H = W = config["train"]["img_size"]
    focal = float(config["render"]["focal"])
    phi = float(config["render"]["phi"])
    radius = float(config["render"]["radius"])
    step = int(config["render"]["step"])

    # --- Output folder ---
    output_dir = Path("outputs") / Path("renders") / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Precompute all views with tqdm ---
    theta_list = np.arange(0, 361, 5)  # par exemple tous les 5°
    print(f"Génération et sauvegarde de {len(theta_list)} vues dans {output_dir}")

    for i, theta in enumerate(tqdm(theta_list, desc="Rendu vues")):
        c2w = pose_spherical(theta, phi, radius)
        img = render_image(model, H, W, focal, c2w, device=device)  # [H, W, 3], float32 [0,1]

        # Convertir en PIL image
        img_uint8 = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)

        img_pil.save(output_dir / f"view_{i:03d}.png")

    print(f"✅ Sauvegarde terminée dans {output_dir}")
