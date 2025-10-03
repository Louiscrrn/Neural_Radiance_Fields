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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Modèle chargé depuis {model_path}")

    # --- Render params ---
    H = W = config["train"]["img_size"]
    focal = float(config["render"]["focal"])
    phi = float(config["render"]["phi"])
    radius = float(config["render"]["radius"])
    step = int(config["render"]["step"])

    # --- Precompute all views with tqdm ---
    theta_list = np.arange(0, 361, 5)  # par exemple tous les 5°
    images = []

    print(f"Pré-calcul des {len(theta_list)} vues en cours...")
    for theta in tqdm(theta_list, desc="Pré-calcul"):
        c2w = pose_spherical(theta, phi, radius)
        img = render_image(model, H, W, focal, c2w, device=device)
        images.append(img)
    print("Pré-calcul terminé !")

    # --- Interactive viewer ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    im = ax.imshow(images[0])
    ax.axis('off')
    ax.set_title(f"Angle θ = {theta_list[0]:.1f}°")

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_theta = Slider(
        ax_slider,
        label='Theta (angle)',
        valmin=0,
        valmax=len(theta_list)-1,
        valinit=0,
        valstep=1
    )

    def update(idx):
        idx = int(slider_theta.val)
        im.set_data(images[idx])
        ax.set_title(f"Angle θ = {theta_list[idx]:.1f}°")
        fig.canvas.draw_idle()

    slider_theta.on_changed(update)

    plt.show()
