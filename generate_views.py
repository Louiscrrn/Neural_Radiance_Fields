from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from dataset import LegoDataset
import os
from model import NeRF, render_image, pose_spherical, get_rays
import torch
from dotenv import load_dotenv
from torchsummary import summary

if __name__ == "__main__":

    path = "outputs/lego_orbit/renders/lego/view_058.png"
    load_dotenv()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- Dataset Loading ---
    data_path = Path(os.getenv('DATA_PATH'))
    img_size = config["train"]["img_size"]
    train_dataset = LegoDataset(
            data_path / config["path"]["root"], 
            type="train",
            img_size=img_size
        )
    print(f"----------------------------------------------------------------")
    print(f"Dataset size : {train_dataset.__len__()}")
  
    # --- Load model ---
    device = config["train"]["device"]
    model = NeRF().to(device)
    model_name = str(config["render"]["model_name"])
    model_path =  "outputs/lego_orbit/" + model_name + ".pth"
    checkpoint = torch.load( model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"----------------------------------------------------------------")
    print(f"Model loaded from : {model_path}")
    summary(model, input_size=(3, ))

    # --- Params de rendu 
    H = W = img_size                             
    near, far = config["train"]["near"], config["train"]["far"]                         
    N_samples = int(config["train"]["N_samples"]) 

    # --- Render params ---
    H = W = config["train"]["img_size"]
    theta = float(config["render"]["theta"])
    phi = float(config["render"]["phi"])
    radius = float(config["render"]["radius"])
    step = int(config["render"]["step"])
    focal = (0.5 * W) / np.tan(0.5 * train_dataset.camera_angle_x)
    print(f"Focal : {focal}")
    print(f"----------------------------------------------------------------")

    # --- Choix d'une vue : caméra sur une orbite 

    for theta in range(0, 360, 1):
        c2w = pose_spherical(theta, phi, radius).to(torch.float32).to(device)
        print(f"Spherical Pose : {c2w}")
        print(f"----------------------------------------------------------------")
        # Après avoir construit c2w avec la fonction ci-dessus
        rays_o, rays_d = get_rays(H=img_size, W=img_size, focal=float(focal), c2w=c2w)
        center = (img_size * img_size) // 2
        cam_pos = c2w[:3, 3]
        to_origin = (-cam_pos).numpy()
        ray = rays_d[center].numpy()

        # Le produit scalaire doit être > 0 (le rayon va vers l'origine)
        print("dot(ray, to_origin) =", (ray * to_origin).sum())
        print(f"----------------------------------------------------------------")


        # --- Génération
        
        img = render_image(
            model, H, W,
            focal=float(focal),  
            c2w=c2w,
            near=near, far=far,
            N_samples=N_samples,
            device=device
        ) 
    
        # --- Sauvegarde
        out_dir = Path("outputs/lego_orbit/renders/single_views")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"view_{theta:03d}.png"

        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(out_path)
        
    print(f"360 Vue sauvegardée : {out_path}")
