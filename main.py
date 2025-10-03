import torch
import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
from dataset import LegoDataset, collate_rays
from torch.utils.data import DataLoader
from model import NeRF
from trainer import NeRFTrainer
from datetime import datetime
from utils import plot_training_curves


if __name__ == "__main__":

    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d-%H-%M")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # --- Dataset Loading ---
    img_size = config["train"]["img_size"]
    train_dataset = LegoDataset(
            data_path / config["path"]["root"], 
            type="train",
            img_size=img_size
        )
    val_dataset = LegoDataset(
            data_path / config["path"]["root"], 
            type="val",
            img_size=img_size
        )
    print("Dataset length:", len(train_dataset))
    
    N_rays = config["train"]["N_rays"]
    train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],      
            shuffle=True,
            collate_fn=lambda b: collate_rays(b, N_rays=N_rays)
    )
    val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],      
            shuffle=True,
            collate_fn=lambda b: collate_rays(b, N_rays=N_rays)
    )


    # --- Fitting --- 
    model = NeRF()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    trainer = NeRFTrainer(model, optimizer, device=str(config["train"]["device"]), near=float(config["train"]["near"]), far=float(config["train"]["far"]), N_samples=int(config["train"]["N_samples"]))

    print("\n\n TRAINING STARTS : \n")
    historic = trainer.fit(train_loader, val_loader=val_loader, epochs=int(config["train"]["epoch"]), log_every=int(config["train"]["log_every"]))
    
    # --- Saving --- 
    prefix_name = config["path"]["outputs"] + "/" + config["experiment"]["name"] + "_" + formatted + "_"
    historic.to_csv(prefix_name + "historic.csv", index=False)
    plot_training_curves(historic, save_dir=prefix_name + "curves/")

    model_path = prefix_name + "nerf_weights.pth"
    torch.save(trainer.model.state_dict(), model_path)
