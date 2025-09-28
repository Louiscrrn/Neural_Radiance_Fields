import torch
import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
from dataset import LegoDataset, collate_rays
from torch.utils.data import DataLoader
from model import NeRF
from train import NeRFTrainer

if __name__ == "__main__":
    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    train_dataset = LegoDataset(
            data_path / config["path"]["root"], 
            type="train",
            img_size=config["train"]["img_size"]
        )
    
    print("Dataset length:", len(train_dataset))
    
    train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],      
            shuffle=True,
            collate_fn=lambda b: collate_rays(b, N_rays=config["train"]["N_rays"])
    )

    first_sample = train_dataset[0]
    print("   Sample keys:", first_sample.keys())
    print("   Image shape:", first_sample["image"].shape)
    print("   Pose shape:", first_sample["pose"].shape)
    print("   Camera angle:", first_sample["camera_angle_x"].item())

    model = NeRF()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))

    trainer = NeRFTrainer(model, optimizer, device="cpu", near=2.0, far=6.0, N_samples=config["train"]["N_samples"])

    print("\n\n TRAINING STARTS : \n")
    trainer.train(train_loader, epochs=config["train"]["epoch"], log_every=config["train"]["log_every"])
