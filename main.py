import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
from dataset import LegoDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    load_dotenv()
    data_path = Path(os.getenv('DATA_PATH'))

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    train_dataset = LegoDataset(
            data_path / config["path"]["root"], 
            type="train")

    train_loader = DataLoader(
            train_dataset,
            batch_size=4,      
            shuffle=True,      
    )

    for batch in train_loader:
        images = batch["image"]         
        poses = batch["pose"]           
        camera_angle_x = batch["camera_angle_x"]  
        print(images.shape, poses.shape, camera_angle_x)
        break