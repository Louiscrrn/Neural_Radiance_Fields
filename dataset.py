from torch.utils.data import Dataset
import torch
from pathlib import Path
from PIL import Image
import json
from torchvision import transforms

class LegoDataset(Dataset):
    
    def __init__(self, root_dir: str, type: str, img_size :int=800, transform: bool= None):

        if type not in ["train", "test", "val"] :
            raise Exception("Error : Unrecognized Dataset type")

        self.root_dir = Path(root_dir)
        self.files_directory = Path(root_dir) / Path(type) 

        self.images_files = sorted([
            f for f in self.files_directory.iterdir() 
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

        self.frames_files = f"transforms_{type}.json"
        with open(self.root_dir / self.frames_files) as f :
            X = json.load(f)

        self.camera_angle_x = X["camera_angle_x"]
        self.frames = X["frames"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
        ])

    def __len__(self) :
        return len(self.images_files)

    def __getitem__(self, idx):

        frame = self.frames[idx]
        img = Image.open(self.images_files[idx]).convert("RGB")
        img = self.transform(img)

        return {
            "image": img, 
            "pose": torch.tensor(frame["transform_matrix"], dtype=torch.float32),
            "camera_angle_x": torch.tensor(self.camera_angle_x, dtype=torch.float32),
            "rotation": torch.tensor(frame["rotation"], dtype=torch.float32)
        }



