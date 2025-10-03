from torch.utils.data import Dataset
import torch
from pathlib import Path
from PIL import Image
import json
from torchvision import transforms

class LegoDataset(Dataset):
    
    def __init__(self, root_dir: str, type: str, img_size: int, size: int, transform: bool= None):

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

        # 🟢 Ici on applique le size (si non None)
        if size is not None:
            self.images_files = self.images_files[:size]
            self.frames = self.frames[:size]

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

def get_rays(H, W, focal, c2w):
    """Retourne les origines et directions des rayons pour chaque pixel"""
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy"
    )
    dirs = torch.stack(
        [(i - W*0.5)/focal, -(j - H*0.5)/focal, -torch.ones_like(i)],
        -1
    )  # (H, W, 3)

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # rotation
    rays_o = c2w[:3,3].expand(rays_d.shape)  # translation répétée

    return rays_o, rays_d

def collate_rays(batch, N_rays=1024):
   
    sample = batch[0]
    img, pose, camera_angle_x = sample["image"], sample["pose"], sample["camera_angle_x"]

    H, W = img.shape[1], img.shape[2]
    focal = 0.5 * W / torch.tan(0.5 * camera_angle_x)

    rays_o, rays_d = get_rays(H, W, focal, pose)  # (H,W,3)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    img_flat = img.permute(1,2,0).reshape(-1, 3)  # (H*W, 3)

    select_inds = torch.randint(0, rays_o.shape[0], (N_rays,))
    rays_o = rays_o[select_inds]
    rays_d = rays_d[select_inds]
    target_rgb = img_flat[select_inds]

    return {
        "rays_o": rays_o,         # (N_rays, 3)
        "rays_d": rays_d,         # (N_rays, 3)
        "target_rgb": target_rgb  # (N_rays, 3)
    }

