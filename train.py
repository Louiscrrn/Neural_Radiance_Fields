import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import sample_points, volume_rendering
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import pandas as pd
import math

class NeRFTrainer:
    def __init__(self, model, optimizer, device="cuda", 
                 near=2.0, far=6.0, N_samples=64):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.near = near
        self.far = far
        self.N_samples = N_samples
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)

    def train_step(self, batch):
        rays_o = batch["rays_o"].to(self.device)
        rays_d = batch["rays_d"].to(self.device)
        target_rgb = batch["target_rgb"].to(self.device)

        pts, t_vals = sample_points(rays_o, rays_d, self.near, self.far, self.N_samples)

        pts_flat = pts.reshape(-1, 3)
        rgb, sigma = self.model(pts_flat)  
        rgb = rgb.view(pts.shape[0], self.N_samples, 3)
        sigma = sigma.view(pts.shape[0], self.N_samples, 1)

        rgb_map, acc_map, weights = volume_rendering(rgb, sigma, t_vals, rays_d)

        loss = F.mse_loss(rgb_map, target_rgb)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        psnr = mse2psnr(loss.detach()).item()

        N_rays = rgb_map.shape[0]
        H = W = int(math.sqrt(N_rays))
        rgb_map_img = rgb_map.view(H, W, 3).permute(2, 0, 1).unsqueeze(0)
        target_img  = target_rgb.view(H, W, 3).permute(2, 0, 1).unsqueeze(0)
        
        ssim = self.ssim_fn(rgb_map_img,
                    target_img).item()

        return loss.item(), psnr, ssim

    def train(self, train_loader, epochs=10, log_every=100):
        self.model.train()
        global_step = 0
        data = []

        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            batch_count = 0
            epoch_ssim = 0.0
            epoch_psnr = 0.0

            for batch in pbar:
                loss, psnr, ssim = self.train_step(batch)
                global_step += 1
                epoch_loss += loss
                batch_count += 1
                epoch_psnr += psnr
                epoch_ssim += ssim

                if global_step % log_every == 0:
                    pbar.set_postfix({"loss": f"{loss:.6f}"})

            avg_loss = epoch_loss / batch_count
            avg_psnr = epoch_psnr / batch_count
            avg_ssim = epoch_ssim / batch_count
            print(f" Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f}")
            data.append({'epoch' : epoch + 1, 'avg_loss' : avg_loss, 'avg_psnr' : avg_psnr, 'avg_ssim' : avg_ssim})
        
        historic = pd.DataFrame(data)
        return historic
        

def mse2psnr(mse):
    return -10.0 * torch.log10(mse)
