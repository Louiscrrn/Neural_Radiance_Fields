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
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def step(self, batch):
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

        psnr = mse2psnr(loss.detach()).item()

        N_rays = rgb_map.shape[0]
        H = W = int(math.sqrt(N_rays))
        rgb_map_img = rgb_map.view(H, W, 3).permute(2, 0, 1).unsqueeze(0)
        target_img  = target_rgb.view(H, W, 3).permute(2, 0, 1).unsqueeze(0)
        
        ssim = self.ssim_fn(rgb_map_img,
                    target_img).item()

        return loss, psnr, ssim

    def fit(self, train_loader, val_loader=None, epochs=10, log_every=100):

        global_step = 0
        history = []

        for epoch in range(epochs):

            # --- TRAIN ---
            self.model.train()
            train_loss, train_psnr, train_ssim, train_count = 0, 0, 0, 0
            pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                loss, psnr, ssim = self.step(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_psnr += psnr
                train_ssim += ssim
                train_count += 1
                global_step += 1

                if global_step % log_every == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})


            train_loss /= train_count
            train_psnr /= train_count
            train_ssim /= train_count

            # --- VALIDATION ---
            if val_loader is not None:
                self.model.eval()
                val_loss, val_psnr, val_ssim, val_count = 0, 0, 0, 0

                with torch.no_grad():
                    for batch in val_loader:
                        loss, psnr, ssim = self.step(batch)
                        val_loss += loss.item()
                        val_psnr += psnr
                        val_ssim += ssim
                        val_count += 1

                val_loss /= val_count
                val_psnr /= val_count
                val_ssim /= val_count

                print(f"Epoch {epoch+1}/{epochs} "
                      f"| Train Loss: {train_loss:.6f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.3f} "
                      f"| Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.3f}")
            else:
                val_loss = val_psnr = val_ssim = None
                print(f"Epoch {epoch+1}/{epochs} "
                      f"| Train Loss: {train_loss:.6f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.3f}")
                
            history.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_psnr": train_psnr,
                "train_ssim": train_ssim,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            })
        
        return pd.DataFrame(history)
        

def mse2psnr(mse):
    return -10.0 * torch.log10(mse)
