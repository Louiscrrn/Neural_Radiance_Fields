import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import sample_points, volume_rendering

class NeRFTrainer:
    def __init__(self, model, optimizer, device="cuda", 
                 near=2.0, far=6.0, N_samples=64):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.near = near
        self.far = far
        self.N_samples = N_samples

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

        return loss.item()

    def train(self, train_loader, epochs=10, log_every=100):
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            batch_count = 0

            for batch in pbar:
                loss = self.train_step(batch)
                global_step += 1
                epoch_loss += loss
                batch_count += 1

                if global_step % log_every == 0:
                    pbar.set_postfix({"loss": f"{loss:.6f}"})

            avg_loss = epoch_loss / batch_count
            print(f" Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.6f}")
