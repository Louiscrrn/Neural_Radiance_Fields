import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, L, include_input=False):
        super().__init__()
        self.L = L
        self.include_input = include_input

    def forward(self, x):
        out = [x] if self.include_input else []
        for i in range(self.L):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)
    

class NeRF(nn.Module):

    def __init__(self, depth: int = 8, L_encoding: int = 10, neurons: int = 256, include_input=True):
        super().__init__()
        self.neurons = neurons
        self.positional_encoding = PositionalEncoding(L=L_encoding, include_input=include_input)

        self.in_neurons = 3 * (2 * L_encoding + (1 if include_input else 0))

        self.in_layer = nn.Linear(self.in_neurons, neurons)

        self.layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(3)])

        self.skip_layer = nn.Linear(neurons + self.in_neurons, neurons)

        self.layers_post_skip = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(depth - 4)])

        self.down_layer = nn.Linear(neurons, neurons // 2)
        self.out_rgb = nn.Linear(neurons // 2, 3)       
        self.out_sigma = nn.Linear(neurons, 1)         

    def forward(self, x):
        gamma_x = self.positional_encoding(x)  

        out = F.relu(self.in_layer(gamma_x))

        for layer in self.layers:
            out = F.relu(layer(out))

        out = torch.cat([out, gamma_x], dim=-1)
        out = F.relu(self.skip_layer(out))

        for layer in self.layers_post_skip:
            out = F.relu(layer(out))

        sigma = F.softplus(self.out_sigma(out)) 

        h = self.down_layer(out) 
        rgb = torch.sigmoid(self.out_rgb(h))   

        return rgb, sigma
    

def sample_points(rays_o, rays_d, near, far, N_samples, perturb=True):
 
    N_rays = rays_o.shape[0]

    t_vals = torch.linspace(near, far, N_samples, device=rays_o.device) 
    t_vals = t_vals.expand(N_rays, N_samples) 

    if perturb:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        upper = torch.cat([mids, t_vals[:, -1:]], -1)
        lower = torch.cat([t_vals[:, :1], mids], -1)
        t_rand = torch.rand(t_vals.shape, device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None] 

    return pts, t_vals

def volume_rendering(rgb, sigma, t_vals, rays_d):
  
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples)

    deltas = deltas * torch.norm(rays_d[:, None, :], dim=-1)  # (N_rays, N_samples)


    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * deltas)  # (N_rays, N_samples)

    T = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]  # (N_rays, N_samples)

    weights = alpha * T  # (N_rays, N_samples)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (N_rays, 3)

    acc_map = torch.sum(weights, dim=-1)  # (N_rays,)

    return rgb_map, acc_map, weights

def get_rays(H, W, focal, c2w):
    """
    Génère les rayons (origine, direction) pour chaque pixel d'une caméra pinhole.
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='ij'
    )
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], -1)  # (W, H, 3)

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

@torch.no_grad()
def render_image(model, H, W, focal, c2w, near=2.0, far=6.0, N_samples=64, device="cuda"):
    """
    Rendu d'une image complète à partir du modèle NeRF entraîné.
    """
    model.eval()
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)

    # 1. Échantillonnage
    pts, t_vals = sample_points(rays_o, rays_d, near, far, N_samples)

    # 2. Évaluation du NeRF
    pts_flat = pts.reshape(-1, 3)
    rgb, sigma = model(pts_flat)
    rgb = rgb.view(pts.shape[0], N_samples, 3)
    sigma = sigma.view(pts.shape[0], N_samples, 1)

    # 3. Volume rendering
    rgb_map, _, _ = volume_rendering(rgb, sigma, t_vals, rays_d)

    # 4. Mise en forme en image
    img = rgb_map.view(H, W, 3).cpu().numpy()
    img = np.clip(img, 0, 1)
    return img

def _normalize(v, eps=1e-8):
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

def pose_spherical(theta, phi, radius):
    cam_pos = torch.tensor([
        radius * np.cos(np.radians(phi)) * np.sin(np.radians(theta)),
        radius * np.sin(np.radians(phi)),
        radius * np.cos(np.radians(phi)) * np.cos(np.radians(theta))
    ], dtype=torch.float32)

    forward = _normalize(-cam_pos.unsqueeze(0)).squeeze(0) 
    up_world = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    back  = -forward
    right = _normalize(torch.cross(up_world, back, dim=0))
    up    = torch.cross(back, right, dim=0)

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = right  
    c2w[:3, 1] = up    
    c2w[:3, 2] = back   
    c2w[:3, 3] = cam_pos
    return c2w