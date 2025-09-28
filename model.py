import torch.nn as nn
import torch.nn.functional as F
import torch
import math

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

import torch

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
