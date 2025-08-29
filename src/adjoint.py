import torch
import torch.nn as nn
import torch.optim as optim
from .utils.model import *
from .utils.energy import *
from datetime import datetime
import numpy as np
import mdtraj as md
import os
import psutil
import gc
from memory_profiler import profile
from .utils.chi import *
from copy import deepcopy

rtp_data, lj_data, bond_data = get_ff_data()

def sigma(t, dt, cg_noise):
    return cg_noise * torch.sqrt((2 * (1 - t + dt)) / (t + dt))

def stochastic_trajectory(v_finetune, sigma_t, **kwargs):
    n_coords = kwargs.get('n_coords')
    batch_size = kwargs.get('batch_size')
    topology = kwargs.get('topology')
    timesteps = kwargs.get('num_steps')
    cg_noise = kwargs.get('cg_noise')
    device = kwargs.get('device')
    int_ff = kwargs.get('int_ff')
    t_flip = kwargs.get('t_flip')
    ca_pos = torch.tensor(kwargs.get('ca_pos'), device=device)
  
    energy_funcs = {
        'RDKit': rdkit_traj_to_energy,
        'CHARMM': charmm_traj_to_energy
    }
    energy_func = energy_funcs[kwargs.get('ff')]
    
    """Simulate trajectory using memoryless noise schedule with alpha_t = t and beta_t = 1 - t."""
    dt = torch.tensor(1.0 / timesteps)
    count = 0
    trajectory = torch.zeros(batch_size, timesteps + 1, n_coords, 3, device=device)
    trajectory[:, 0, :, :] = torch.randn(batch_size, n_coords, 3).to(device) * cg_noise  # Samples from initial Gaussian
    x_t = trajectory[:, 0, :, :]
    res_maps = build_residue_maps(topology)
    
    for t in range(timesteps):
        alpha_t = (t+1) / timesteps
        alpha_prime = 1
        t_val = t / timesteps
        with torch.no_grad():
            if t == timesteps - 1:
                if int_ff:
                    ff_velocity = t_val**20 * torch.stack([
                        0.1 * torch.clamp(lj_velocity_fn(x_t[i:i+1] + ca_pos, t_val, rtp_data, lj_data, dt, topology, device), -5, 5) +
                        torch.clamp(bond_velocity_fn(x_t[i:i+1] + ca_pos, t_val, rtp_data, bond_data, dt, topology, device), -1, 1)
                        for i in range(x_t.shape[0])
                    ])
                    drift = v_finetune(t_val, x_t) + ff_velocity
                else:
                    drift = v_finetune(t_val, x_t)
                diffusion = torch.zeros_like(x_t)
            else:
                if t_val > 0.95 and int_ff:
                    ff_velocity =  t_val**20 * torch.stack([
                        0.1 * torch.clamp(lj_velocity_fn(x_t[i:i+1] + ca_pos, t_val, rtp_data, lj_data, dt, topology, device), -5, 5) +
                        torch.clamp(bond_velocity_fn(x_t[i:i+1] + ca_pos, t_val, rtp_data, bond_data, dt, topology, device), -1, 1)
                        for i in range(x_t.shape[0])
                    ])
                elif t_val > 0.85 and int_ff:
                    ff_velocity = 0.1 * t_val**20 * torch.stack([torch.clamp(lj_velocity_fn(x_t[i:i+1] + ca_pos, t_val, rtp_data, lj_data, dt, topology, device), -5, 5) for i in range(x_t.shape[0])])
                else:
                    ff_velocity = torch.zeros_like(x_t)

                if t_val > t_flip and int_ff:
                    chi_vel = torch.stack([chirality_fix_tensor(x_t[i] + ca_pos[0], res_maps, t_val,
                                                   k_side_init=0.4, k_oxy_init=-0.2) for i in range(x_t.shape[0])])
                    ff_velocity = ff_velocity + chi_vel         # same units: nm/ps
        
                drift = 2 * (v_finetune(t_val, x_t) + ff_velocity) - (alpha_prime / alpha_t) * x_t
                diffusion = sigma_t[t] * torch.randn_like(x_t)

                
            x_t = x_t + dt * drift + torch.sqrt(dt) * diffusion
            trajectory[:, t+1, :] = x_t
    energy_model = EnergyModel(energy_func, topology)
    
    frame = trajectory[:, -1, :, :].requires_grad_(True) + ca_pos
    
    energy = energy_model(frame)
    frame.retain_grad()

    
    energy.backward()
    returned_grads = frame.grad.view(batch_size, n_coords * 3)   

       
    return trajectory, energy.item(), returned_grads



def lean_adjoint_ode(X, v_base, grad, **kwargs):
    job_dir = kwargs.get('job_dir')
    device = kwargs.get('device')
    topology = kwargs.get('topology')
    lam = kwargs.get('lam')
    n_coords = kwargs.get('n_coords')
    timesteps = kwargs.get('num_steps') 
    batch_size = kwargs.get('batch_size')
    max_grad = kwargs.get('max_grad')
    torch.tensor(kwargs.get('ca_pos'), device=device)
    int_ff = kwargs.get('int_ff')
    
    dt = 1.0 / timesteps
    a_t = torch.zeros(batch_size, timesteps+1, n_coords * 3, device=device)
    
    a_t[:, -1, :] =  torch.clamp(lam * grad, -max_grad, max_grad)
    for t in range(timesteps - 1, -1, -1):
        alpha_t = (t+1) / timesteps
        alpha_t_dot = 1
        t_val = t / timesteps
        xgrad = X[:, t, :].requires_grad_(True)

        grad_input = lambda xi: (2 * (v_base(t_val, xi.view(1, n_coords, 3))) - (alpha_t_dot / alpha_t) * xi.view(1, n_coords, 3)).flatten()

        vjp = torch.stack([torch.autograd.functional.vjp(grad_input, xgrad[i].flatten(), a_t[i, t + 1, :])[0] for i in range(batch_size)])
        with torch.no_grad():
            a_t[:, t, :] = torch.clamp((a_t[:, t + 1, :] + dt * vjp).detach(), -max_grad, max_grad)
    return a_t
    
def adjoint_matching_loss(X, v_finetune, v_base, a_t, selected_timesteps, sigma_t, **kwargs):
    device = kwargs.get('device')
    timesteps = kwargs.get('num_steps')
    batch_size = kwargs.get('batch_size')
    n_coords = kwargs.get('n_coords')
    cg_noise = kwargs.get('cg_noise')
    loss = torch.tensor(0.0, device=device)
    
    for k, t in enumerate(selected_timesteps):
        t_val = t / timesteps
        diff = (v_finetune(t_val, X[:, t, :, :]) - v_base(t_val, X[:, t, :, :])).view(batch_size, n_coords * 3)
        loss_t = torch.sum(torch.norm((2 / sigma_t[k]) * diff * (cg_noise ** 2) + sigma_t[k] * a_t[:, t, :], dim=1) ** 2)
        loss += loss_t
        
    return loss


def trajectory_and_adjoint(v_base, v_finetune,  **kwargs):
    n_coords = kwargs.get('n_coords')
    cg_noise = kwargs.get('cg_noise')
    batch_size = kwargs.get('batch_size')
    num_steps = kwargs.get('num_steps')
    device = kwargs.get('device')
    sigma_all = sigma(torch.linspace(0, 1, num_steps + 1), 1 / num_steps, cg_noise)
    xyz, energy, gradients = stochastic_trajectory(v_finetune, sigma_all, **kwargs)
    traj = xyz.view(batch_size, num_steps+1, n_coords, 3)
    a_t = lean_adjoint_ode(traj, v_base, gradients, **kwargs)
    ### Timestep selection
    

    return traj, a_t


    


