### input directory to pdbs/trajs and return N generated samples of each ###
import os
from file_config import FLOWBACK_JOBDIR, FLOWBACK_DATA, FLOWBACK_BASE
import argparse
import glob
import pickle as pkl
from tqdm import tqdm
import time
import datetime
# need to test these for preproccessing
from src.utils.evaluation import *
from src.adjoint import *
# import functions to check and correct chirality
from src.utils.chi import *
from torch.utils.data import Dataset, DataLoader, Subset
from src.conditional_flow_matching import ConditionalFlowMatcher
import argparse
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils as nn_utils
import pytorch_lightning as pl
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
import argparse
from argparse import ArgumentParser
from typing import Tuple, Any, Dict
import yaml
import shutil
import gc
import subprocess

import random, numpy as np, torch

seed = 234
random.seed(seed)            # Python RNG
np.random.seed(seed)         # NumPy RNG
torch.manual_seed(seed)      # PyTorch CPU RNG
torch.cuda.manual_seed_all(seed)   # (optional) all GPU RNGs
torch.backends.cudnn.deterministic = True   # for deterministic convs
torch.backends.cudnn.benchmark = False

def setup_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Set up command line arguments for the application.
    
    Args:
        parser: ArgumentParser instance to add arguments to
        
    Returns:
        ArgumentParser with added arguments
    """
    
    parser.add_argument('--config', type=str, default=f'{FLOWBACK_BASE}/configs/pt_config.yaml', help='Path to config file')
    parser.add_argument('--test', action="store_true", help='For testing purposes')
    parser.add_argument('--compare', action='store_true', help='Just for printing energies')
    return parser


def config_to_args(config):
    """
    Convert a configuration dictionary into an argparse Namespace object.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        argparse.Namespace: Args object with attributes based on the config.
    """
    return argparse.Namespace(**config)

def graph_size_and_depth(loss: torch.Tensor):
    """
    Return (#Function nodes, max depth) of the autograd graph rooted at `loss`.
    Call this *before* loss.backward() unless you used retain_graph=True.
    """
    if loss.grad_fn is None:
        raise ValueError("`loss` has no grad_fn – did you call .backward() already?")

    seen = set()

    def dfs(fn, depth=1):
        if fn is None or fn in seen:
            return 0, depth - 1        # 0 extra nodes, depth so far
        seen.add(fn)

        total_nodes = 1                # count this node
        max_d = depth

        for nxt, _ in fn.next_functions:
            n, d = dfs(nxt, depth + 1)
            total_nodes += n
            max_d = max(max_d, d)

        return total_nodes, max_d

    return dfs(loss.grad_fn)


def get_args() -> Tuple[Any, Any]:
    """
    Parse command line arguments and load configuration from YAML file.
    
    Returns:
        Tuple containing parsed command line arguments and configuration arguments
    """
    parser = ArgumentParser(description='BoltzmannFlow')
    parser = setup_args(parser=parser) 
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert config dictionary to args object  
    config_args = config_to_args(config)
    
    return args, config_args

def _select_timesteps(N, split=0.8, num_samples=10, seed=None):
    """
    Randomly select timesteps from a range [0, N-1]
    
    Parameters:
    -----------
    N : int
        The size of the range [0, N-1]
    num_samples : int
        Number of timesteps to sample from the first 80% of the range
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    list
        Selected timesteps (10 random from first 80% + all from last 20%)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate the 80% mark
    split_idx = int(split * N)
    
    # First 80% of timesteps
    first_80_percent = np.arange(0, split_idx + 1)
    
    # Last 20% of timesteps
    last_20_percent = np.arange(split_idx + 1, N + 1)
    
    # Randomly select 10 timesteps from the first 80%
    selected_from_first = np.random.choice(first_80_percent, size=min(num_samples, len(first_80_percent)), replace=False)
    
    # Combine the selections
    selected_timesteps = np.concatenate([selected_from_first, last_20_percent])
    
    # Sort the timesteps in ascending order
    np.random.shuffle(selected_timesteps)
    
    return torch.tensor(selected_timesteps)


if __name__ == '__main__':
    args, config_args = get_args()
    
    config_yaml = f'{FLOWBACK_BASE}/configs/config.yaml'
    # for enantiomer correction
    
    load_dir = config_args.load_dir
    CG_noise = config_args.CG_noise
    Ca_std = CG_noise
    model_path = config_args.model_path
    ckp = config_args.ckp
    n_gens = config_args.n_gens
    n_epochs = config_args.n_epochs
    stride = config_args.stride
    batch_size = config_args.batch
    
    mask_prior = config_args.mask_prior
    retain_AA = config_args.retain_AA
    nsteps = config_args.nsteps
    vram = config_args.vram
    
    lr = config_args.lr
    wdecay = config_args.wdecay 
    grad_clip = config_args.grad_clip
    lam = config_args.lam
    n_samples = config_args.n_samples
    save_dir = config_args.save_dir
    eval_every = config_args.evalf
    num_steps = config_args.num_steps
    pdb_list = config_args.pdb_list
    restart = config_args.restart
    
    if hasattr(config_args, 'max_grad'):
        MAX_GRAD = config_args.max_grad
    else:
        MAX_GRAD = 5e3

    
    if hasattr(config_args, 't_flip'):
        t_flip = config_args.t_flip
    else:
        t_flip = 0.55

    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # add optional preprocessing to save files -- skip if already cleaned
    
    
    
   
    model = load_model(model_path, ckp, device)  
    
    model.eval()                        # turn off dropout / BN updates
    model.requires_grad_(False)         # no gradients or optimizer states
 
    model_ft = deepcopy(model).to(device)   # device == "cuda:0", etc.
    model_ft.train()                        # enable training mode
    model_ft.requires_grad_(True)           # grads for *this* copy
    
    
    
    optimizer = torch.optim.AdamW(model_ft.parameters(), lr=lr, weight_decay=wdecay)
    lr_scheduler = StepLR(optimizer, step_size=80, gamma=0.5)
    # optimizer = torch.optim.SGD(model_ft.parameters(), lr=lr)
    
    optimizer.zero_grad()
    
    
    
    # save time for inference as a function of size -- over n-res?
    
    job_dir = f'{FLOWBACK_JOBDIR}/{save_dir}_post'
    os.makedirs(job_dir, exist_ok=True)
    
    all_dirs = load_dir.split()
    full_trj_list = []
    for dir in all_dirs:
        ldir = f'{FLOWBACK_DATA}/{dir}'
        if retain_AA:
            ldir = process_pro_aa(ldir)
        else:
            ldir = process_pro_cg(ldir)
    
    
        if pdb_list == 'default':
            full_trj_list.extend(sorted(glob.glob(f'{ldir}/*.pdb')))
        else:
            with open(pdb_list, 'r') as f:
                full_trj_list.extend([f'{ldir}/{line[:-1]}.pdb' for line in f.readlines()])
    
    np.random.shuffle(full_trj_list)

    if restart:
        cur_epoch = config_args.restart_ckp
        
        model_ft.load_state_dict(torch.load(f'{job_dir}/state-{cur_epoch}.pth'))
        start_epoch = cur_epoch + 1
    else:
        start_epoch = 0
        
    
    
    
    
    if args.test:
        full_trj_list = full_trj_list[:100]
    idxs = np.arange(len(full_trj_list))
    # train_idxs, valid_idxs = train_test_split(idxs, test_size=0, random_state=42)
    
    
    xyz_true = []
    res_list = []
    atom_list = []
    mask_list = []
    top_list = []
    aa_to_cg_list = []
    ca_pos = []
    
    
    shutil.copyfile(config_yaml, f'{job_dir}/config.yaml')
    if hasattr(config_args, 'ff') and config_args.ff == 'CHARMM':
        ff = 'CHARMM'
    else:
        ff = 'RDKit'

    int_ff = hasattr(config_args, 'int_ff') and config_args.int_ff == True
    
    acc_grad_batch = 1
    if hasattr(config_args, 'acc_grad_batch'):
        acc_grad_batch = config_args.acc_grad_batch
        
    loss_list = []
    MAX_STRUCTURES = 1000
    
    
    if args.test or len(full_trj_list) < 1000:
        multi_trj_list = [full_trj_list]
    else:
        multi_trj_list = split_list(full_trj_list, len(full_trj_list) // 1000)
    
    for epoch in range(start_epoch, n_epochs):
        print(f'Epoch {epoch}')
        epoch_step = 0
        for trj_list in multi_trj_list:
            for trj_name in trj_list:
                trj = md.load(trj_name)
                n_frames = trj.n_frames
                res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_pro(trj)
                res_list.append(res_ohe)
                atom_list.append(atom_ohe)
                xyz_true.append(xyz)
                mask_list.append(mask)
                top_list.append(top)
                ca_pos.extend(xyz[:, aa_to_cg])
    
            train_idxs = np.arange(len(trj_list))
            # ensures using new noise profile at each epoch
            # prior = [np.random.randn(xyz_true[i].shape[0], xyz_true[i].shape[1]) * Ca_std for i in train_idxs]
        
            # xyz_diff = [xyz_true[i] - ca_pos[i] for i in train_idxs]
            pt_dataset = PostTrainDataset([res_list[i] for i in train_idxs], [atom_list[i] for i in train_idxs], 
                                           [ca_pos[i] for i in train_idxs], [mask_list[i] for i in train_idxs])
        
            shuffle_idxs = torch.randperm(len(pt_dataset))
            shuffle_set = Subset(pt_dataset, shuffle_idxs)
            train_loader = DataLoader(shuffle_set, batch_size=1, shuffle=False, drop_last=True)
            
            mean_loss = []
            bf_list = []
            cls_list = []
            for i, ((res_feats, atom_feats, ca_p, mask), top) in tqdm(enumerate(zip(train_loader, [top_list[i] for i in shuffle_idxs]))):
                # if res_feats.shape[1] > 6000:
                res_feats = res_feats.to(device)
                atom_feats = atom_feats.to(device)
                # esm_feats = esm_feats.to(device)
                mask = mask.to(device)
                ca_p = ca_p.to(device)
    
                batch = 1
                # x1 = x1.repeat(batch, 1, 1)
                # x0 = x0.repeat(batch, 1, 1)
                res_feats = res_feats.repeat(batch, 1)
                atom_feats = atom_feats.repeat(batch, 1)
                mask = mask.repeat(batch, 1)
                # md.Trajectory(x1.cpu() + ca_p.cpu(), top).save_pdb(f'{job_dir}/debug.pdb')
        
                model_wrpd  =   ModelWrapper(model=model, 
                                feats=res_feats, 
                                mask=mask, 
                                atom_feats=atom_feats,
                                ca_pos=ca_p)
                model_ft_wrpd = ModelWrapper(model=model_ft, 
                                feats=res_feats, 
                                mask=mask, 
                                atom_feats=atom_feats,
                                ca_pos=ca_p)
        
        
                if grad_clip > 0.001:
                    nn_utils.clip_grad_norm_(model_ft.parameters(), grad_clip)
        
                kwargs = {
                    'job_dir': job_dir,
                    'topology': top,
                    'device': device,
                    'lam': lam,
                    'n_coords': res_feats.shape[1],
                    'cg_noise': Ca_std,
                    'batch_size': batch,
                    'ca_pos': ca_p.cpu().detach().numpy(),
                    'num_steps': num_steps,
                    'ff': ff,
                    'int_ff': int_ff,
                    'max_grad': MAX_GRAD,
                    't_flip': t_flip
                }
                loss = 0

                if args.compare:
                    try:
                        traj, a_t = trajectory_and_adjoint(model_wrpd, model_ft_wrpd, **kwargs)
                    except:
                        continue
                elif args.test:
                    traj, a_t = trajectory_and_adjoint(model_wrpd, model_ft_wrpd, **kwargs)
                    select_steps = _select_timesteps(num_steps, config_args.selection_split).to(device)
                    sigma_select = sigma(select_steps / num_steps, 1 / num_steps, Ca_std)
                    max_steps = atom_to_steps(res_feats.shape[1])
                    step_list = split_list(select_steps, int(np.ceil(len(select_steps) / max_steps)))
                    sigma_list = split_list(sigma_select, int(np.ceil(len(select_steps) / max_steps)))
                    for steps, sigmas in zip(step_list, sigma_list):
                        loss = adjoint_matching_loss(traj, model_ft_wrpd, model_wrpd, a_t, steps, sigmas, **kwargs)
                        print('Loss', loss.item())
                        loss.backward()
                    if (i+1) % acc_grad_batch == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    mean_loss.append(loss.item())
                else:
                    try:
                        traj, a_t = trajectory_and_adjoint(model_wrpd, model_ft_wrpd, **kwargs)
                        select_steps = _select_timesteps(num_steps, config_args.selection_split).to(device)
                        sigma_select = sigma(select_steps / num_steps, 1 / num_steps, Ca_std)
                        max_steps = atom_to_steps(res_feats.shape[1])
                        if max_steps < 30:
                            step_list = split_list(select_steps, int(np.ceil(len(select_steps) / max_steps)))
                            sigma_list = split_list(sigma_select, int(np.ceil(len(select_steps) / max_steps)))
                        else:
                            step_list = [select_steps]
                            sigma_list = [sigma_select]
                        for steps, sigmas in zip(step_list, sigma_list):
                            loss = adjoint_matching_loss(traj, model_ft_wrpd, model_wrpd, a_t, steps, sigmas, **kwargs)
                            loss.backward()
                        if (i+1) % acc_grad_batch == 0:
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                        mean_loss.append(loss.item())
                    except:
                        continue
                epoch_step += 1
                if i > 0 and epoch_step % 100 == 0:
                    torch.save(model_ft.state_dict(), f'{job_dir}/state-{epoch * len(full_trj_list) + epoch_step}.pth')
                    loss_list.append(np.mean(mean_loss))
                    np.save(f'{job_dir}/losses-epoch-{epoch}-step-{epoch_step}.npy', np.array(loss_list))
        print(epoch, np.mean(mean_loss))
        loss_list.append(np.mean(mean_loss))
        np.save(f'{job_dir}/losses-epoch-{epoch}.npy', np.array(loss_list))
        
        # save ode outputs for visualization
        torch.save(model_ft.state_dict(), f'{job_dir}/state-{epoch}.pth')
    
    
    
    
