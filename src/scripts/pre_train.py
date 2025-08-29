from src.utils.model import *
from src.conditional_flow_matching import ConditionalFlowMatcher
from file_config import FLOWBACK_JOBDIR, FLOWBACK_INPUTS, FLOWBACK_BASE
import argparse
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils as nn_utils
import pytorch_lightning as pl
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from copy import deepcopy
import argparse
from argparse import ArgumentParser
from typing import Tuple, Any, Dict
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import psutil
import shutil
from sklearn.model_selection import train_test_split




def setup_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Set up command line arguments for the application.
    
    Args:
        parser: ArgumentParser instance to add arguments to
        
    Returns:
        ArgumentParser with added arguments
    """
    
    parser.add_argument('--config', type=str, default=f'{FLOWBACK_BASE}/configs/config.yaml', help='Path to config file')

    
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


def get_args() -> Tuple[Any, Any]:
    """
    Parse command line arguments and load configuration from YAML file.
    
    Returns:
        Tuple containing parsed command line arguments and configuration arguments
    """
    parser = ArgumentParser(description='BoltzmannFlow')
    parser = setup_args(parser=parser)  # Note: This seems recursive - might want to rename
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert config dictionary to args object  
    config_args = config_to_args(config)
    
    return args, config_args

args, config_args = get_args()

config_yaml = args.config

job_dir = f'{FLOWBACK_JOBDIR}/{config_args.save_dir}'
os.makedirs(job_dir, exist_ok=True)

# load different systems with max_atoms and encoding dim to ensure features will fit
load_path = config_args.load_path
top_path = config_args.top_path
if load_path == 'default':
    load_dict = pkl.load(open(f'{FLOWBACK_INPUTS}/train_features/feats_pro_0-1000_all_max-8070.pkl', 'rb')) 
    top_list = pkl.load(open(f'{FLOWBACK_INPUTS}/train_features/tops_pro_0-1000_all.pkl', 'rb'))
else:
    load_dict = pkl.load(open(load_path, 'rb')) 
    top_list= pkl.load(open(top_path, 'rb')) 


# standard for 20-residue proteins up to 1000 residues
max_atoms = config_args.max_atoms  # max atoms in training set
res_dim = config_args.res_dim
atom_dim = config_args.atom_dim

device = config_args.device
sigma = config_args.fmsigma
Ca_std = config_args.CG_noise
batch_size = config_args.batch
n_epochs = config_args.eps
eval_every = config_args.evalf
lr = config_args.lr
wdecay = config_args.wdecay
lrdecay = config_args.lrdecay
depth = config_args.depth
num_nearest_neighbors = config_args.nneigh
dim = config_args.dim
loss_type = config_args.loss
mdim = config_args.mdim
clamp = config_args.clamp
attnevery = config_args.attnevery
CGadj = config_args.CGadj
system = config_args.system
pos = config_args.pos
solver = config_args.solver
batch_pack = config_args.batch_pack
diff_type = config_args.diff_type
load_path = config_args.load_path
top_path = config_args.top_path
# test_size = config_args.test_size
sym = config_args.sym
sym_rep = config_args.sym_rep
mask_prior = config_args.mask_prior
pos_cos = config_args.pos_cos
seq_feats = config_args.seq_feats
seq_decay = config_args.seq_decay
act = config_args.act
grad_clip = config_args.grad_clip

max_train = config_args.max_train
max_val = config_args.max_val


# idxs = np.arange(len(load_dict['res']))
# train_idxs, valid_idxs = train_test_split(idxs, test_size=0.01, random_state=42)
train_idxs = np.load(f'{FLOWBACK_INPUTS}/train_features/idxs_train_pro.npy')[:max_train]
valid_idxs = np.load(f'{FLOWBACK_INPUTS}/train_features/idxs_valid_pro.npy')[:max_val]

# save hyperparams to pkl to reload model
params_dict = { 'depth': depth,
                'num_nearest_neighbors': num_nearest_neighbors,
                'dim': dim, 
                'mdim': mdim,
                'max_atoms': max_atoms,
                'res_dim': res_dim,
                'atom_dim': atom_dim,
                'sym':sym,
                'pos_cos': pos_cos,
                'seq_feats':seq_feats,
                'seq_decay':seq_decay,
                'act':act,
                'grad_clip':grad_clip,
               'train_idxs': train_idxs,
               'valid_idxs': valid_idxs
                }

pkl.dump(params_dict, open(f'{job_dir}/params.pkl', 'wb'))
shutil.copyfile(config_yaml, f'{job_dir}/config.yaml')
# reformat CG mask
masks = []
for res, m_idxs in zip(load_dict['res'], load_dict['mask']):
    mask = np.ones(len(res))
    mask[m_idxs] = 0
    masks.append(mask) 

# whether or not to include a positional embedding
if pos==1:
    pos_emb = max_atoms
elif pos==0:
    pos_emb = None

# whether or no to include sin/cos pos embedding 
if pos_cos < 0.001:
    pos_cos = None

model = EGNN_Network_time(
    num_tokens = res_dim,
    num_positions = pos_emb,
    dim = dim,               
    depth = depth,
    num_nearest_neighbors = num_nearest_neighbors,
    global_linear_attn_every = attnevery,
    coor_weights_clamp_value = clamp,  
    m_dim= mdim,
    fourier_features = 4, 
    time_dim=0,
    res_dim=res_dim,
    atom_dim=atom_dim,
    sym= sym,
    emb_cos_scale=pos_cos,
    seq_feats=seq_feats,
    seq_decay=seq_decay,
    act=act
).to(device)

cur_epoch = int(config_args.checkpoint)
if hasattr(config_args, 'model_path') and config_args.model_path is not None:
    model.load_state_dict(torch.load(config_args.model_path))
elif config_args.checkpoint > 0:
    model.load_state_dict(torch.load(f'{job_dir}/state-{config_args.checkpoint}.pth'))


# should be able to remove cFM here
FM = ConditionalFlowMatcher(sigma=sigma)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay)
if lrdecay > 0.0:
    scheduler = StepLR(optimizer, step_size=1, gamma=lrdecay)

# set xyz_tru directly
xyz_true = load_dict['xyz']

# fix xyz for dna trajs
if xyz_true[0].shape[0] == 1:
    xyz_true = [xyz[0] for xyz in xyz_true]

loss_list = []
rtp_data, lj_data, bond_data = get_ff_data()

for epoch in range(cur_epoch + 1, n_epochs):
    ca_pos = get_ca_pos(xyz_true, load_dict['map'])
    # ensures using new noise profile at each epoch
    prior = [np.random.randn(xyz_true[i].shape[0], xyz_true[i].shape[1]) * Ca_std for i in train_idxs]
  
    xyz_diff = [xyz_true[i] - ca_pos[i] for i in train_idxs]
    train_dataset = StructureDataset(xyz_diff, prior, 
                                   [load_dict['res'][i] for i in train_idxs], [load_dict['atom'][i] for i in train_idxs], 
                                   [ca_pos[i] for i in train_idxs], [masks[i] for i in train_idxs])

    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    mean_loss = []
    
    for i, (x1, x0, res_feats, atom_feats, ca_p, mask) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        x1 = x1.to(device)
        x0 = x0.to(device)
        res_feats = res_feats.to(device)
        atom_feats = atom_feats.to(device)
        # esm_feats = esm_feats.to(device)
        mask = mask.to(device)
        ca_p = ca_p.to(device)
        
        # maximize batch size based on molecule size
        if batch_pack == 'max':
            time_batch = (max_atoms // len(res_feats[0])) * batch_size
        elif batch_pack == 'uniform':
            time_batch = batch_size
           
        # repeat values over time batch
        x1 = x1.repeat(time_batch, 1, 1)
        x0 = x0.repeat(time_batch, 1, 1)
        res_feats = res_feats.repeat(time_batch, 1)
        atom_feats = atom_feats.repeat(time_batch, 1)
        mask = mask.repeat(time_batch, 1)

        # replace with FM code
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        
        t_pad = t.reshape(-1, *([1] * (xt.dim() - 1)))
        epsilon = torch.randn_like(xt)
        xt_mask =  t_pad * x1 + (1 - t_pad) * x0
        
        # calculate sigma_t as in stochastic interpolants
        sigma_t = sigma
        
        # only add noise to unmasked values
        extended_mask = torch.unsqueeze(mask.int(), -1)
        extended_mask = torch.repeat_interleave(extended_mask, 3, dim=-1)
        xt_mask += sigma_t * epsilon * extended_mask
        
        # pred the structure
        _, xt_pred = model(res_feats, xt_mask, time=t, atom_feats=atom_feats, mask = mask, ca_pos=ca_p)
       
        if diff_type == 'xt_mask':
            vt = xt_pred - xt_mask
        elif diff_type == 'xt':
            vt = xt_pred - xt
        elif diff_type == 'x0':
            vt = xt_pred - x0
       
        if loss_type == 'L2':
            loss = torch.mean((vt - ut) ** 2)
        elif loss_type == 'L1':
            loss = torch.mean(torch.abs(vt - ut))
        
        loss.backward()

        # add clipping
        if grad_clip > 0.001:
            nn_utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        mean_loss.append(loss.item())
        
    print('Epoch:', epoch, 'Loss:', np.mean(mean_loss))
    loss_list.append(np.mean(mean_loss))
    
    # update lr scheduler if included
    if config_args.lrdecay > 0.0:
        scheduler.step()
    
    # get bond quality (and clash) every N epochs
    if epoch%eval_every==0 and epoch>0:
        
        # can iterate over this and test one at a time
        n_gens = 1    
        bf_list = []
        cls_list = []
        
        for idx in valid_idxs:

            xyz_test_real = [xyz_true[idx]]
            map_test = [load_dict['map'][idx]]

            ca_test_pos = get_ca_pos(xyz_test_real, map_test)
            
            xyz_test_diff = np.array([xyz_test_real[0] - ca_test_pos[0]])
            # if top_list not provided, load from main dict
            try:
                top = top_list[idx]
            except:
                top = load_dict['top'][idx]
                
         
            prior = [np.random.randn(xyz_test_diff.shape[1], xyz_test_diff.shape[2]) * Ca_std]

            model_wrpd = ModelWrapper(model=model, 
                              feats=torch.tensor(np.array([load_dict['res'][idx]])).int().to(device), 
                              mask=torch.tensor(np.array([masks[idx]])).bool().to(device).to(device), 
                              atom_feats=torch.tensor(np.array([load_dict['atom'][idx]])).to(device),
                              ca_pos=torch.tensor(np.array(ca_test_pos)).to(device))
            with torch.no_grad():
                if solver == 'euler':
                    ode_traj = euler_integrator(model_wrpd, torch.tensor(prior,
                                                                dtype=torch.float32).to(device))
                elif solver == 'euler_ff':
                    ode_traj = euler_ff_integrator(model_wrpd, torch.tensor(prior,
                                                                dtype=torch.float32).to(device), torch.tensor(ca_test_pos).to(device), rtp_data, lj_data, bond_data, top, device)
            print_memory_usage()           
            # assume we're working with one structure at a time
            xyz_gens = ode_traj[-1] + ca_test_pos[0]
            xyz_ref = xyz_true[idx]
            
            
            
            print(xyz_gens.shape, xyz_ref.shape, top.n_atoms)
            
            # need n_atoms to account for pro-dna case
            trj_gens = md.Trajectory(xyz_gens[:, :top.n_atoms], top)
            trj_ref = md.Trajectory(xyz_ref[:top.n_atoms], top)
            bf_list += [bond_fraction(trj_ref, trj_gen) for trj_gen in trj_gens]
            
            
            try: cls_list += [clash_res_percent(trj_gen) for trj_gen in trj_gens]
            except: print('Failed', [res for res in top.residues])
            
            np.save(f'{job_dir}/ode-{epoch}_f-{idx}.npy', ode_traj)
                
        np.save(f'{job_dir}/bf-{epoch}.npy', bf_list)
        np.save(f'{job_dir}/cls-{epoch}.npy', cls_list)
    np.save(f'{job_dir}/losses-epoch-{epoch}.npy', np.array(loss_list)) 
    # save ode outputs for visualization
    torch.save(model.state_dict(), f'{job_dir}/state-{epoch}.pth')
        
  
   

