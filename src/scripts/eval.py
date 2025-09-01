### input directory to pdbs/trajs and return N generated samples of each ###

import os
from file_config import FLOWBACK_OUTPUTS, FLOWBACK_DATA, FLOWBACK_MODELS

import argparse
import glob
import pickle as pkl
from tqdm import tqdm
import time
import datetime

# need to test these for preproccessing
from src.utils.evaluation import *

# import functions to check and correct chirality
from src.utils.chi import *

parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', default='PDB', type=str, help='Path to input pdbs -- Can be AA or CG')
parser.add_argument('--CG_noise', default=0.003, type=float, help='Noise profile to use as prior')
parser.add_argument('--ckp', default=14, type=int, help='Checkpoint for given mode')
parser.add_argument('--n_gens', default=1, type=int, help='N generated samples per structure')
parser.add_argument('--solver', default='euler_ff', type=str, help='Which type of ODE solver to use')
parser.add_argument('--stride', default='1', type=int, help='Stride applie to trajectories')
parser.add_argument('--check_clash', action='store_true',  help='Calculate clash for each sample')
parser.add_argument('--check_bonds', action='store_true',  help='Calculate bond quality for each sample')
parser.add_argument('--check_div', action='store_true',  help='Calculate diversity score (for multi-gen)')
parser.add_argument('--mask_prior', action='store_true',  help='Enforce CG positions remain the same')
parser.add_argument('--retain_AA', action='store_true',  help='Hold AA positions for scoring')
parser.add_argument('--model_path', default='../models/Pro_pretrained', type=str, help='Trained model')
parser.add_argument('--tolerance', default=3e-5, type=float, help='Tolerance if using NN solver')
parser.add_argument('--nsteps', default=100, type=int, help='Number of steps in Euler integrator')
parser.add_argument('--system', default='pro', type=str, help='Pro or DNAPro CG input')
parser.add_argument('--save_dir', default='', type=str, help='Where to save structures')
parser.add_argument('--external', action='store_true',  help='load_dir is outside this directory')


parser.add_argument('--vram', default=32, type=int, help='Scale batch size to fit max gpu VRAM')
parser.add_argument('--save_traj', action='store_true',  help='Save all flow-matching timesteps')
parser.add_argument('--save_dcd', action='store_true',  help='Save traj output as a pdb+dcd')
parser.add_argument('--overwrite', action='store_true',  help='Save traj output as a pdb+dcd')

# for enantiomer correction
parser.add_argument('--t_flip', default=0.2, type=float,  help='ODE time to correct fo D-residues')
parser.add_argument('--type_flip', type=str, default='ref-ter', help='Method used to fix chirality')

args = parser.parse_args()
load_dir = args.load_dir
CG_noise = args.CG_noise
model_path = args.model_path
ckp = args.ckp
n_gens = args.n_gens
solver = args.solver
stride = args.stride
check_clash = args.check_clash
check_bonds = args.check_bonds
check_div = args.check_div
mask_prior = args.mask_prior
retain_AA = args.retain_AA
tol = args.tolerance
nsteps = args.nsteps
t_flip = args.t_flip
type_flip = args.type_flip
system = args.system
vram = args.vram
save_traj = args.save_traj
save_dcd = args.save_dcd
overwrite = args.overwrite

if args.save_dir == '':
    save_dir = f'../outputs/{load_dir}'
    if solver == 'euler_ff':
        # save_prefix = f'{save_dir}/{model_path.split("/")[-1]}_ckp-{ckp}_noise-{CG_noise}/'
        save_prefix = f'{save_dir}/{model_path.split("/")[-1]}_ckp-{ckp}_noise-{CG_noise}_chi_{t_flip}/'
    elif solver == 'euler':
        save_prefix = f'{save_dir}/{model_path.split("/")[-1]}_ckp-{ckp}_euler_noise-{CG_noise}/'
    elif solver == 'euler_chi':
        save_prefix = f'{save_dir}/{model_path.split("/")[-1]}_ckp-{ckp}_euler_noise-{CG_noise}_chi_{t_flip}/'
    # if noise_flow:
    #     save_prefix = f'{save_prefix[:-1]}_noised/'
    # if mask_prior:
    #     save_prefix = save_prefix[:-1] + '_masked/'
else:
    save_dir = args.save_dir + '/'
    save_prefix = args.save_dir + '/'
    # remove mask and nsteps to simplify naming





if not args.external:
    load_dir = f'data/{load_dir}'


os.makedirs(save_prefix, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# add optional preprocessing to save files -- skip if already cleaned
if system == 'pro':
    if retain_AA:
        load_dir = process_pro_aa(load_dir)
    else:
        load_dir = process_pro_cg(load_dir)
elif system == 'DNApro':
    if retain_AA:
        # accounted for in featurization
        pass
    else:
        pass
        # either combine standard order with protein function (for aa and cg)
        # or call each seperately and recombine (keep pro-DNA order consistent)
        #load_dir = process_DNApro_cg(load_dir)
        
else:
    print('Invalid system type')

# load model
model = load_model(model_path, ckp, device) #, sym, pos_cos, seq_feats, seq_decay)
print('params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# Track scores
bf_list, clash_list, div_list = [], [], []

# save time for inference as a function of size -- over n-res?
time_list, res_list = [], []
print(load_dir)
trj_list = sorted(glob.glob(f'{load_dir}/*.pdb'))
print(f'Found {len(trj_list)} trajs to backmap')

rtp_data, lj_data, bond_data = get_ff_data()

for trj_name in tqdm(trj_list, desc='Iterating over trajs'):
    save_name = f'{save_prefix}{trj_name.split("/")[-1]}'
    if save_traj:
        save_fn = save_name.replace('.pdb', f'_dt.pdb')
    else:
        save_fn = save_name.replace('.pdb', f'_1.pdb')

    if overwrite or not os.path.exists(save_fn):
        trj = md.load(trj_name)[::stride]
        n_frames = trj.n_frames
        start_time = datetime.datetime.now()
    
        # load features for the given topology and system type
        if system=='pro':
            res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_pro(trj)
        elif system=='DNApro':
            res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_DNApro(trj)
        else:
            print('Invalid system type')
    
        test_idxs = list(np.arange(n_frames))*n_gens
        xyz_ref = xyz[test_idxs]
         
        print(f'{trj_name.split("/")[-1]}   {n_frames} frames   {n_atoms} atoms   {n_gens} samples')
    
        # ensure input will fit into specified VRAM (16GB by default)
        n_iters = int(len(test_idxs) * len(res_ohe) / (vram*6_000)) + 1
        idxs_lists = split_list(test_idxs, n_iters)
        print(f'breaking up into {n_iters} batches:\n')
              
        xyz_gen = []
        prior = np.random.randn(n_gens, xyz_ref.shape[1], xyz_ref.shape[2]) * CG_noise
        for n, test_idxs in enumerate(idxs_lists):
            # print(test_idxs)
            n_test = len(test_idxs)
            print(f'iter {n+1} / {n_iters}')
            
            xyz_test_real = [xyz[i] for i in test_idxs]
            # print([x_[0] for x_ in xyz_test_real])
            map_test =      [aa_to_cg]*n_test
            mask_test =     [mask]*n_test
            res_test =      [res_ohe]*n_test
            atom_test =     [atom_ohe]*n_test
            ca_pos_test =   get_ca_pos(xyz_test_real, map_test)
            # wrap model -- update this so that the function multiplies by the dim of n_gens * n_frames 
            model_wrpd = ModelWrapper(model=model, 
                            feats=torch.tensor(np.array(res_test)).int().to(device), 
                            mask=torch.tensor(np.array(mask_test)).bool().to(device), 
                            atom_feats=torch.tensor(np.array(atom_test)).to(device),
                            ca_pos=torch.tensor(np.array(ca_pos_test)).to(device))
    
           
            with torch.no_grad():
                if solver == 'euler_chi':
                    ode_traj = euler_integrator_chi_check(model_wrpd, 
                                      torch.tensor(prior, dtype=torch.float32).to(device), torch.tensor(ca_pos_test).to(device),
                                          nsteps=nsteps, t_flip=t_flip, top_ref=top, type_flip=type_flip)
                elif solver == 'euler_ff':
                    ode_traj = euler_ff_integrator(model_wrpd, torch.tensor(prior,
                                                                    dtype=torch.float32).to(device), torch.tensor(ca_pos_test).to(device), rtp_data, lj_data, bond_data, top, device, t_flip=t_flip) #, t_flip=t_flip, type_flip=type_flip
           
            # elif solver == 'euler_chi':
            #     with torch.no_grad():
            #         ode_traj, chi_list = euler_integrator_chi_check(model_wrpd, 
            #                               torch.tensor(xyz_test_prior, dtype=torch.float32).to(device), 
            #                               nsteps=nsteps, 
            #                               t_flip=t_flip,
            #                               top_ref=top,
            #                               type_flip=type_flip)
              
            # end time and save
            time_diff = datetime.datetime.now() - start_time
            time_list.append(time_diff.total_seconds())
            res_list.append(trj.n_residues)
            print(ode_traj.shape)
            
            # save trj -- optionally save ODE integration not just last structure -- only for one gen
            if save_traj:
                xyz_gen.append(ode_traj.squeeze() + ca_pos_test)
            else:
                xyz_gen.append(ode_traj[-1] + ca_pos_test) 

        # trj = md.Trajectory(ode_traj.squeeze() + ca_pos_test, top)
        # chi = get_all_chiralities_vec(trj)
        # print(chi[:, 18], chi[-1])
        print(np.shape(xyz_gen))
        xyz_gen = np.concatenate(xyz_gen)
        print(xyz_gen.shape)
              
        # don't include DNA virtual atoms in top 
        aa_idxs = top.select(f"not name DS and not name DP and not name DB")
        trj_gens = md.Trajectory(xyz_gen[:, :top.n_atoms], top).atom_slice(aa_idxs)
        trj_refs = md.Trajectory(xyz_ref[:, :top.n_atoms], top).atom_slice(aa_idxs)
        print(trj_gens.xyz.shape)
    
        # Can only calculate bonds and div if an AA reference is provided
        if check_bonds:
            bf = [bond_fraction(t_ref, t_gen) for t_gen, t_ref in zip(trj_gens, trj_refs)]
            bf = np.array(bf).reshape(n_frames, n_gens)
            bf_list.append(bf)
              
        # protein only clash only for now
        if check_clash:
            print(trj_gens[0])
            clash = [clash_res_percent(t_gen) for t_gen in trj_gens]
            clash = np.array(clash).reshape(n_frames, n_gens)
            clash_list.append(clash)
              
        # Need multiple gens to calculate diversity
        if check_div:
            div_frames = []
            for f in range(n_frames):
                trj_ref_div = trj_refs[f]
                trj_gens_div = trj_gens[f::n_frames]
                div, _ = sample_rmsd_percent(trj_ref_div, trj_gens_div)
                div_frames.append(div)
            div_list.append(div_frames)
    
        # save gen using same pdb name -- currently saving as n_frames * n_gens
        save_name = f'{save_prefix}{trj_name.split("/")[-1]}'
              
        # if saving dt, only save a single gen
        
        if save_traj:
            save_i = save_name.replace('.pdb', f'_dt.pdb')
            trj_gens.save_pdb(save_i)
            print(save_i)
        else:
            for i in range(n_gens):
                save_i = save_name.replace('.pdb', f'_{i+1}.pdb')
                if save_dcd:
                    trj_gens[i*n_frames].save_pdb(save_i)
                    trj_gens[i*n_frames].save_dcd(save_i.replace('.pdb', '.dcd'))
                else:
                    trj_gens[i*n_frames:(i+1)*n_frames].save_pdb(save_i)

# save all scores to same dir
if check_bonds:
    np.save(f'{save_prefix}bf.npy', np.array(bf_list))
    print('mean bf: ', np.mean(bf_list))
if check_clash:
    np.save(f'{save_prefix}cls.npy', np.array(clash_list)) 
    print('mean cls: ', np.mean(clash_list))
if check_div:
    np.save(f'{save_prefix}div.npy', np.array(div_list)) 
    print('mean div: ', np.mean(div_list))
          
# save ordered list of trajs
with open(f'{save_prefix}trj_list.pkl', "wb") as output_file:
    pkl.dump(trj_list, output_file)
     
try:
    np.save(f'{save_prefix}time_gen-{n_gens}.npy', np.array([res_list, time_list]))     
except:
    pass

print(f'\nSaved to:  {save_prefix}\n')

#
