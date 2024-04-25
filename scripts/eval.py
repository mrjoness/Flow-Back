### input directory to pdbs/trajs and return N generated samples of each ###

import sys, os
sys.path.append('../')

from utils import *   # only import essentias here to reduce time
import argparse
import glob
import pickle as pkl

from tqdm import tqdm
import time

# need to test these for preproccessing
from eval_utils import process_pro_aa, process_pro_cg

# from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
# from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP
# THREE_TO_ONE_LETTER_MAP = {y: x for x, y in ONE_TO_THREE_LETTER_MAP.items()}

# ATOM_MAP_14 = {}
# for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
#     ATOM_MAP_14[one_letter] = ["N", "CA", "C", "O"] + list(
#         SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
#     ATOM_MAP_14[one_letter].extend(["PAD"] * (14 - len(ATOM_MAP_14[one_letter])))
    
# looks for dir name in ./data, use 'clean' version if exists
# create dir with same name in outputs (no clean)

parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', default='../sidechainnet_data/PDB', type=str, help='Path to input pdbs -- Can be AA or CG')
parser.add_argument('--save_dir', default='../sidechainnet_scores/PDB', type=str, help='Path to input pdbs -- Can be AA or CG')
parser.add_argument('--CG_noise', default=0.003, type=float, help='Noise profile to use as prior (use training value by default)')
parser.add_argument('--model_path', default='../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt', type=str, help='Path to the model we want to load')
parser.add_argument('--ckp', default=10, type=int, help='Checkpoint for given mode')
parser.add_argument('--n_gens', default=1, type=int, help='N generated samples per structure')
parser.add_argument('--solver', default='euler', type=str, help='Which type of ODE solver to use')
parser.add_argument('--stride', default='1', type=int, help='Stride to apply to frames of large trajectories')
parser.add_argument('--check_clash', action='store_true',  help='Calculate clash for each sample')
parser.add_argument('--check_bonds', action='store_true',  help='Calculate bond quality for each sample')
parser.add_argument('--check_div', action='store_true',  help='Calculate diversity score (for multi-gen)')
parser.add_argument('--mask_prior', action='store_true',  help='Enforce CG positions remain the same')
parser.add_argument('--retain_AA', action='store_true',  help='Keep AA positions from input for validation (takes longer to process)')
parser.add_argument('--system', default='1000-full', type=str,  help='Training set used for the model (sets max atoms)')
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
system = args.system
mask_prior = args.mask_prior
retain_AA = args.retain_AA

save_dir = f'../outputs/{load_dir}'
load_dir = f'../data/{load_dir}'

# add optional preprocessing to save files -- skip if clean dir exists
if retain_AA:
    load_dir = process_pro_aa(load_dir)
else:
    load_dir = process_pro_cg(load_dir)

# save with model prefix + ckp and noise (simplify naming for final model)
save_prefix = f'{save_dir}/{model_path.split("/")[-1]}_ckp-{ckp}_noise-{CG_noise}/'
if mask_prior:
    save_prefix = save_prefix[:-1] + '_masked/'
os.makedirs(save_prefix, exist_ok=True)

# should be loaded in from dict associate with model
model_params = {'depth':6,
                'num_nearest_neighbors':15,
                'dim':32,
                'mdim':32,
                'pos': False}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add these to utils -- for loading DES data
# load DESshaw and SCN test comparisons -- ensure proper sorting

def load_model(model_path, ckp, device, model_params, system):
    '''Load model from a given path to device'''
    
    res_dim = 21
    atom_dim = 37
    
    # load max positional embedding corresponding to specific model
    if system == '1000-full':
        max_atoms = 8070
    elif system == '600all' or system == '600comb':
        max_atoms = 4945 
    elif system == '600max':
        max_atoms = 5000 
    elif system == '200maxaa':
        max_atoms = 1662   
    elif system == '200maxaa':
        max_atoms = 1662

    model = EGNN_Network_time(
        num_tokens = res_dim,
        num_positions = max_atoms,
        dim = model_params['dim'],               
        depth = model_params['depth'],
        num_nearest_neighbors = model_params['num_nearest_neighbors'],
        global_linear_attn_every = 0,
        coor_weights_clamp_value = 2.,  
        m_dim=model_params['mdim'],
        fourier_features = 4, 
        time_dim=0,
        res_dim=res_dim,
        atom_dim=atom_dim,
    ).to(device)

    # load model 
    state_dict_path = f'{model_path}/state-{ckp}.pth' 
    model.load_state_dict(torch.load(state_dict_path))

    return model

def load_features(trj, CG_type='pro-CA'):
    '''Converts trj with a single topology to features
       Can substitue different masks for other CG representations'''
    
    top = trj.top
    n_atoms = trj.n_atoms
    
    # get ohes
    res_ohe, atom_ohe, allatom_ohe = get_pro_ohes(top)
    mask_idxs = top.select('name CA')
    aa_to_cg = get_aa_to_cg(top, mask_idxs)
    xyz = trj.xyz
    
    # convert mask idxs to bool of feature size
    mask = np.ones(len(res_ohe))
    mask[mask_idxs] = 0

    return res_ohe, allatom_ohe, xyz, aa_to_cg, mask, n_atoms, top

def split_list(lst, n):
    """Splits a list into n approximately equal parts"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
 
# for now assume trajs have already been processed
# for preprocessing can use funciton in CG_to_AA and save in temp_dir if needed

model = load_model(model_path, ckp, device, model_params, system)
print(model.eval())

# Track scores -- track each score
# Can only calculate bonds and div if an AA reference is provided 
# Can only calculate div if there are multiple generated structures
bf_list, clash_list, div_list = [], [], []

trj_list = glob.glob(f'{load_dir}/*')
print(trj_list)

for trj_name in tqdm(trj_list, desc='Iterating over structures'):

    trj = md.load(trj_name)[::stride]
    n_frames = trj.n_frames

    # Run cleaning here if cg indicated or if below line fails
    #if clean_trj:
    
    # load features for the given topology
    res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features(trj)

    test_idxs = list(np.arange(n_frames))*n_gens
    xyz_ref = xyz[test_idxs]
    print(f'{trj_name.split("/")[-1]}   {n_frames} frames   {n_atoms} atoms   {n_gens} samples')

    # ensure input will fit in 16GB VRAM 
    n_iters = int(len(test_idxs) * len(res_ohe) / 100_000) + 1  # 50_000 worked but not consistently
    idxs_lists = split_list(test_idxs, n_iters)
    print(n_iters, len(idxs_lists))

    #test_idxs = idxs_lists[0]
          
    xyz_gen = []
    for n, test_idxs in enumerate(idxs_lists):
        n_test = len(test_idxs)
        print(f'iter {n+1} / {n_iters}')

        xyz_test_real = [xyz[i] for i in test_idxs]
        map_test =      [aa_to_cg]*n_test
        mask_test =     [mask]*n_test
        res_test =      [res_ohe]*n_test
        atom_test =     [atom_ohe]*n_test

        # wrap model -- update this so that the function multiplies by the dim of n_gens * n_frames 
        model_wrpd = ModelWrapper(model=model, 
                        feats=torch.tensor(np.array(res_test)).int().to(device), 
                        mask=torch.tensor(np.array(mask_test)).bool().to(device).to(device), 
                        atom_feats=torch.tensor(np.array(atom_test)).to(device))
          
        # apply noise -- only masked values need to be filled here
        if mask_prior:
            xyz_test_prior = get_prior_mix(xyz_test_real, map_test, scale=CG_noise, masks=mask_test)
        else:
            xyz_test_prior = get_prior_mix(xyz_test_real, map_test, scale=CG_noise, masks=None)

        # select solver (adaptive neural sovleer by default)
        if solver == 'adapt':
            node = NeuralODE(model_wrpd, solver="dopri5", sensitivity="adjoint", atol=3e-5, rtol=3e-5) 
            with torch.no_grad():
                ode_traj = node.trajectory(torch.tensor(xyz_test_prior, dtype=torch.float32).to(device), t_span=torch.linspace(0, 1, 2).to(device),)
                ode_traj = ode_traj.cpu().numpy()

        elif solver == 'euler':
            with torch.no_grad():
                ode_traj = euler_integrator(model_wrpd, torch.tensor(xyz_test_prior, dtype=torch.float32).to(device))

        # save trj -- optionally save ODE integration not just last structure
        xyz_gen.append(ode_traj[-1]) 

    # save trj -- optionally save ODE integration not just last structure
    xyz_gen = np.concatenate(xyz_gen)

    # still using original top
    trj_gens = md.Trajectory(xyz_gen, top)
    trj_refs = md.Trajectory(xyz_ref, top)

    # calculate scores
    if check_bonds:
        bf = [bond_fraction(t_ref, t_gen) for t_gen, t_ref in zip(trj_gens, trj_refs)]
        bf = np.array(bf).reshape(n_frames, n_gens)
        bf_list.append(bf)
          
    if check_clash:
        clash = [clash_res_percent(t_gen) for t_gen in trj_gens]
        clash = np.array(clash).reshape(n_frames, n_gens)
        clash_list.append(clash)
          
    # if multiple frames need to rearrange this?
    # we want one div per frame
    if check_div:
        div_frames = []
        for f in range(n_frames):
            print(f, f*n_gens)
            trj_ref_div = trj_refs[f]
            trj_gens_div = trj_gens[f::n_frames]
            div, _ = sample_rmsd_percent(trj_ref_div, trj_gens_div)
            div_frames.append(div)
        div_list.append(div_frames)

    # save gen using same pdb name -- currently saving as n_frames * n_gens
    save_name = f'{save_prefix}{trj_name.split("/")[-1]}'
          
    for i in range(n_gens):
        save_i = save_name.replace('.pdb', f'_{i+1}.pdb')
        trj_gens[i*n_frames:(i+1)*n_frames].save_pdb(save_i)

# save all scores to same dir
if check_bonds:
    np.save(f'{save_prefix}bf.npy', np.array(bf_list))
if check_clash:
    np.save(f'{save_prefix}cls.npy', np.array(clash_list)) 
if check_div:
    np.save(f'{save_prefix}div.npy', np.array(div_list)) 

print(f'\nSaved to:  {save_prefix}\n')