### input directory to pdbs/trajs and return N generated samples of each ###

import sys, os
sys.path.append('../')

from utils import *   # only import essentias here to reduce time
import argparse
import glob
import pickle as pkl

from tqdm import tqdm
import time

from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP
THREE_TO_ONE_LETTER_MAP = {y: x for x, y in ONE_TO_THREE_LETTER_MAP.items()}

ATOM_MAP_14 = {}
for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
    ATOM_MAP_14[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_14[one_letter].extend(["PAD"] * (14 - len(ATOM_MAP_14[one_letter])))

parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', default='../sidechainnet_data/DNAPro', type=str, help='Path to input pdbs -- Can be AA or CG')
parser.add_argument('--save_dir', default='../sidechainnet_scores/DNAPro', type=str, help='Path to input pdbs -- Can be AA or CG')
parser.add_argument('--CG_noise', default=0.003, type=float, help='Noise profile to use as prior (use training value by default)')
parser.add_argument('--model_path', default='../jobs/time-batch_adamW_dna-pro-no-rev-100-500-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.01_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt', type=str, help='Path to the model we want to load')
parser.add_argument('--ckp', default=500, type=int, help='Checkpoint for given mode')
parser.add_argument('--n_gens', default=1, type=int, help='N generated samples per structure')
parser.add_argument('--solver', default='euler', type=str, help='Which type of ODE solver to use')
parser.add_argument('--stride', default='1', type=int, help='Stride to apply to frames of large trajectories')
parser.add_argument('--check_clash', action='store_true',  help='Calculate clash for each sample')
parser.add_argument('--check_bonds', action='store_true',  help='Calculate bond quality for each sample')
parser.add_argument('--check_div', action='store_true',  help='Calculate diversity score (for multi-gen)')
parser.add_argument('--mask_prior', action='store_true',  help='Ensure exact match to CG')
parser.add_argument('--system', default='dna-pro-no-rev-100-500', type=str, help='Dataset this model was trained on -- only need if loading directly from test set')
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
system = args.system

# use updated parsing function if P035 is in system name -- set this to default once confirmed working
if 'P035' in system or 'clean' in system :
    from utils_P035 import parse_dna_3spn
    print('loaded P035 parser')
else:
    print('Using default parser')
    
# add preprocessing and new save functions

# save with model prefix -- leave off first n chacracters to save space
save_prefix = f'{args.save_dir}/{model_path.split("/")[-1][15:]}_ckp-{ckp}_noise-{CG_noise}/'

# update name is using mask prior
if mask_prior:
    save_prefix = save_prefix[:-1] + '_masked/'
    
os.makedirs(save_prefix, exist_ok=True)

# hyperparameters associateed with model training
model_params = {'depth':6,
                'num_nearest_neighbors':15,
                'dim':32,
                'mdim':32,
                'pos': False,
                'res_dim': 25,
                'atom_dim': 68}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add these to utils -- for loading DES data
# load DESshaw and SCN test comparisons -- ensure proper sorting

def load_model(model_path, ckp, device, model_params, system):
    '''Load model from a given path to device'''
    
    # define max positional embedding based on specific training
    if system == 'dna-pro-no-rev-100-500':
        max_atoms = 6107
    elif system == 'dna-range-rev-reformat-P035_100-500':
        max_atoms = 6110
    elif system == 'dna-range-rev-reformat-P035-remap-psb_100-500':
        max_atoms = 6107
    elif system == 'dna-clean-fix-mis-0_120-500':
        max_atoms = 6299

    model = EGNN_Network_time(
        num_tokens = model_params['res_dim'],
        num_positions = max_atoms,
        dim = model_params['dim'],               
        depth = model_params['depth'],
        num_nearest_neighbors = model_params['num_nearest_neighbors'],
        global_linear_attn_every = 0,
        coor_weights_clamp_value = 2.,  
        m_dim=model_params['mdim'],
        fourier_features = 4, 
        time_dim=0,
        res_dim = model_params['res_dim'],
        atom_dim= model_params['atom_dim'],
    ).to(device)

    # load model 
    state_dict_path = f'{model_path}/state-{ckp}.pth' 
    model.load_state_dict(torch.load(state_dict_path))

    return model

# code taken from DNA_new_dataset
def load_features(trj, CG_type='pro-CA'):
    '''Converts trj with a single topology to features
       Can substitue different masks for other CG representations'''
    
    amino_acids_three_letter = ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
    aa_set = set([acid.upper() for acid in amino_acids_three_letter])
    nuc_set = set(['DT', 'DA', 'DC', 'DG'])
    
    heavy_idxs = trj.top.select("mass > 1.1 and not name OXT") 
    traj_full = trj.atom_slice(heavy_idxs) #[0] 

    top = traj_full.top
    nuc_types = np.array([a.residue.name in nuc_set for a in top.atoms])
    pro_types = np.array([a.residue.name in aa_set for a in top.atoms])
    dna_idxs, pro_idxs = np.where(nuc_types)[0], np.where(pro_types)[0]

    # define seperate and combined trajsthet
    dna_traj, pro_traj = traj_full.atom_slice(dna_idxs), traj_full.atom_slice(pro_idxs)
    dna_pro_traj = traj_full.atom_slice(sorted(list(pro_idxs) + list(dna_idxs)))
    top = dna_pro_traj.top
    
    # parse protein graph
    xyz_p, mask_idxs_p, aa_to_cg_p, res_ohe_p, atom_ohe_p = parse_pro_CA(pro_traj)

    # get dna feats -- add residue values to ohes
    xyz_d, mask_idxs_d, aa_to_cg_d, res_ohe_d, atom_ohe_d = parse_dna_3spn(dna_traj, with_pro=True)

    # append protein residues to map and map h
    n_masks = xyz_p.shape[1]
    mask_idxs_d = mask_idxs_d + n_masks
    aa_to_cg_d = aa_to_cg_d + n_masks

    # combine pro and dna feats (pro always comes first)
    xyz = np.concatenate([xyz_p, xyz_d], axis=1)
    mask_idxs = np.concatenate([mask_idxs_p, mask_idxs_d])
    aa_to_cg = np.concatenate([aa_to_cg_p, aa_to_cg_d])
    res_ohe = np.concatenate([res_ohe_p, res_ohe_d])
    atom_ohe = np.concatenate([atom_ohe_p, atom_ohe_d])

    # convert mask idxs to bool of feature size
    mask = np.ones(len(res_ohe))
    mask[mask_idxs] = 0

    return res_ohe, atom_ohe, xyz, aa_to_cg, mask, top

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

    # just looks at first frame for now (need to fix how parseing ensemble xyz)
    trj = md.load(trj_name)  #[0]
    n_frames = trj.n_frames

    # Can do a check here to make sure traj is correctly formatted
    #if clean_trj:
    
    # load features for the given topology
    res_ohe, atom_ohe, xyz, aa_to_cg, mask, top = load_features(trj)

    test_idxs = list(np.arange(n_frames))*n_gens
    xyz_ref = xyz[test_idxs]
    print(f'{trj_name.split("/")[-1]}   {n_frames} frames   {top.n_atoms} atoms   {n_gens} samples')
          
    # ensure input will fit in 16GB VRAM 
    n_iters = int(len(test_idxs) * len(res_ohe) / 50_000) + 1 # 50_000 worked but not consistently
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
                ode_traj = node.trajectory(torch.tensor(xyz_test_prior, dtype=torch.float32).to(device), 
                                           t_span=torch.linspace(0, 1, 2).to(device),)
                ode_traj = ode_traj.cpu().numpy()

        elif solver == 'euler':
            with torch.no_grad():
                ode_traj = euler_integrator(model_wrpd, torch.tensor(xyz_test_prior, dtype=torch.float32).to(device))

        # save trj -- optionally save ODE integration not just last structure
        xyz_gen.append(ode_traj[-1]) 
          
        # save one frame traj that includes the integration over time -- should turn this off to save time
        if n == 0:
            save_name_dt = f'{save_prefix}{trj_name.split("/")[-1]}_dt.pdb'
            xyz_gen_dt = ode_traj[:, -1]
            aa_idxs = top.select(f"not name DS and not name DP and not name DB")
            trj_gen_dt = md.Trajectory(xyz_gen_dt[:, :top.n_atoms], top).atom_slice(aa_idxs)
            trj_gen_dt.save_pdb(save_name_dt)
          
    xyz_gen = np.concatenate(xyz_gen)
    print(xyz_gen.shape, xyz_ref.shape)

    # don't include virtual atoms in top 
    aa_idxs = top.select(f"not name DS and not name DP and not name DB")
    trj_gens = md.Trajectory(xyz_gen[:, :top.n_atoms], top).atom_slice(aa_idxs)
    trj_refs = md.Trajectory(xyz_ref[:, :top.n_atoms], top).atom_slice(aa_idxs)

    # calculate scores
    if check_bonds:
        bf = [bond_fraction(t_ref, t_gen) for t_gen, t_ref in zip(trj_gens, trj_refs)]
        bf = np.array(bf).reshape(n_frames, n_gens)
        bf_list.append(bf)

    # protein only for now -- add check on    
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
    trj_gens.save_pdb(save_name)
         

# save all scores to same dir
if check_bonds:
    np.save(f'{save_prefix}bf.npy', np.array(bf_list))
if check_clash:
    np.save(f'{save_prefix}cls.npy', np.array(clash_list)) 
if check_div:
    np.save(f'{save_prefix}div.npy', np.array(div_list)) 

print(f'\nSaved to:  {save_prefix}\n')