### input directory to pdbs/trajs and return N generated samples of each ###
import re
import os
from src.file_config import FLOWBACK_JOBDIR, FLOWBACK_DATA, FLOWBACK_OUTPUTS, FLOWBACK_MODELS
from collections import defaultdict
import argparse
import glob
import pickle as pkl
from tqdm import tqdm

# Function to extract state and step numbers
def extract_numbers(filename):
    state_match = re.search(r'state-(\d+)', filename)
    step_match = re.search(r'step-(\d+)', filename)
    
    state_num = int(state_match.group(1)) if state_match else float('inf')
    step_num = int(step_match.group(1)) if step_match else float('inf')  # If no step, set to infinity
    
    return (state_num, step_num)
    
# need to test these for preproccessing
from src.utils.evaluation import *
from src.utils.energy import charmm_structure_to_energy

# import functions to check and correct chirality
from src.utils.chi import *

parser = argparse.ArgumentParser()
parser.add_argument('--CG_noise', default=0.003, type=float, help='Noise profile to use as prior')
parser.add_argument('--n_gens', default=2, type=int, help='N generated samples per structure')
parser.add_argument('--solver', default='euler_ff', type=str, help='Which type of ODE solver to use')
parser.add_argument('--model_path', default=f'{FLOWBACK_MODELS}/Pro_pretrained', type=str, help='Trained model')
parser.add_argument('--nsteps', default=100, type=int, help='Number of steps in Euler integrator')
parser.add_argument('--vram', default=32, type=int, help='Scale batch size to fit max gpu VRAM')
parser.add_argument('--save_traj', action='store_true',  help='Save all flow-matching timesteps')
parser.add_argument('--save_dcd', action='store_true',  help='Save traj output as a pdb+dcd')

# for enantiomer correction
parser.add_argument('--t_flip', default=0.2, type=float,  help='ODE time to correct fo D-residues')
parser.add_argument('--type_flip', type=str, default='ref-ter', help='Method used to fix chirality')

args = parser.parse_args()
CG_noise = args.CG_noise
model_path = args.model_path
n_gens = args.n_gens
solver = args.solver

nsteps = args.nsteps
t_flip = args.t_flip
type_flip = args.type_flip
vram = args.vram



models = glob.glob(f'{FLOWBACK_JOBDIR}/{model_path}_post/state*step*.pth')
sorted_models = sorted(models, key=extract_numbers)
proteins = ['chignolin', '2JOF', 'GTT', 'PRB']
trj_dict = defaultdict(list)
for protein in proteins:
    for frame in range(0, 2000, 200):
        trj_dict[protein].append(f'{FLOWBACK_DATA}/{protein}_clean_AA/frame_{frame}.pdb')

energies = np.zeros((len(sorted_models), 4))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rtp_data, lj_data, bond_data = get_ff_data()


# load model
for m_i, mpath in enumerate(sorted_models):
    state, step = extract_numbers(mpath)
    if np.isinf(step):
        model = load_model(f'{FLOWBACK_JOBDIR}/{model_path}_post', state, device) #, sym, pos_cos, seq_feats, seq_decay)
    else:
        model = load_model(f'{FLOWBACK_JOBDIR}/{model_path}_post', f'{state}-step-{step}', device) #, sym, pos_cos, seq_feats, seq_decay)
    for p_i, protein in enumerate(proteins):
        protein_energies = []
        for trj_name in tqdm(trj_dict[protein], desc='Iterating over trajs'):
        
            trj = md.load(trj_name)
            n_frames = trj.n_frames
        
            res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_pro(trj)
        
        
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
                n_test = len(test_idxs)
                print(f'iter {n+1} / {n_iters}')
        
                xyz_test_real = [xyz[i] for i in test_idxs]
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
                    if solver == 'euler':
                        ode_traj = euler_integrator(model_wrpd, 
                                          torch.tensor(prior, dtype=torch.float32).to(device), 
                                              nsteps=nsteps)
                    elif solver == 'euler_ff':
                        ode_traj = euler_ff_integrator(model_wrpd, torch.tensor(prior,
                                                                        dtype=torch.float32).to(device), torch.tensor(ca_pos_test).to(device), rtp_data, lj_data, bond_data, top, device)
               
        
        
                # save trj -- optionally save ODE integration not just last structure -- only for one gen
                xyz_gen.append(ode_traj[-1] + ca_pos_test) 
        
            xyz_gen = np.concatenate(xyz_gen)
                  
            # don't include DNA virtual atoms in top 
            aa_idxs = top.select(f"not name DS and not name DP and not name DB")
            for i in range(xyz_gen.shape[0]):
                e, g = charmm_structure_to_energy(top, xyz_gen[i, :top.n_atoms])
                protein_energies.append(e)
        energies[m_i, p_i] = np.mean(protein_energies)
    

np.save(f'{FLOWBACK_OUTPUTS}/validation_{model_path}.npy', energies)
