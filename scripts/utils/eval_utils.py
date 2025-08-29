#from egnn_utils import *

import sys, os, tempfile
from file_config import FLOWBACK_SRC, FLOWBACK_OUTPUTS
sys.path.append(FLOWBACK_SRC)
from src.utils import *

import glob
import pickle as pkl 

from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP
THREE_TO_ONE_LETTER_MAP = {y: x for x, y in ONE_TO_THREE_LETTER_MAP.items()}

ATOM_MAP_14 = {}
for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
    ATOM_MAP_14[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_14[one_letter].extend(["PAD"] * (14 - len(ATOM_MAP_14[one_letter])))

# CG functions for protein and DNA-prot

def process_pro_aa(load_dir, stride=1):
    '''Re-order and clean all-atom trajectory prior to inference
       Retain xyz position for use as a reference against generated structures
       Should resave pdb + xtc by default'''
    
    save_dir = load_dir + '_clean_AA'
    
    # skip straight to inference if data already cleaned
    if os.path.exists(save_dir):
        print('Data already cleaned')
        return save_dir
        
    os.makedirs(save_dir, exist_ok=True)
    pdb_list = sorted(glob.glob(f'{load_dir}/*pdb'))
    
    print('\nCleaning data -- Only need to do this once')
    print('\nRetaining atoms... this could be slow')
    for pdb in tqdm(pdb_list):
    
        # assume for now aa_trj -- can be CG as welle
        aa_trj = md.load(pdb, stride=stride)
        aa_top = aa_trj.top

        # convert to CG (CA only)
        cg_trj = aa_trj.atom_slice(aa_trj.top.select('name CA'))

        # generate a new all-atom pdb
        aa_pdb = 'MODEL        0\n'
        msk_idxs = []
        idx_list = []
        atom_cnt = 0
        x = y = z = 0.000

        for i, res in enumerate(cg_trj.top.residues):
            one_letter = THREE_TO_ONE_LETTER_MAP[res.name]
            atom_map = ATOM_MAP_14[one_letter]

            for a in atom_map:
                if a=='PAD': break

                try:    

                    idx = aa_top.select(f'resid {i} and name {a}')[0]
                    idx_list.append(idx)

                    #if a=='CA': msk_idxs.append(i)

                    # Format each part of the line according to the PDB specification
                    atom_serial = f"{atom_cnt+1:5d}"
                    atom_name = f"{a:>4}"  # {a:^4} Centered in a 4-character width, adjust as needed
                    residue_name = f"{res.name:3}"
                    chain_id = "A"
                    residue_number = f"{res.index+1:4d}"
                    x_coord = f"{x:8.3f}"
                    y_coord = f"{y:8.3f}"
                    z_coord = f"{z:8.3f}"
                    occupancy = "  1.00"
                    temp_factor = "  0.00"
                    element_symbol = f"{a[:1]:>2}"  # Right-aligned in a 2-character width

                    # Combine all parts into the final PDB line
                    aa_pdb += f"ATOM  {atom_serial} {atom_name} {residue_name} {chain_id}{residue_number}    {x_coord}{y_coord}{z_coord}{occupancy}{temp_factor}           {element_symbol}\n"

                except:
                    print('No matching atom!')

                atom_cnt += 1

        # add TER
        atom_serial = f"{atom_cnt+1:5d}"
        atom_name = f"{' ':^4}"
        residue_name = f"{res.name:3}"
        aa_pdb += f'TER   {atom_serial} {atom_name} {residue_name} {chain_id}{residue_number}\nENDMDL\nEND'

        # if aa traj exists, reorder idxs -- really just need to do this for CAs idxs right?
        xyz = aa_trj.xyz[:, np.array(idx_list)]

        # save txt as temporary pdb and load new molecules
        with tempfile.NamedTemporaryFile('w', suffix='.pdb') as tmp:
            tmp.write(aa_pdb)
            tmp.flush()
            trj_aa_fix = md.load(tmp.name)
        trj_aa_fix = md.Trajectory(xyz, topology=trj_aa_fix.top)

        # save pdb -- save as dcd if longer than
        save_path = pdb.replace(load_dir, save_dir)
        print('save:', save_path)
        trj_aa_fix.save_pdb(save_path)
              
    return save_dir


def process_pro_cg(load_dir, stride=1):
    '''Retain Ca positions only and initializes all other atoms positions to 0,0,0'''
    
    save_dir = load_dir + '_clean'
    
    # skip straight to inference if data already cleaned
    if os.path.exists(save_dir):
        print('Data already cleaned')
        return save_dir
        
    os.makedirs(save_dir, exist_ok=True)
    pdb_list = glob.glob(f'{load_dir}/*pdb')
    
    print('Cleaning data -- Only need to do this once\n\nRetaining Ca positions only')
    for pdb in tqdm(pdb_list):
        
        # search for dcd or xtc corresponding to pdb
        dcd = pdb.replace('.pdb', '.dcd')
        xtc = pdb.replace('.pdb', '.xtc')
        
        if os.path.exists(dcd):
            cg_trj = md.load(dcd, top=pdb, stride=stride)
        elif os.path.exists(xtc):
            cg_trj = md.load(xtc, top=pdb, stride=stride)
        else: 
            cg_trj = md.load(pdb, stride=stride)

        cg_trj = cg_trj.atom_slice(cg_trj.top.select('name CA'))
        cg_xyz = cg_trj.xyz

        # generate a new all-atom pdb
        aa_pdb = 'MODEL        0\n'
        msk_idxs = []
        idx_list = []
        ca_idxs = []
        atom_cnt = 0
        x = y = z = 0.000

        # need to iterate over chians?
        for i, res in enumerate(cg_trj.top.residues):
            one_letter = THREE_TO_ONE_LETTER_MAP[res.name]
            atom_map = ATOM_MAP_14[one_letter]

            for a in atom_map:
                if a=='PAD': break

                try:    
                    # optional if there is a corresponding aa trace
                    if a == 'CA':
                        ca_idxs.append(atom_cnt)

                    # Format each part of the line according to the PDB specification
                    atom_serial = f"{atom_cnt+1:5d}"
                    atom_name = f"{a:>4}"  # {a:^4} Centered in a 4-character width, adjust as needed
                    residue_name = f"{res.name:3}"
                    chain_id = "A"
                    residue_number = f"{res.index+1:4d}"
                    x_coord = f"{x:8.3f}"
                    y_coord = f"{y:8.3f}"
                    z_coord = f"{z:8.3f}"
                    occupancy = "  1.00"
                    temp_factor = "  0.00"
                    element_symbol = f"{a[:1]:>2}"  # Right-aligned in a 2-character width

                    # Combine all parts into the final PDB line
                    aa_pdb += f"ATOM  {atom_serial} {atom_name} {residue_name} {chain_id}{residue_number}    {x_coord}{y_coord}{z_coord}{occupancy}{temp_factor}           {element_symbol}\n"

                except:
                    print('Error in ', pdb, 'no matching atom!')

                atom_cnt += 1

        # add TER
        atom_serial = f"{atom_cnt+1:5d}"
        atom_name = f"{' ':^4}"
        residue_name = f"{res.name:3}"
        aa_pdb += f'TER   {atom_serial} {atom_name} {residue_name} {chain_id}{residue_number}\nENDMDL\nEND'

        # set cg xyz positions
        xyz = np.zeros((len(cg_xyz), atom_cnt, 3))
        xyz[:, np.array(ca_idxs)] = cg_xyz

        # save txt as temporary pdb and load new molecules
        with tempfile.NamedTemporaryFile('w', suffix='.pdb') as tmp:
            tmp.write(aa_pdb)
            tmp.flush()
            trj_aa_fix = md.load(tmp.name)
        trj_aa_fix = md.Trajectory(xyz, topology=trj_aa_fix.top)

        # save pdb
        save_path = pdb.replace(load_dir, save_dir)
        trj_aa_fix.save_pdb(save_path)
              
    return save_dir

def process_dna_cg(pdb, dcd=None, pro_trj=None, save_path=f'{FLOWBACK_OUTPUTS}/sidechainnet_data/test', stride=1, bsp_reorder=False):
    '''Write a blank pdb for the dna coords -- keep the CG section the same as ref'''
    
    standard_order = {
    'DA': ['O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'C1\'', 'N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],       # Adenine nucleotide
    'DT': ['O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'C1\'', 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6'],              # Thymine nucleotide
    'DC': ['O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'C1\'', 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],                    # Cytosine nucleotide
    'DG': ['O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'C1\'', 'N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],  # Guanine nucleotide
    }
    phos_seq = ['P', 'OP1', 'OP2']
    
    # convert to CG (CA only)
    if dcd is not None: cg_trj = md.load(dcd, top=pdb, stride=stride)
    else: cg_trj = md.load(pdb, stride=stride)
    n_frames = cg_trj.n_frames
    
    dna_idxs = cg_trj.topology.select("resname DG or resname DA or resname DT or resname DC")
    cg_trj = cg_trj.atom_slice(dna_idxs)
        
    # extract DNA sequence and keep track of 5', 3' termini
    res_list = []
    natoms_list = []
    for res in cg_trj.top.residues:
        res_list.append(res.name)
        natoms_list.append(len([a for a in res.atoms]))
            
    # specify if ter or internal
    ter_list = np.zeros(len(natoms_list))
    five_idxs = np.where(np.array(natoms_list)==2)[0]
    ter_list[five_idxs] = 5
    ter_list[five_idxs[1:] - 1] = 3
    ter_list[-1] = 3
            
    # generate a new all-atom pdb
    aa_pdb = 'MODEL        0\n'
    msk_idxs = []
    idx_list = []
    ca_idxs = []
    atom_cnt = 0
    x = y = z = 0.000
    
    # iterate over residue and terminal status
    for i, (res, ter) in enumerate(zip(res_list, ter_list)):
        
        # get standard order for each residue type
        atom_map = standard_order[res]
        
        # add phosphate unless its a 5' terminal
        if ter != 5:
            atom_map = phos_seq + atom_map

        for a in atom_map:

            try:    
                # Format each part of the line according to the PDB specification
                atom_serial = f"{atom_cnt+1:5d}"
                atom_name = f"{a:>4}"  # {a:^4} Centered in a 4-character width, adjust as needed
                residue_name = f"{res:3}"
                chain_id = "A"
                residue_number = f"{i+1:4d}"
                x_coord = f"{x:8.3f}"
                y_coord = f"{y:8.3f}"
                z_coord = f"{z:8.3f}"
                occupancy = "  1.00"
                temp_factor = "  0.00"
                element_symbol = f"{a[:1]:>2}"  # Right-aligned in a 2-character width

                # Combine all parts into the final PDB line
                aa_pdb += f"ATOM  {atom_serial} {atom_name} {residue_name} {chain_id}{residue_number}    {x_coord}{y_coord}{z_coord}{occupancy}{temp_factor}           {element_symbol}\n"

            except:
                print('No matching atom!')

            atom_cnt += 1
            
        if ter==3:

            # add TER
            atom_serial = f"{atom_cnt+1:5d}"
            atom_name = f"{' ':^4}"
            residue_name = f"{res:3}"
            aa_pdb += f'TER   {atom_serial} {atom_name} {residue_name} {chain_id}{residue_number}\n'
            
    aa_pdb += '\nENDMDL\nEND'
    
    # save txt as temporary pdb and load new molecules
    with tempfile.NamedTemporaryFile('w', suffix='.pdb') as tmp:
        tmp.write(aa_pdb)
        tmp.flush()
        # matching aa_traj with original number of frames
        aa_trj = md.load(tmp.name)
    aa_xyz = np.zeros((n_frames, aa_trj.n_atoms, 3))
    aa_trj= md.Trajectory(aa_xyz, topology=aa_trj.top)
    print(n_frames, aa_trj.xyz.shape, cg_trj.xyz.shape)
    
    # can drop this now?
    # depending on parser may need to re-order cg_trj as B-S-P to P-S-B (or CG ordering is different from AICG+3spn2)
    if bsp_reorder:
        cg_idxs = []
        cg_cnt = 0
        for i, ter in enumerate(ter_list):
            if ter == 5:
                cg_idxs += [cg_cnt+1, cg_cnt]
                cg_cnt += 2
            else:
                cg_idxs += [cg_cnt+2, cg_cnt+1, cg_cnt]
                cg_cnt += 3

        cg_xyz = cg_trj.xyz[:, np.array(cg_idxs)]
    else:
        print('No re-order')
        cg_xyz = cg_trj.xyz

    # concatenate aa trace with cg trace
    if pro_trj is not None:
        pro_top, aa_top, cg_top = pro_trj.top, aa_trj.top, cg_trj.top
        comb_top = pro_top.join(aa_top)
        comb_top = comb_top.join(cg_top)
        comb_xyz = np.concatenate((pro_trj.xyz, aa_xyz,cg_xyz), axis=1)
        comb_trj = md.Trajectory(xyz=comb_xyz, topology=comb_top)   
    else:
        aa_top, cg_top = aa_trj.top, cg_trj.top
        comb_top = aa_top.join(cg_top)
        comb_xyz = np.concatenate((aa_xyz, cg_xyz), axis=1)
        comb_trj = md.Trajectory(xyz=comb_xyz, topology=comb_top)
        
    # save pdb
    if save_path is not None:
        comb_trj.save_pdb(save_path)
    
    return aa_trj

#@ optionally add these functions to eval_utils
def load_model(model_path, ckp, device): #, sym='e3', pos_cos=None, seq_feats=0, seq_decay=100):
    '''Load model from a given path to device'''
    
    # load hyperparams associated with specific model
    model_params = pkl.load(open(f'{model_path}/params.pkl', 'rb')) 

    ## TODO -- get rid or these inputs and always set pos_cos, num_positions to None
    pos_cos = model_params['pos_cos']

    # don't use normal pos encoding if using sin/cos
    if pos_cos > 0.001:
        num_positions = None
    else: 
        pos_cos = None
        num_positions = None  # change this back ifthe 
        #num_positions = model_params['max_atoms']

    model = EGNN_Network_time(
        num_tokens = model_params['res_dim'],
        atom_dim = model_params['atom_dim'],
        num_positions = num_positions, #model_params['max_atoms'],
        dim = model_params['dim'],               
        depth = model_params['depth'],
        num_nearest_neighbors = model_params['num_nearest_neighbors'],
        global_linear_attn_every = 0,
        coor_weights_clamp_value = 2.,  
        m_dim=model_params['mdim'],
        fourier_features = 4, 
        time_dim=0,
        res_dim=model_params['res_dim'],
        sym=model_params['sym'],
        emb_cos_scale=pos_cos,
        seq_feats=model_params['seq_feats'],
        seq_decay=model_params['seq_decay'],
        act=model_params['act'],
    ).to(device)

    # load model 
    state_dict_path = f'{model_path}/state-{ckp}.pth' 
    model.load_state_dict(torch.load(state_dict_path))

    return model

def load_features_pro(trj, CG_type='pro-CA'):
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

# code taken from DNA_new_dataset
def load_features_DNApro(trj, CG_type='pro-CA'):
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

    # define seperate and combined trajs
    dna_traj, pro_traj = traj_full.atom_slice(dna_idxs), traj_full.atom_slice(pro_idxs)
    dna_pro_traj = traj_full.atom_slice(sorted(list(pro_idxs) + list(dna_idxs)))
    top = dna_pro_traj.top
    
    # parse protein graph
    xyz_p, mask_idxs_p, aa_to_cg_p, res_ohe_p, atom_ohe_p = parse_pro_CA(pro_traj)

    # parse dna graph
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
    n_atoms = len(res_ohe)

    # convert mask idxs to bool of feature size
    mask = np.ones(len(res_ohe))
    mask[mask_idxs] = 0

    # check we shouldn't be returns allatom_ohe here
    return res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top

def split_list(lst, n):
    """Splits a list into n approximately equal parts"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]