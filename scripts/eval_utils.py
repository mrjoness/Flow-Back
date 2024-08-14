import glob
import mdtraj as md
import os
import numpy as np
import tqdm

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
    pdb_list = glob.glob(f'{load_dir}/*pdb')
    
    print('\nCleaning data -- Only need to do this once')
    print('\nRetaining atoms... this could be slow')
    for pdb in tqdm.tqdm(pdb_list):
    
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
        open('.temp.pdb', 'w').write(aa_pdb)
        trj_aa_fix = md.load('.temp.pdb')
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
    for pdb in tqdm.tqdm(pdb_list):
        
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
        open('.temp.pdb', 'w').write(aa_pdb)
        trj_aa_fix = md.load('.temp.pdb')
        trj_aa_fix = md.Trajectory(xyz, topology=trj_aa_fix.top)

        # save pdb
        save_path = pdb.replace(load_dir, save_dir)
        trj_aa_fix.save_pdb(save_path)
              
    return save_dir