import numpy as np
import mdtraj as md
from itertools import islice, count

from utils import get_dna_ohes, dna_res_to_ohe, dna_atom_to_ohe, dna_allatom_to_ohe

def parse_dna_3spn(dna_trj, with_pro=False):
    '''Extract GNN parameters compatible with 3sn2 CG representation of DNA
       Ensure that dna_trj only includes dna residues
       If proteins also included in then need to add constant to ohes'''
    
    print('init trj shape', dna_trj.xyz.shape)
    
    # seperate AA and CG components if both are included
    try:
        cg_idxs = dna_trj.top.select(f"name DS or name DP or name DB")
        all_idxs = range(dna_trj.n_atoms)
        aa_idxs = [idx for idx in all_idxs if idx not in cg_idxs]

        cg_trj = dna_trj.atom_slice(cg_idxs)
        dna_trj = dna_trj.atom_slice(aa_idxs)
        
    except:
        cg_trj = None
        
    # get all 5' and 3' residues idxs 
    ter_res_list = []
    for chain in dna_trj.topology.chains:
        residues = list(chain.residues)
    
        # Determine the site type for each residue in the chain
        for index, residue in enumerate(residues):
            if index == 0:
                ter_res_list.append(5)
            elif index == len(residues) - 1:
                ter_res_list.append(3)
            else:
                ter_res_list.append(0)
    
    dna_top = dna_trj.top
    n_resid = dna_top.n_residues
    xyz = dna_trj.xyz
    n_frames = len(xyz)
    n_atoms = xyz.shape[1]
    
    xyz_com = [xyz]
    aa_to_cg = np.zeros(n_atoms)
    mask_idxs = []
    cg_atom_list = []
    cg_res_list = []

    for n in range(0, n_resid):

        # make sure to collect O3 from the previous residue
        res_idxs = dna_top.select(f'resid {n} and not name "O3\'"')
        chain_id = dna_top.atom(res_idxs[0]).residue.chain.index
        
        # if not a 5' end then include the O3'
        if ter_res_list[n] != 5:
            O3_prev = dna_top.select(f'resid {n-1} and name "O3\'"')
            res_idxs = np.concatenate([res_idxs, O3_prev])
            
        # if a 3' end then incldue terminal 03' in mapping but not in com
        if ter_res_list[n] == 3:
            O3_curr = dna_top.select(f'resid {n} and name "O3\'"')[0]
        else:
            O3_curr = None
            
        # get names of all atoms in resid
        atom_list = [next(islice(dna_top.atoms, idx, None)).name for idx in res_idxs]

        # get res name
        res_name = next(islice(dna_top.atoms, res_idxs[0], None)).residue.name

        # break if hit a CG coord:
        if 'DS' in atom_list:
            continue

        # passing each res to each chain type
        b_idxs, s_idxs, p_idxs = [], [], []
        b_names, s_names, p_names = [], [], []
        
        for idx, name in zip(res_idxs, atom_list):
            
            # need to get exact lists here and verify against 3spn code -- eg 3 
            if name in ['P', 'OP2', 'OP1', 'O5\'', 'O3\'']:
                p_idxs.append(idx)
                p_names.append(name)
            elif "'" in name: 
                s_idxs.append(idx)
                s_names.append(name)
            else: 
                b_idxs.append(idx)
                b_names.append(name)
                
        # compute center of mass for each
        b_coms = md.compute_center_of_mass(dna_trj.atom_slice(b_idxs)).reshape((n_frames, 1, 3))
        s_coms = md.compute_center_of_mass(dna_trj.atom_slice(s_idxs)).reshape((n_frames, 1, 3))
        
        # append terminal 03' after calculating coms (part of mask but not COM)
        if O3_curr is not None:
            s_idxs.append(O3_curr)

        # check if any phosphates in the residue (don't want to group based on 05' alone)
        if len(p_idxs) > 1:
            p_coms = md.compute_center_of_geometry(dna_trj.atom_slice(p_idxs)).reshape((n_frames, 1, 3))
            xyz_com.append(p_coms)
            xyz_com.append(s_coms)
            xyz_com.append(b_coms)
            
            # check why not getting any b_idxs -- residue has phosphat but no b or s?
            #print('b s p', len(b_idxs), len(s_idxs), len(p_idxs))

            # map to b, s, or p coms
            aa_to_cg[np.array(p_idxs)] = n_atoms + len(xyz_com) - 4
            aa_to_cg[np.array(s_idxs)] = n_atoms + len(xyz_com) - 3
            aa_to_cg[np.array(b_idxs)] = n_atoms + len(xyz_com) - 2

            cg_atom_list += ['PCOM', 'SCOM', 'BCOM']  #['BCOM', 'SCOM', 'PCOM']
            cg_res_list += [res_name]*3

        else:
            # map any missing atoms to the sugar
            xyz_com.append(s_coms)
            xyz_com.append(b_coms)
            
            s_idxs += p_idxs
            aa_to_cg[np.array(s_idxs)] = n_atoms + len(xyz_com) - 3
            aa_to_cg[np.array(b_idxs)] = n_atoms + len(xyz_com) - 2

            cg_atom_list += ['SCOM', 'BCOM'] #['BCOM', 'SCOM']
            cg_res_list += [res_name]*2

    # a lot easier to append everything at the end the xyz
    xyz_com = np.concatenate(xyz_com, axis=1)
    n_atoms_com = xyz_com.shape[1]

    # if cg coords exist, replace the xyz_com values with these
    # need to change order from B-S-P to P-S-B
    if cg_trj is not None:
        print('xyz_com', xyz_com.shape, cg_trj.xyz.shape)
        xyz_com[:, -cg_trj.xyz.shape[1]:] = cg_trj.xyz

    # set mask values to COMs only
    mask_idxs = np.arange(n_atoms, n_atoms_com)

    # set CG mask values to themselves 
    aa_to_cg = np.concatenate([aa_to_cg, np.arange(n_atoms, n_atoms_com)])
    aa_to_cg = np.array(aa_to_cg, dtype=int)
    
    # get res and atom feats for standard atoms
    res_ohe, atom_ohe, all_atom_ohe = get_dna_ohes(dna_top)

    # manually add com encodings for now -- based on pos encoding this might work better dispersed in sequence
    res_ohe = np.concatenate([res_ohe, dna_res_to_ohe(cg_res_list)])
    atom_ohe = np.concatenate([atom_ohe, dna_atom_to_ohe(cg_atom_list)])
    all_atom_ohe = np.concatenate([all_atom_ohe, dna_allatom_to_ohe(cg_atom_list)])
    
    # ensure no overlap with pro encoding
    if with_pro:
        res_ohe = res_ohe + 20
        atom_ohe = atom_ohe + 5
        all_atom_ohe = all_atom_ohe + 36
    
    return xyz_com, mask_idxs, aa_to_cg, res_ohe, all_atom_ohe