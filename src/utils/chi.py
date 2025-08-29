import numpy as np
import torch 
import mdtraj as md
from scipy.spatial.transform import Rotation as R

def get_all_chiralities_vec(traj):
    chiralities = []

    for frame_idx in range(traj.n_frames):
        frame_chiralities = []
        for residue in traj.top.residues:
            try:
                # Get atom indices for N, CA, C, and CB
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index

                # Get the coordinates of the atoms for the current frame
                n = traj.xyz[frame_idx, n_idx]
                ca = traj.xyz[frame_idx, ca_idx]
                c = traj.xyz[frame_idx, c_idx]
                cb = traj.xyz[frame_idx, cb_idx]

                # Calculate the vectors
                v1 = n - ca
                v2 = c - ca
                v3 = cb - ca

                # Compute the volume of the parallelepiped formed by v1, v2, and v3
                volume = np.dot(np.cross(v1, v2), v3)

                if volume > 0.:
                    frame_chiralities.append(-1)
                elif volume < 0.:
                    frame_chiralities.append(1)
                else:
                    frame_chiralities.append(0)

            except KeyError:
                frame_chiralities.append(0)

        chiralities.append(frame_chiralities)

    return np.array(chiralities)

def get_atom_indices_by_name(topology, residue, atom_names):
    indices = []
    for atom_name in atom_names:
        try:
            indices.append([atom.index for atom in residue.atoms if atom.name == atom_name][0])
        except IndexError:
            indices.append(None)  # Append None if the atom is not found (e.g., CB in glycine)
    return indices

def get_dihed_idxs(top):

    # List to hold the atom indices for each residue
    atom_indices = []

    # Get the indices of N, CA, CB, and C atoms for each residue
    for residue in top.residues:
        if residue.name != 'GLY':  # Skip glycine residues
            indices = get_atom_indices_by_name(top, residue, ['N', 'CA', 'CB', 'C'])
            if None not in indices:  # Ensure all atoms are present
                atom_indices.append(indices)

    print("Atom indices for each residue (excluding glycine):")
    for idx_set in atom_indices:
        print(idx_set)
        
    return atom_indices


def invert_chirality(traj, chi):
    """
    Invert the chirality of specific residues.

    Parameters:
    - traj: MDTraj trajectory object
    - res_list: List of residue indices to invert chirality
    """
    
    for frame_index in range(traj.n_frames):
        res_list = np.where(chi[frame_index] > 0.001)[0]
    
        for residue_index in res_list:

            # Get the specific residue
            residue = traj.topology.residue(residue_index)

            # Identify the atoms in the chiral center (N, CA, C, CB)
            try:
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index
            except KeyError:
                print(f"Residue {residue_index} does not have the required atoms for chirality inversion.")
                continue

            # Get the indices of all side chain atoms
            side_chain_indices = [atom.index for atom in residue.atoms if atom.name not in ['N', 'CA', 'C', 'O', 'OXT']]

            # Include the CB atom in the side chain if it's not already included
            if cb_idx not in side_chain_indices:
                side_chain_indices.append(cb_idx)
        
            # Get the coordinates of the chiral center atoms for the first frame
            n = traj.xyz[frame_index, n_idx]
            ca = traj.xyz[frame_index, ca_idx]
            c = traj.xyz[frame_index, c_idx]
            cb = traj.xyz[frame_index, cb_idx]

            # Calculate the centroid of the chiral center (excluding CB for the rotation axis calculation)
            centroid = (n + ca + c) / 3.0
        
            # Calculate the rotation axis (from CA to the centroid of N, CA, and C)
            rotation_axis = np.cross(ca - centroid, c - centroid)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize the axis

            # Calculate the 180-degree rotation matrix
            # MJ add negative to rotation
            rot_matrix = -R.from_rotvec(np.pi * rotation_axis).as_matrix()

            # Apply the rotation to each side chain atom
            for atom_idx in side_chain_indices:
                atom_coords = traj.xyz[frame_index, atom_idx]  # Get coordinates for all frames
                # Translate to the origin (centroid)
                translated_coords = atom_coords - centroid
                # Rotate the coordinates
                rotated_coords = np.dot(rot_matrix, translated_coords.T).T
                # Translate back to the original position
                new_coords = rotated_coords + centroid
                # Update the coordinates in the trajectory
                traj.xyz[frame_index, atom_idx] = new_coords

    return traj


def invert_chirality_reflection(traj, chi):
    """
    Invert the chirality of specific residues.
    Flips everything across the pplan formed by n-ca and c-ca 
    Seems to increase clash but preserve bond, not sure if different from rotation
    """
    
    for frame_index in range(traj.n_frames):
        res_list = np.where(chi[frame_index] > 0.001)[0]
    
        for residue_index in res_list:

            # Get the specific residue
            residue = traj.topology.residue(residue_index)

            # Identify the atoms in the chiral center (N, CA, C, CB)
            try:
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index
            except KeyError:
                print(f"Residue {residue_index} does not have the required atoms for chirality inversion.")
                continue

            # Get the coordinates of the chiral center atoms
            n = traj.xyz[frame_index, n_idx]
            ca = traj.xyz[frame_index, ca_idx]
            c = traj.xyz[frame_index, c_idx]
            cb = traj.xyz[frame_index, cb_idx]

            # Calculate the normal vector to the plane defined by N, CA, and C
            normal_vector = np.cross(n - ca, c - ca)
            normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector

            # Get the indices of all side chain atoms
            side_chain_indices = [atom.index for atom in residue.atoms if atom.name not in ['N', 'CA', 'C', 'O', 'OXT']]

            # Reflect the side chain atoms through the plane defined by N, CA, and C
            for atom_idx in side_chain_indices:
                atom_coords = traj.xyz[frame_index, atom_idx]
                reflected_coords = atom_coords - 2 * np.dot(atom_coords - ca, normal_vector) * normal_vector
                traj.xyz[frame_index, atom_idx] = reflected_coords

    return traj


def invert_chirality_reflection_ter(traj, chi):
    """
    Invert the chirality of specific residues.
    Flips everything across the plane formed by N-CA and C-CA.
    """
    
    for frame_index in range(traj.n_frames):
        res_list = np.where(chi[frame_index] > 0.001)[0]
    
        for residue_index in res_list:

            # Get the specific residue
            residue = traj.topology.residue(residue_index)

            # Identify the atoms in the chiral center (N, CA, C, CB)
            try:
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index
                o_idx = residue.atom('O').index
            except KeyError:
                print(f"Residue {residue_index} does not have the required atoms for chirality inversion.")
                continue

            # Check if the residue is at the N-terminus or C-terminus
            is_n_terminus = False
            is_c_terminus = False
            
            if residue_index == 0:
                is_n_terminus = True
            elif residue_index == traj.n_residues-1:
                is_c_terminus = True
            elif residue.chain.index != traj.topology.residue(residue_index-1).chain.index:
                is_n_terminus = True
            elif residue.chain.index != traj.topology.residue(residue_index+1).chain.index:
                is_c_terminus = True

            # Get the coordinates of the chiral center atoms
            n = traj.xyz[frame_index, n_idx]
            ca = traj.xyz[frame_index, ca_idx]
            c = traj.xyz[frame_index, c_idx]
            cb = traj.xyz[frame_index, cb_idx]
            o = traj.xyz[frame_index, o_idx]

            if is_n_terminus:
                normal_vector = np.cross(c - ca, cb - ca)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
                
                # Reflect the N atom through the plane
                n_coords = traj.xyz[frame_index, n_idx]
                reflected_coords = n_coords - 2 * np.dot(n_coords - ca, normal_vector) * normal_vector
                traj.xyz[frame_index, n_idx] = reflected_coords

            elif is_c_terminus:
                normal_vector = np.cross(n - ca, cb - ca)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
                
                # Reflect the C and O atoms through the plane
                co_indices = [atom.index for atom in residue.atoms if atom.name in ['C', 'O', 'OXT']]
                for co_idx in co_indices:
                    co_coords = traj.xyz[frame_index, co_idx]
                    reflected_coords = co_coords - 2 * np.dot(co_coords - ca, normal_vector) * normal_vector
                    traj.xyz[frame_index, co_idx] = reflected_coords

            else:
                # Calculate the normal vector to the plane defined by N, CA, and C
                normal_vector = np.cross(n - ca, c - ca)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
                
                # Get the indices of all side chain atoms
                side_chain_indices = [atom.index for atom in residue.atoms if atom.name not in ['N', 'CA', 'C', 'O', 'OXT']]

                # Reflect the side chain atoms through the plane defined by N, CA, and C
                for atom_idx in side_chain_indices:
                    atom_coords = traj.xyz[frame_index, atom_idx]
                    reflected_coords = atom_coords - 2 * np.dot(atom_coords - ca, normal_vector) * normal_vector
                    traj.xyz[frame_index, atom_idx] = reflected_coords

    return traj


def euler_integrator_chi_check(model, x, nsteps=100, x0_diff=False, t_flip=1.01, top_ref=None, keep_flip=False, type_flip='ref-ter'):
    
    # try adding small amounts of noise during integration
    
    ode_list = []
    dt = 1./(nsteps-1)
    x0 = x.detach()
    n_real = top_ref.n_atoms
    device = x.device
    
    chi_list = []
    chiral_flipped = False
    
    # select the flipping function
    if keep_flip or not chiral_flipped:
        if type_flip=='rot':
            flip_func = invert_chirality
        elif  type_flip=='ref':
            flip_func = invert_chirality_reflection
        elif  type_flip=='ref-CB':
            flip_func = invert_chirality_CB_only
        elif  type_flip=='ref-ter':
            flip_func = invert_chirality_reflection_ter
    
    for t in np.linspace(0, 1, nsteps):
        
        # Evaluate dx/dt using the model for the current state and time
        with torch.no_grad():
            if x0_diff:
                dx_dt = model(t, x.detach(), x0=x0)
            else:
                dx_dt = model(t, x.detach())

        # Compute the next state using Euler's
        x = (x + dx_dt * dt).detach() 
        
        # Along the way check for enantiomers and try flip them
        # first frame only for now
        if t > t_flip:
            if keep_flip or not chiral_flipped:
                
                # add this to the t_flip loop to save time (especially if don't keep flipping)
                x_real = x[:, :n_real]
                trj = md.Trajectory(x_real.cpu().numpy(), top_ref)
                chi = get_all_chiralities_vec(trj)
                chi_list.append(chi)
                trj = flip_func(trj, chi)
                x[:, :n_real] = torch.Tensor(trj.xyz).to(device)
                chiral_flipped = True
        
        ode_list.append(x.cpu().numpy())
        
    # final flip to ensure no enantiomiers are left
    x_real = x[:, :n_real]
    trj = md.Trajectory(x_real.cpu().numpy(), top_ref)
    chi = get_all_chiralities_vec(trj)
    res_list = np.where(chi[0] > 0.001)[0]
    trj = flip_func(trj, chi)
    x[:, :n_real] = torch.Tensor(trj.xyz).to(device)
    
    ode_list.append(x.cpu().numpy())
    chi_list.append(get_all_chiralities_vec(trj))

    return np.array(ode_list), chi_list

