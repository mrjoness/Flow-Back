from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import mdtraj as md
import torch
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger
from rdkit import rdBase
from openmm import *
import os
from openmm.app import *
from openmm.unit import *
import argparse
import tempfile
import MDAnalysis as mda
import subprocess
import warnings
import gc
from collections import defaultdict
import io
import time
from pathlib import Path
import shutil
import re
from src.file_config import FLOWBACK_BASE, FLOWBACK_FF
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

def _osremove(f):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

def compute_all_distances(traj):
    idxs = np.arange(traj.top.n_atoms)
    grid = np.array(np.meshgrid(idxs, idxs)).T.reshape(-1, 2)
    pairs = grid[grid[:, 0] > grid[:, 1]]
    dists = md.compute_distances(traj, pairs)
    return dists




_TOPLEVEL_SUBSECTIONS = {
    "replace", "add", "delete"
}

_BLOCK_HEADER_RE = re.compile(r'^\[\s*([^\]]+?)\s*\]\s*$', re.MULTILINE)

def _is_subsection(name: str) -> bool:
    # Case-insensitive match for subsection keywords
    return name.strip().lower() in _TOPLEVEL_SUBSECTIONS

def _find_first_existing(files: List[Path], preferred_names: Tuple[str, ...]) -> List[Path]:
    """Return files in priority order: any explicitly preferred names first (if present),
    else all matching files as given."""
    preferred = []
    others = []
    for f in files:
        if f.name in preferred_names:
            preferred.append(f)
        else:
            others.append(f)
    # Try preferred ones first, but still consider others for missed entries
    return preferred + others

def _gather_ff_tdb_files(ff_dir: Path, suffix: str) -> List[Path]:
    """Gather *.n.tdb or *.c.tdb files from a force field directory."""
    return sorted(ff_dir.glob(f"*{suffix}.tdb"))

def _extract_named_blocks_from_text(text: str, wanted_names: List[str]) -> Dict[str, str]:
    """
    Extract blocks by name from a .tdb file's text.
    A block starts at a top-level header line: [ NAME ] where NAME is not a lowercase subsection.
    The block ends just before the next top-level block header.
    """
    results: Dict[str, str] = {}
    # Find all bracket headers with their positions
    matches = list(_BLOCK_HEADER_RE.finditer(text))
    # Precompute header names and which are top-level vs subsection
    headers = []
    for m in matches:
        name = m.group(1).strip()
        headers.append((name, m.start(), m.end(), not _is_subsection(name)))

    # Iterate top-level headers and grab content until next top-level header
    for i, (name, start, end, is_top) in enumerate(headers):
        if not is_top:
            continue
        block_start = start
        # Find next top-level header's start
        j = i + 1
        block_end = len(text)
        while j < len(headers):
            n2, s2, e2, is_top2 = headers[j]
            if is_top2:
                block_end = s2
                break
            j += 1
        block_text = text[block_start:block_end].rstrip() + "\n"
        # If this top-level name is one we want, and we don't already have it, store it
        if name in wanted_names and name not in results:
            results[name] = block_text

    return results

def _extract_blocks(ff_dir: Path,
                    tdb_suffix: str,
                    names_to_find: List[str],
                    preferred_files: Tuple[str, ...]) -> Dict[str, str]:
    """
    Search across all matching *.tdb files (prioritizing names in preferred_files)
    and extract the given block names.
    """
    candidates = _gather_ff_tdb_files(ff_dir, tdb_suffix)
    if not candidates:
        raise RuntimeError(f"No '*{tdb_suffix}.tdb' files found in '{ff_dir}'")

    # Prioritize common merged filenames first (e.g., merged.n.tdb / merged.c.tdb)
    ordered_files = _find_first_existing(candidates, preferred_files)

    found: Dict[str, str] = {}
    remaining = set(names_to_find)

    for f in ordered_files:
        if not remaining:
            break
        text = f.read_text()
        blocks = _extract_named_blocks_from_text(text, list(remaining))
        for k, v in blocks.items():
            found[k] = v
        remaining -= set(blocks.keys())

    if remaining:
        searched = ", ".join(p.name for p in ordered_files)
        missing = ", ".join(sorted(remaining))
        raise RuntimeError(
            f"Missing required terminal entries [{missing}] in {tdb_suffix} databases. "
            f"Searched files: {searched}"
        )
    return found

def ensure_charmm_ff(version: str = 'auto') -> Path:
    """Copy user's CHARMM force field into FLOWBACK_FF with merged termini
    extracted directly from the force field's own terminal databases.
    """
    if version == 'auto' or version == 'charmm':
        keyword = 'charmm'
    else:
        keyword = version

    dest = Path(FLOWBACK_FF) / f"{keyword}.ff"
    if dest.exists():
        return dest

    gmxlibrary = os.environ.get("GMXLIB")
    if not gmxlibrary:
        raise RuntimeError("$GMXLIB is not set; cannot locate CHARMM force field")

    entries = [e for e in os.listdir(gmxlibrary)
               if e.lower().startswith(keyword) and e.endswith(".ff")]
    if not entries:
        raise RuntimeError(f"No CHARMM force field found in GMXLIB '{gmxlibrary}'")

    def version_key(name: str) -> int:
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    entries.sort(key=version_key)
    src = Path(gmxlibrary) / entries[-1]

    # Copy the entire force field directory first
    shutil.copytree(src, dest, dirs_exist_ok=True)

    # Extract needed terminal blocks from the *source* FF (before we delete in dest)
    # Prioritize merged files if present; still search others as fallback.
    # N-termini: exact names
    n_names = ["PRO-NH2+", "GLY-NH3+", "NH3+"]
    n_blocks = _extract_blocks(
        ff_dir=src,
        tdb_suffix=".n",
        names_to_find=n_names,
        preferred_files=("merged.n.tdb",)
    )

    # C-terminus: prefer COO-. Some CHARMM packs may name the default negative terminus "CTER".
    c_targets_ordered = ["COO-"]  # search order; we will pick the first found
    c_blocks_all = _extract_blocks(
        ff_dir=src,
        tdb_suffix=".c",
        names_to_find=c_targets_ordered,
        preferred_files=("merged.c.tdb",)
    )
    # Choose COO- if present; otherwise use CTER
    c_choice_name = "COO-" if "COO-" in c_blocks_all else "CTER"
    c_block_text = c_blocks_all[c_choice_name]

    # Now remove existing terminal DBs in the *destination* and write merged ones
    for pattern in ("*.n.tdb", "*.c.tdb"):
        for f in dest.glob(pattern):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    # Write merged.n.tdb with the three N-terminus entries in a fixed order
    merged_n_path = dest / "merged.n.tdb"
    merged_n_text = ""
    for name in n_names:
        merged_n_text += n_blocks[name].rstrip() + "\n\n"
    merged_n_path.write_text(merged_n_text.rstrip() + "\n")

    # Write merged.c.tdb with the chosen C-terminus entry
    merged_c_path = dest / "merged.c.tdb"
    merged_c_path.write_text(c_block_text)

    return dest







class EnergyModel(torch.nn.Module):
    def __init__(self, energy_func, topology):
        super().__init__()
        self.energy_func = energy_func
        self.topology_pdb = topology


    def forward(self, x, **kwargs):
        return EnergyFunction.apply(x, self.energy_func, self.topology_pdb).requires_grad_(True)
        
class EnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, energy_func, topology):
        """
        ctx: A context object for saving information for backward computation.
        input_tensor: The input tensor for which energy is computed.
        external_energy_func: A function that returns (energy, gradient).
        REQUIRES input_tensor to be coordinates IDENTICAL to topology_pdb
        """
        input_numpy = input_tensor.detach().cpu().numpy()  # Convert to NumPy
        energies, gradients = energy_func(topology, input_numpy)
        ctx.save_for_backward(torch.from_numpy(gradients).to(input_tensor.device))  # Save gradient
        return input_tensor.new_tensor(energies).requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient by applying the chain rule.
        """
        (external_gradient,) = ctx.saved_tensors
        result = grad_output * external_gradient
      
        return result, None, None, None  # Gradient w.r.t. input, ignore func

def charmm_traj_to_energy(topology: md.Topology, xyz: np.ndarray, ff_version: str = "auto"):
    gradients = np.zeros_like(xyz)
    energies = np.zeros(xyz.shape[0])
    for i in range(xyz.shape[0]):
        energy, gradient = charmm_structure_to_energy(topology, xyz[i:i+1], ff_version=ff_version)  # External function call
        energies[i] = energy
        gradients[i] = gradient
    #Divide by ten to convert from angstroms to nm
    return energies, gradients * angstrom / nanometer

# @time_elapsed
def charmm_structure_to_energy(topology: md.Topology, xyz: np.ndarray, nonbonded=True, ff_version: str = "auto"):
    t = md.Trajectory(xyz, topology)
    ff_dir = ensure_charmm_ff(ff_version)
    if np.max(compute_all_distances(t)) > 4 * t.top.n_residues ** 0.5:
        raise RuntimeError("Crazy Structure. Could Not Compute Energy")
    with tempfile.TemporaryDirectory() as temp_dir:
        pdb_file = f'{temp_dir}/temp.pdb'
        structure_file = f"{temp_dir}/structure.gro"
        topology_file = f"{temp_dir}/topol.top"
        
        t.save_pdb(pdb_file)
        commands = [
           "gmx_mpi", "pdb2gmx", "-f", pdb_file, "-o", structure_file, "-p", topology_file,
            "-ff", ff_dir.stem, "-water", "spce", "-ter", "-nobackup", "-quiet", "-i", f'{temp_dir}/posre.itp'
        ]
        env = os.environ.copy()
        env["GMXLIB"] = str(ff_dir.parent)
        process = subprocess.Popen(
            commands,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024 * 8,
            text=True,
            env=env,
        )
        
        
        stdout, stderr = process.communicate('0\n0\n')
        if process.returncode != 0:
            print(f"Energy calculation failed at pdb2gmx step. Error:\n{stderr}")
            return np.zeros_like(xyz)
        index_map = _map_original_to_processed_indices(pdb_file, structure_file)
        gro = GromacsGroFile(structure_file)
        original_box = gro.getPeriodicBoxVectors()
        expanded_box = (
            original_box[0] + Vec3(2, 0, 0) * nanometer,
            original_box[1] + Vec3(0, 2, 0) * nanometer,
            original_box[2] + Vec3(0, 0, 2) * nanometer,
        )
        topology = GromacsTopFile(
            topology_file,
            periodicBoxVectors=expanded_box,
            includeDir=str(ff_dir)
            # includeDir=[temp_dir, ff_dir.parent],
        )
        # Create simulation system
        
        system = topology.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer,
                                       constraints=HBonds)
        t2 = md.load(structure_file)
        
        selected_atoms = index_map
        new_bond_force = HarmonicBondForce()
        new_bond_force.setForceGroup(10)
        new_angle_force = HarmonicAngleForce()
        new_angle_force.setForceGroup(11)
        new_torsion_force = PeriodicTorsionForce()
        new_torsion_force.setForceGroup(12)
        new_custom_torsion_force = CustomTorsionForce("0.5*k*(thetap-theta0)^2; thetap = step(-(theta-theta0+pi))*2*pi+theta+step(theta-theta0-pi)*(-2*pi); pi = 3.14159265358979")
        new_custom_torsion_force.addPerTorsionParameter('theta0')
        new_custom_torsion_force.addPerTorsionParameter('k')
        new_custom_torsion_force.setForceGroup(13)
        
        for force in enumerate(system.getForces()):
            force_ = force[1]
            if isinstance(force_, HarmonicBondForce):
                for i in range(force_.getNumBonds()):
                    p1, p2, length, k = force_.getBondParameters(i)
                    if p1 in selected_atoms and p2 in selected_atoms:
                        new_bond_force.addBond(p1, p2, length, k)
        
            elif isinstance(force_, HarmonicAngleForce):
                for i in range(force_.getNumAngles()):
                    p1, p2, p3, angle, k = force_.getAngleParameters(i)
                    if p1 in selected_atoms and p2 in selected_atoms and p3 in selected_atoms:
                        new_angle_force.addAngle(p1, p2, p3, angle, k)
        
            elif isinstance(force_, PeriodicTorsionForce):
                for i in range(force_.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force_.getTorsionParameters(i)
                    if p1 in selected_atoms and p2 in selected_atoms and p3 in selected_atoms and p4 in selected_atoms:
                        new_torsion_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)

            
            elif isinstance(force_, CustomTorsionForce):
                for i in range(force_.getNumTorsions()):
                    p1, p2, p3, p4, params = force_.getTorsionParameters(i)
                    if p1 in selected_atoms and p2 in selected_atoms and p3 in selected_atoms and p4 in selected_atoms:
                        new_custom_torsion_force.addTorsion(p1, p2, p3, p4, params)
        # Remove old forces and add the new ones
        for fi in range(len(system.getForces()) - 1, -1, -1):
            force_ = system.getForce(fi)
            if isinstance(force_, CMAPTorsionForce):
                force_.setForceGroup(14)
                continue
            elif isinstance(force_, NonbondedForce) and nonbonded:
                mute_atoms = np.setdiff1d(np.arange(force_.getNumParticles()), selected_atoms)
                silence_atoms_and_shift_charge(force_, t2.top, mute_atoms)
                force_.setForceGroup(15)
                continue
            else:
                system.removeForce(fi)  # Remove each old force
        
        system.addForce(new_bond_force)
        system.addForce(new_angle_force)
        system.addForce(new_torsion_force)
        system.addForce(new_custom_torsion_force)

        # Set integrator
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        context = Context(system, integrator)
        context.setPositions(gro.positions)
        
        state = context.getState(getEnergy=True, getForces=True)
        
        forces = state.getForces(asNumpy=True)[index_map] 
  
        energy = state.getPotentialEnergy()

    return energy.value_in_unit(kilojoules_per_mole), -1 * forces

def rdkit_traj_to_energy(topology: md.Topology, xyz: np.ndarray):
    gradients = np.zeros_like(xyz)
    energies = np.zeros(xyz.shape[0])
    for i in range(xyz.shape[0]):
        energy, gradient = rdkit_structure_to_energy(topology, xyz[i:i+1])  # External function call
        energies[i] = energy
        gradients[i] = gradient
    return energies, gradients

def rdkit_structure_to_energy(topology: md.Topology, xyz: np.ndarray):
    """
    Convert an MDTraj topology and XYZ coordinates to an RDKit molecule.
    
    Args:
        topology (mdtraj.Topology): MDTraj topology object.
        xyz (np.ndarray): (N, 3) array of atomic coordinates.
        
    Returns:
        Chem.Mol: RDKit molecule with 3D coordinates.
    """
    blocker = rdBase.BlockLogs()
    t = md.Trajectory(xyz, topology)
    with tempfile.TemporaryDirectory() as temp_dir:
        t.save_pdb(f'{temp_dir}/temp.pdb')
        mol = Chem.MolFromPDBFile(f'{temp_dir}/temp.pdb')
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)  # Generates a 3D conformation
    # Set up the Universal Force Field (UFF)
    ff = AllChem.UFFGetMoleculeForceField(mol)
    
    # Compute Energy
    energy = ff.CalcEnergy()
    
    # Compute Gradients (negative of forces)
    num_atoms = mol.GetNumAtoms()
    
    gradients = np.array(ff.CalcGrad())
    # gradients = torch.from_numpy(gradients)
    return energy, gradients.reshape(-1, 3)

# @time_elapsed
def _map_original_to_processed_indices(original_pdb, processed_pdb):
    """
    Create a mapping from original PDB atom indices to processed PDB indices after pdb2gmx.
    
    Parameters:
    -----------
    original_pdb : str
        Path to the original PDB file
    processed_pdb : str
        Path to the processed PDB file (after pdb2gmx)
        
    Returns:
    --------
    list
        List where index position is the original atom index and value is the new index.
        -1 indicates no match was found for that original atom.
    """
    # Load both structures
    orig_u = mda.Universe(original_pdb)
    proc_u = mda.Universe(processed_pdb)
    
    # Initialize mapping list with -1 (indicating no match)
    n_orig_atoms = len(orig_u.atoms)
    index_map = -1 * np.ones(n_orig_atoms).astype(int)
    
    # Create unique identifiers for each atom in processed structure
    proc_dict = {}
    for atom in proc_u.atoms:
        identifier = (atom.resnum, atom.resname, atom.name)
        proc_dict[identifier] = atom.ix
    
    # Map original atoms to processed atoms
    for atom in orig_u.atoms:
        identifier = (atom.resnum, atom.resname, atom.name)
        if identifier in proc_dict:
            index_map[atom.ix] = proc_dict[identifier]
        elif atom.name == 'O':
            identifier = (atom.resnum, atom.resname, 'OT1')
            index_map[atom.ix] = proc_dict[identifier]
        elif atom.resname == 'ILE' and atom.name == 'CD1':
            identifier = (atom.resnum, atom.resname, 'CD')
            index_map[atom.ix] = proc_dict[identifier]
    return index_map

def silence_atoms_and_shift_charge(nbforce, topology, mute, context=None):
    """
    Zero LJ and charge on each atom in *mute* and transfer that charge
    to its first bonded neighbour that is **not** in *mute*.

    Parameters
    ----------
    nbforce : openmm.NonbondedForce
        The force we are modifying.
    topology : openmm.app.Topology
        Needed to discover bonding so we know where to move the charge.
    mute : iterable of int
        Particle indices to silence.
    context : openmm.Context, optional
        Pass the existing Context if you already created one so we can
        push the new parameters with `updateParametersInContext`.
    """
    mute    = set(int(i) for i in mute)
    shifts  = defaultdict(lambda: 0.0*elementary_charge)   # charge → heavy atom
    nbforce.setUseDispersionCorrection(False)
    # nbforce.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    # ------------------------------------------------------------------
    # 1. build a quick neighbour map from the Topology
    # ------------------------------------------------------------------
    neighbours = defaultdict(list)
    for bond in topology.bonds:
        i, j = bond[0].index, bond[1].index
        neighbours[i].append(j)
        neighbours[j].append(i)

    # ------------------------------------------------------------------
    # 2. zero LJ & charge on the muted atoms, remember the charge removed
    # ------------------------------------------------------------------
    for idx in mute:
        q, sigma, eps = nbforce.getParticleParameters(idx)

        # accumulate the charge so we can add it to the parent later
        if abs(q.value_in_unit(elementary_charge)) > 1e-12:
            # first bonded neighbour that is NOT muted
            try:
                parent = next(n for n in neighbours[idx] if n not in mute)
                shifts[parent] += q
            except StopIteration:
                warnings.warn(f"Atom {idx} is muted but has no non-muted neighbours;"
                              " total charge will not be conserved!")

        # zero LJ *and* charge
        nbforce.setParticleParameters(idx,
                                      0.0*elementary_charge,
                                      sigma,
                                      0.0*kilojoule_per_mole)

    # ------------------------------------------------------------------
    # 3. apply the accumulated charge shifts to the parent atoms
    # ------------------------------------------------------------------
    for idx, dq in shifts.items():
        q, sigma, eps = nbforce.getParticleParameters(idx)
        nbforce.setParticleParameters(idx, q + dq, sigma, eps)

    # ------------------------------------------------------------------
    # 4. update every exception so chargeProd matches the new charges
    # ------------------------------------------------------------------
    for k in range(nbforce.getNumExceptions()):
        i, j, qprod, sigma, eps = nbforce.getExceptionParameters(k)
        
        qi = nbforce.getParticleParameters(i)[0]
        qj = nbforce.getParticleParameters(j)[0]
        if topology.atom(i).element.name == 'hydrogen' or topology.atom(j).element.name =='hydrogen':
            nbforce.setExceptionParameters(k, i, j, qi*qj, sigma, 0.0*kilojoule_per_mole)
        else:
            nbforce.setExceptionParameters(k, i, j, qi*qj, sigma, eps)
