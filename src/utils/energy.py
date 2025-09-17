from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import mdtraj as md
import torch
from rdkit import rdBase
from openmm import *
import os
from openmm.app import *
from openmm.unit import *
import tempfile
import subprocess
from .energy_helpers import *
import warnings

warnings.filterwarnings('ignore')


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
    with tempfile.TemporaryDirectory(dir='/project2/andrewferguson/berlaga') as temp_dir:
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
        index_map = map_original_to_processed_indices(pdb_file, structure_file)
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

def amber_solv_traj_to_energy(topology: md.Topology, xyz: np.ndarray):
    gradients = np.zeros_like(xyz)
    energies = np.zeros(xyz.shape[0])
    for i in range(xyz.shape[0]):
        energy, gradient = amber_solv_structure_to_energy(topology, xyz[i:i+1])  # External function call
        energies[i] = energy
        gradients[i] = gradient
    return energies, gradients * angstrom / nanometer

def amber_solv_structure_to_energy(topology: md.Topology, xyz: np.ndarray):
    t = md.Trajectory(xyz, topology)
    

    with tempfile.TemporaryDirectory() as temp_dir:
        pdb_file = f'{temp_dir}/temp.pdb'
        t.save_pdb(pdb_file)
    # --- AMBER14 with implicit solvent (GBn2) ---
        fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    # ---- Example usage ----
    # Build OpenMM objects from the fixed structure
    
    
    # Snapshot BEFORE adding atoms
    fixer.findMissingAtoms()
    ctr0, _ = counter_from_topology(fixer.topology)
    
    fixer.addMissingAtoms()
    
    # Snapshot AFTER addMissingAtoms, BEFORE hydrogens
    ctr1, idx_after_atoms = counter_from_topology(fixer.topology)
    added_heavy, added_heavy_idxs = diff_added_atoms(ctr0, ctr1, idx_after_atoms)
    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    fixer.addMissingHydrogens(pH=7.0, forcefield=ff)
    topology  = fixer.topology
    positions = fixer.positions
    # Snapshot AFTER hydrogens
    ctr2, idx_after_H = counter_from_topology(fixer.topology)
    added_h, added_h_idxs = diff_added_atoms(ctr1, ctr2, idx_after_H)
    # Define the output PDB filename

    
    # print(f"\nAdded {len(added_h)} hydrogens:")
    system = ff.createSystem(
        fixer.topology,
        nonbondedMethod=NoCutoff,   # implicit solvent: NoCutoff / CutoffNonPeriodic / CutoffPeriodic only
        constraints=HBonds
    )
    
    added_idxs = np.concatenate([added_heavy_idxs, added_h_idxs])

    move_set = set(added_idxs)  # e.g., all hydrogens + all atoms in N- and C-termini
    # Build a 0/1 mask per DoF (1 = movable, 0 = frozen)
    n = system.getNumParticles()
    mask_vecs = [Vec3(1,1,1) if i in move_set else Vec3(0,0,0) for i in range(n)]
    
    # Custom "minimizer" integrator: naive steepest descent with a tiny step
    integ = CustomIntegrator(0.0)
    integ.addPerDofVariable("m", 0)         # per-DoF mask
    integ.addGlobalVariable("alpha", 1e-7)  # small step size (nm / (kJ/mol/nm)); tuned empirically
    integ.addComputePerDof("x", "x + alpha*m*f")
    integ.addConstrainPositions()
    
    # IMPORTANT: set the mask AFTER Simulation is created
    integ.setPerDofVariableByName("m", mask_vecs)

    platform = None
    platform_props = {}
    for name, props in [
        ("CUDA", {"DeviceIndex": "0", "Precision": "mixed", "DeterministicForces": "true"}),
        ("OpenCL", {"DeviceIndex": "0", "Precision": "single"}),
        ("CPU", {}),
    ]:
        try:
            platform = Platform.getPlatformByName(name)
            platform_props = props
            break
        except Exception:
            continue
    if platform.getName() == "CPU":
        warnings.warn("Falling back to CPU platform. This will be slower.", RuntimeWarning)
    sim = Simulation(topology, system, integ, platform, platform_props)
    sim.context.setPositions(positions)
    
    ref_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    for _ in range(500):
        sim.step(10)
    heavy_idxs = reset_nonH_nonOXT_positions(sim, ref_positions)

    state = sim.context.getState(getEnergy=True, getForces=True)
    
    forces = state.getForces(asNumpy=True)
    heavy_forces = forces[heavy_idxs, :] 

    energy = state.getPotentialEnergy()

    return energy.value_in_unit(kilojoules_per_mole), -1 * heavy_forces
