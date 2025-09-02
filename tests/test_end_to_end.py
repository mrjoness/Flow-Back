import pytest
pytest.importorskip('numpy')
pytest.importorskip('rdkit')
pytest.importorskip('mdtraj')
pytest.importorskip('torch')

import numpy as np
import mdtraj as md
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile

from src.utils.energy import rdkit_traj_to_energy, EnergyModel


def _build_rdkit_molecule():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    AllChem.UFFOptimizeMolecule(mol)
    block = Chem.MolToPDBBlock(mol)
    with tempfile.NamedTemporaryFile('w+', suffix='.pdb') as tmp:
        tmp.write(block)
        tmp.flush()
        traj = md.load(tmp.name)
    return mol, traj.topology, traj.xyz.astype(np.float32)


def test_rdkit_energy_model_end_to_end():
    mol, top, xyz = _build_rdkit_molecule()
    energies, grads = rdkit_traj_to_energy(top, xyz)
    assert energies.shape == (1,)
    assert grads.shape == xyz.shape
    assert np.all(np.isfinite(grads))

    ff = AllChem.UFFGetMoleculeForceField(mol)
    expected = ff.CalcEnergy()
    assert np.allclose(energies[0], expected, atol=1e-5)

    model = EnergyModel(rdkit_traj_to_energy, top)
    x = torch.tensor(xyz, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert torch.all(torch.isfinite(x.grad))
