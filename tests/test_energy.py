import pytest

pytest.importorskip('rdkit')
pytest.importorskip('openmm')
pytest.importorskip('numpy')
pytest.importorskip('torch')
pytest.importorskip('mdtraj')

import numpy as np
import torch
import mdtraj as md

from src.utils.energy_helpers import compute_all_distances
from src.utils.energy import EnergyModel

def _create_simple_traj():
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue('ALA', chain)
    top.add_atom('C1', md.element.carbon, res)
    top.add_atom('C2', md.element.carbon, res)
    top.add_atom('C3', md.element.carbon, res)
    xyz = np.array([[[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 2.0, 0.0]]], dtype=np.float32)
    traj = md.Trajectory(xyz, top)
    return traj

def dummy_energy(topology, coords):
    energy = np.sum(coords ** 2, axis=(1, 2))
    gradient = 2 * coords
    return energy, gradient

def test_compute_all_distances():
    traj = _create_simple_traj()
    dists = compute_all_distances(traj)
    assert dists.shape == (1, 3)
    expected = np.array([[1.0, 2.0, np.sqrt(5.0)]])
    assert np.allclose(dists, expected, atol=1e-6)

def test_energy_model_autograd():
    model = EnergyModel(dummy_energy, topology=None)
    x = torch.randn(1, 2, 3, requires_grad=True)
    energy = model(x)
    energy.sum().backward()
    assert torch.allclose(x.grad, 2 * x)
