import pytest
pytest.importorskip('mdtraj')
import mdtraj as md
from src.utils.chi import get_atom_indices_by_name, get_dihed_idxs


def _build_topology_with_gly():
    top = md.Topology()
    chain = top.add_chain()

    def add_res(name, has_cb=True):
        res = top.add_residue(name, chain)
        top.add_atom('N', md.element.nitrogen, res)
        top.add_atom('CA', md.element.carbon, res)
        if has_cb:
            top.add_atom('CB', md.element.carbon, res)
        top.add_atom('C', md.element.carbon, res)
        top.add_atom('O', md.element.oxygen, res)
        return res

    res1 = add_res('ALA', has_cb=True)
    res2 = add_res('GLY', has_cb=False)
    return top, res1, res2


def _build_topology_two_res():
    top = md.Topology()
    chain = top.add_chain()
    for name in ['ALA', 'VAL']:
        res = top.add_residue(name, chain)
        top.add_atom('N', md.element.nitrogen, res)
        top.add_atom('CA', md.element.carbon, res)
        top.add_atom('CB', md.element.carbon, res)
        top.add_atom('C', md.element.carbon, res)
        top.add_atom('O', md.element.oxygen, res)
    return top


def test_get_atom_indices_by_name():
    top, res1, res2 = _build_topology_with_gly()
    idxs = get_atom_indices_by_name(top, res1, ['N', 'CA', 'CB', 'C'])
    assert idxs[0] == res1.atom('N').index
    assert idxs[2] == res1.atom('CB').index

    idxs2 = get_atom_indices_by_name(top, res2, ['N', 'CA', 'CB', 'C'])
    assert idxs2[2] is None


def test_get_dihed_idxs():
    top = _build_topology_two_res()
    dihed = get_dihed_idxs(top)
    assert len(dihed) == 2
    n_idx = top.residue(0).atom('N').index
    ca_idx = top.residue(0).atom('CA').index
    cb_idx = top.residue(0).atom('CB').index
    c_idx = top.residue(0).atom('C').index
    assert dihed[0] == [n_idx, ca_idx, cb_idx, c_idx]
