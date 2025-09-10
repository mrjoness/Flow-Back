import os
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import MDAnalysis as mda
import mdtraj as md
import numpy as np
from openmm import NonbondedForce
from openmm.unit import elementary_charge, kilojoule_per_mole
from src.file_config import FLOWBACK_FF


def _osremove(f):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass


def compute_all_distances(traj):
    idxs = np.arange(traj.top.n_atoms)
    grid = np.array(np.meshgrid(idxs, idxs)).T.reshape(-1, 2)
    pairs = grid[grid[:, 0] > grid[:, 1]]
    return md.compute_distances(traj, pairs)

_TOPLEVEL_SUBSECTIONS = {"replace", "add", "delete"}
_BLOCK_HEADER_RE = re.compile(r'^\[\s*([^\]]+?)\s*\]\s*$', re.MULTILINE)


def _is_subsection(name: str) -> bool:
    return name.strip().lower() in _TOPLEVEL_SUBSECTIONS


def _find_first_existing(files: List[Path], preferred_names: Tuple[str, ...]) -> List[Path]:
    preferred, others = [], []
    for f in files:
        (preferred if f.name in preferred_names else others).append(f)
    return preferred + others


def _gather_ff_tdb_files(ff_dir: Path, suffix: str) -> List[Path]:
    return sorted(ff_dir.glob(f"*{suffix}.tdb"))


def _extract_named_blocks_from_text(text: str, wanted: List[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    matches = list(_BLOCK_HEADER_RE.finditer(text))
    headers = [(m.group(1).strip(), m.start(), not _is_subsection(m.group(1))) for m in matches]
    for i, (name, start, is_top) in enumerate(headers):
        if not is_top:
            continue
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        if name in wanted and name not in results:
            results[name] = text[start:end].rstrip() + "\n"
    return results


def _extract_blocks(ff_dir: Path, tdb_suffix: str, names_to_find: List[str], preferred_files: Tuple[str, ...]) -> Dict[str, str]:
    candidates = _gather_ff_tdb_files(ff_dir, tdb_suffix)
    if not candidates:
        raise RuntimeError(f"No '*{tdb_suffix}.tdb' files found in '{ff_dir}'")
    ordered_files = _find_first_existing(candidates, preferred_files)
    found: Dict[str, str] = {}
    remaining = set(names_to_find)
    for f in ordered_files:
        if not remaining:
            break
        blocks = _extract_named_blocks_from_text(f.read_text(), list(remaining))
        found.update(blocks)
        remaining -= set(blocks)
    if remaining:
        searched = ", ".join(p.name for p in ordered_files)
        missing = ", ".join(sorted(remaining))
        raise RuntimeError(
            f"Missing required terminal entries [{missing}] in {tdb_suffix} databases. Searched files: {searched}"
        )
    return found

def ensure_charmm_ff(version: str = 'auto') -> Path:
    keyword = 'charmm' if version in ('auto', 'charmm') else version
    dest = Path(FLOWBACK_FF) / f"{keyword}.ff"
    if dest.exists():
        return dest
    gmxlibrary = os.environ.get('GMXLIB')
    if not gmxlibrary:
        raise RuntimeError('$GMXLIB is not set; cannot locate CHARMM force field')
    entries = [e for e in os.listdir(gmxlibrary) if e.lower().startswith(keyword) and e.endswith('.ff')]
    if not entries:
        raise RuntimeError(f"No CHARMM force field found in GMXLIB '{gmxlibrary}'")
    entries.sort(key=lambda n: int(re.search(r"(\d+)", n).group(1)) if re.search(r"(\d+)", n) else 0)
    src = Path(gmxlibrary) / entries[-1]
    shutil.copytree(src, dest, dirs_exist_ok=True)
    n_names = ["PRO-NH2+", "GLY-NH3+", "NH3+"]
    n_blocks = _extract_blocks(src, ".n", n_names, ("merged.n.tdb",))
    c_blocks_all = _extract_blocks(src, ".c", ["COO-"], ("merged.c.tdb",))
    if "COO-" in c_blocks_all:
        c_block_text = c_blocks_all["COO-"]
    else:
        c_block_text = _extract_blocks(src, ".c", ["CTER"], ("merged.c.tdb",))["CTER"]
    for pattern in ("*.n.tdb", "*.c.tdb"):
        for f in dest.glob(pattern):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
    merged_n = "".join(n_blocks[name].rstrip() + "\n\n" for name in n_names).rstrip() + "\n"
    (dest / "merged.n.tdb").write_text(merged_n)
    (dest / "merged.c.tdb").write_text(c_block_text)
    return dest

def _map_original_to_processed_indices(original_pdb, processed_pdb):
    orig_u = mda.Universe(original_pdb)
    proc_u = mda.Universe(processed_pdb)
    index_map = -1 * np.ones(len(orig_u.atoms), dtype=int)
    proc_dict = {(a.resnum, a.resname, a.name): a.ix for a in proc_u.atoms}
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
    mute = set(int(i) for i in mute)
    shifts = defaultdict(lambda: 0.0 * elementary_charge)
    nbforce.setUseDispersionCorrection(False)
    neighbours = defaultdict(list)
    for bond in topology.bonds:
        i, j = bond[0].index, bond[1].index
        neighbours[i].append(j)
        neighbours[j].append(i)
    for idx in mute:
        q, sigma, eps = nbforce.getParticleParameters(idx)
        if abs(q.value_in_unit(elementary_charge)) > 1e-12:
            try:
                parent = next(n for n in neighbours[idx] if n not in mute)
                shifts[parent] += q
            except StopIteration:
                warnings.warn(
                    f"Atom {idx} is muted but has no non-muted neighbours; total charge will not be conserved!"
                )
        nbforce.setParticleParameters(idx, 0.0 * elementary_charge, sigma, 0.0 * kilojoule_per_mole)
    for idx, dq in shifts.items():
        q, sigma, eps = nbforce.getParticleParameters(idx)
        nbforce.setParticleParameters(idx, q + dq, sigma, eps)
    for k in range(nbforce.getNumExceptions()):
        i, j, qprod, sigma, eps = nbforce.getExceptionParameters(k)
        qi = nbforce.getParticleParameters(i)[0]
        qj = nbforce.getParticleParameters(j)[0]
        if topology.atom(i).element.name == 'hydrogen' or topology.atom(j).element.name == 'hydrogen':
            nbforce.setExceptionParameters(k, i, j, qi*qj, sigma, 0.0 * kilojoule_per_mole)
        else:
            nbforce.setExceptionParameters(k, i, j, qi*qj, sigma, eps)
    if context is not None:
        nbforce.updateParametersInContext(context)
