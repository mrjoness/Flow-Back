import os
from file_config import FLOWBACK_BASE, FLOWBACK_DATA, FLOWBACK_OUTPUTS
from src.utils.model import bond_fraction, clash_res_percent
import mdtraj as md
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed            # NEW!
import re
import argparse
import warnings
warnings.filterwarnings("ignore") 

def _natural_key(path: Path):
    """
    Return a list that lets `sorted()` put Path objects
    in “human” (natural) order: struc2.pdb < struc11.pdb.
    """
    # Split the file stem into text and digit runs: ["struc", "2", ""].
    # Digits are converted to int, everything else to lowercase str.
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r'(\d+)', path.stem)
    ]

def load_folder_as_trajectory(folder_path: str) -> md.Trajectory:
    """
    Combine all PDB files in `folder_path` into one MDTraj Trajectory,
    using natural sorting so that struc2.pdb < struc11.pdb.

    Parameters
    ----------
    folder_path : str
        Directory containing PDB files with a consistent topology.

    Returns
    -------
    md.Trajectory
    """
    folder = Path(folder_path).expanduser().resolve()

    pdb_files = sorted(folder.glob("*.pdb"), key=_natural_key)
    if not pdb_files:
        raise ValueError(f"No PDB files found in {folder}")

    # md.load(List[str]) concatenates along the time axis
    try:
        traj = md.load([str(p) for p in pdb_files])
    except Exception as err:
        # Fallback: load one file at a time and join
        frames = [md.load(str(pdb_files[0]))]
        for pdb_file in tqdm(pdb_files[1:]):
            frames.append(md.load(str(pdb_file)))
        traj = frames[0].join(frames[1:], check_topology=True)
        if traj.n_frames != len(pdb_files):
            raise RuntimeError("Failed to concatenate PDB files") from err
    traj.save_xtc(f'{folder_path}/traj.xtc')
    return traj

def load_or_generate_trajectory(path, generate_func, top_file, num_frames=-1):
    traj_path = os.path.join(path, 'traj.xtc')
    if os.path.exists(traj_path):
        traj = md.load(traj_path, top=top_file)
        if len(traj) < num_frames:
            traj = generate_func()
            traj.save_xtc(traj_path)
            return traj
        else:
            return traj
    else:
        traj = generate_func()
        traj.save_xtc(traj_path)
        return traj

# ----------------------------  PARALLEL PART  ------------------------------- #
def compute_clash_list(traj: md.Trajectory, n_jobs: int) -> np.ndarray:
    """
    Compute clash_res_percent for every frame in *traj* using n_jobs workers.
    """
    print(f"  ↳ Computing clashes with {n_jobs} parallel workers …")
    clash_list = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        delayed(clash_res_percent)(traj[i]) for i in range(len(traj))
    )
    return np.asarray(clash_list)

# --------------------------------------------------------------------------- #
def main(list_file: str, n_jobs: int, hpack):

    with open(list_file) as fh:
        proteins = [ln.strip() for ln in fh if ln.strip()]
    print(f"Loaded {len(proteins)} protein IDs from {list_file}\n")

    # models1 = ['lr5_post_ckp-2000', 'big_model_ckp-15', 'big_model_ckp-15_euler']
    models = ['n2503_post_ckp-7000_noise-0.003_chi_0.25',
               'big_model_ckp-15_noise-0.003_chi_0.25',
               'big_model_ckp-15_euler_noise-0.003_chi_0.2']

    for protein in proteins:
        print(f"=== {protein} ===")
        top_file = f"{FLOWBACK_DATA}/{protein}_clean_AA/frame_00000.pdb"
        if hpack:
            print(f"• Model: Hpacker")
            
            traj_dir = f"{FLOWBACK_DATA}/hpack_{protein}"
            
            traj = load_or_generate_trajectory(
                traj_dir,
                lambda: load_folder_as_trajectory(traj_dir),
                top_file
            )
            if os.path.isfile(f'{traj_dir}/cls.npy') and len(np.load(f'{traj_dir}/cls.npy')) == len(traj):
                continue
            clash_arr = compute_clash_list(traj, n_jobs=n_jobs)
            np.save(f"{traj_dir}/cls.npy", clash_arr)
            print(f"  ↳ Saved {len(clash_arr)} clash values to {traj_dir}/cls.npy\n")
        else:
            for model in models:
                chosen = model
                print(f"• Model: {chosen}")
                
                traj_dir = f"{FLOWBACK_OUTPUTS}/{protein}/{chosen}"
                
                traj = load_or_generate_trajectory(
                    traj_dir,
                    lambda: load_folder_as_trajectory(traj_dir),
                    top_file
                )
                if os.path.isfile(f'{traj_dir}/cls.npy') and len(np.load(f'{traj_dir}/cls.npy')) == len(traj):
                    continue
                clash_arr = compute_clash_list(traj, n_jobs=n_jobs)
                np.save(f"{traj_dir}/cls.npy", clash_arr)
                print(f"  ↳ Saved {len(clash_arr)} clash values to {traj_dir}/cls.npy\n")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute clash statistics in parallel.")
    parser.add_argument("--list_file", default=f"{FLOWBACK_BASE}/test_proteins.txt",
                        help="Path to file with one protein ID per line")
    parser.add_argument("--n_jobs", type=int, default=os.cpu_count(),
                        help="How many worker processes to use (default: all cores)")
    parser.add_argument("--hpack", action="store_true", help="Hpacker clashes")
    args = parser.parse_args()
    main(args.list_file, args.n_jobs, args.hpack)
        
        
        