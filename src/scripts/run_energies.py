import argparse
import glob
import mdtraj as md
import multiprocessing as mp
import numpy as np
import os
from datetime import datetime
from pathlib import Path

from file_config import FLOWBACK_DATA, FLOWBACK_OUTPUTS
from src.utils.energy import charmm_structure_to_energy




def compute_energy(i, pdb_template):
    pdb_path = pdb_template.format(i)
    try:
        pdb = md.load(pdb_path)
        energy, _ = charmm_structure_to_energy(pdb.top, pdb.xyz, nonbonded=True)
        return i, energy
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return i, None

def gather_pdb_paths(args) -> list[str]:
    """
    Build a flat list of PDB files that should be processed, according to CLI
    switches.  All code that *decides what to run* lives in this one place.
    """
    # 1) --- “short” set -------------------------------------------------------
    if args.short:
        base = Path(FLOWBACK_DATA) / 'train'
        with open(base / 'under_50.txt') as fh:
            return [str(base / f'{line.strip()}.pdb') for line in fh]

    # 2) --- “valid” set -------------------------------------------------------
    elif args.valid:
        base = Path(FLOWBACK_DATA) / 'valid_clean_AA'
        return [str(p) for p in base.glob('*.pdb')]

    else:
        # 3) --- “no‑model” set ----------------------------------------------------
        if args.nomodel:
            if args.bioemu:
                prefix = f'{FLOWBACK_DATA}/{args.data}'
            else:
                prefix = f'{FLOWBACK_DATA}/{args.data}_clean_AA'
        # 4) --- model‑generated structures ---------------------------------------
        else:
            if args.nonoise:
                model_ckp = f'{args.model}_ckp-{args.checkpoint}'
            else:
                model_ckp = f'{args.model}_ckp-{args.checkpoint}_noise-{args.noise}'
            if args.chi != '':
                model_ckp += f'_chi_{args.chi}'
            prefix = f'{FLOWBACK_OUTPUTS}/{args.data}/{model_ckp}'

    
        # 4a) “discover everything in the folder” (num_structures == 0) -----------
        if args.num_structures == 0:
            pattern = f'{prefix}/*.pdb'
            return glob.glob(pattern)
        # 4b) “template + counter” -------------------------------------------------
        #      possibly using MANY frame names from a text file
        else:
            template = Path(
                f'{prefix}/{args.frame_name}_{{}}{args.suffix}.pdb'
            )
            paths = [str(template).format(i) for i in range(args.num_structures)]
            return paths

def run_energy_pipeline(pdb_paths: list[str], output_file: str, save_dict: bool) -> None:
    """
    Drive multiprocessing + save *.npy*.
    """
    n_total = len(pdb_paths)
    print(f'Computing energies for {n_total} structures → {output_file}')
    print(datetime.now())

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(compute_energy, [(i, p) for i, p in enumerate(pdb_paths)])

    print(datetime.now())

    # keep successful ones, preserving original ordering
    results.sort()
    successful = [(pdb_paths[idx], energy) for idx, energy in results if energy is not None]

    if save_dict:
        output_data = {pdb: energy for pdb, energy in successful}
    else:
        output_data = np.array([energy for _, energy in successful])

    if save_dict:
        # Save dictionary using np.save (which uses .npy, stores Python object)
        np.save(output_file, output_data, allow_pickle=True)
    else:
        np.save(output_file, output_data)

    print(
        f'Finished. {n_total - len(successful)} structures failed. '
        f'Energies saved to {output_file}.'
    )


def default_output_name(args) -> str:
    """
    Replicates the naming scheme you used before.
    """
    if args.short:
        return f'{FLOWBACK_OUTPUTS}/energy_files/energies_short_pdbs_nomodel.npy'
    if args.valid:
        return f'{FLOWBACK_OUTPUTS}/energy_files/energies_true_valid.npy'
    if args.nomodel:
        return f'{FLOWBACK_OUTPUTS}/energy_files/energies_{args.data}_nomodel.npy'

    model_ckp = f'{args.model}_ckp-{args.checkpoint}_noise-{args.noise}'
    if args.chi != '':
        model_ckp += f'_chi_{args.chi}'

    
    return f'{FLOWBACK_OUTPUTS}/energy_files/energies_{args.data}_{model_ckp}.npy'

def main():
    parser = argparse.ArgumentParser(
        description='Compute energies from PDB files using a classical force‑field'
    )
    # dataset selection
    parser.add_argument('--short', action='store_true', help='all short pdbs')
    parser.add_argument('--valid', action='store_true', help='all valid pdbs')
    parser.add_argument('--nomodel', action='store_true', help='just run energies on the data')
    parser.add_argument('--bioemu', action='store_true', help='bioemu structures')
    parser.add_argument('--nonoise', action='store_true', help='bioemu structures')
    # model‑specific parameters
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint number')
    parser.add_argument('--noise', default='0.003', help='CG noise')
    parser.add_argument('--chi', type=str, default='0.55', help='chi')
    # generic parameters
    parser.add_argument('--data', default='finetune_chignolin')
    parser.add_argument('--num_structures', '-n', type=int, default=0)

    # frame‑name handling
    parser.add_argument('--frame_name', default='frame', help='single frame name')
    parser.add_argument('--suffix', default='_1')
    parser.add_argument(
    "--save_dict", action="store_true",
    help="Save a dictionary of {pdb_filename: energy} instead of just an array"
    )

    args = parser.parse_args()

    pdb_paths = gather_pdb_paths(args)
    output_file = default_output_name(args)
    run_energy_pipeline(pdb_paths, output_file, args.save_dict)


if __name__ == '__main__':
    main()
