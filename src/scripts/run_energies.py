import argparse
import mdtraj as md

from src.file_config import FLOWBACK_OUTPUTS
from src.utils.run_energy import run_energy_pipeline
from pathlib import Path
# def compute_energy(pdb_path: str, ff: str) -> float | None:
#     """Return the potential energy for a single PDB file."""
#     try:
#         pdb = md.load(pdb_path)
#         if ff == 'charmm':
#             energy, _ = charmm_structure_to_energy(pdb.top, pdb.xyz, nonbonded=True)
#         elif ff == 'abmer':
#             amber_solv_structure_to_energy(pdb.top, pdb.xyz
#         else:
#             raise ValueError('No FF implemented')
#         return energy
#     except Exception as e:
#         print(f"Error processing {pdb_path}: {e}")
#         return None


# def run_energy_pipeline(pdb_paths: list[str], output_file: str) -> None:
#     """Compute energies for all paths and save them to *output_file*."""
#     n_total = len(pdb_paths)
#     print(f"Computing energies for {n_total} structures -> {output_file}")

#     with mp.Pool(mp.cpu_count()) as pool:
#         energies = pool.map(compute_energy, pdb_paths)

#     energies = np.array([e for e in energies if e is not None])
#     np.save(output_file, energies)

#     print(
#         f"Finished. {n_total - len(energies)} structures failed. "
#         f"Energies saved to {output_file}."
#     )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute energies from generated PDB files"
    )
    parser.add_argument("--data", required=True, help="Dataset name used during evaluation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint number")
    parser.add_argument("--noise", default="0.003", help="CG noise used during evaluation")
    parser.add_argument(
        "--ff",
        default="charmm",
        choices=["charmm", "amber"],
        help="Force field backend to use",
    )
    parser.add_argument(
        "--charmm-ff",
        default="auto",
        help="CHARMM force field version (auto = latest)",
    )
    args = parser.parse_args()

    if args.ff == "charmm":
        ff_dir = ensure_charmm_ff(args.charmm_ff)
        ff_label = ff_dir.stem
    else:
        ff_label = "amber"

    base_dir = Path(FLOWBACK_OUTPUTS) / args.data
    pattern = f"{args.model}_ckp-{args.checkpoint}*noise-{args.noise}"
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No directory matching {pattern} in {base_dir}")

    pdb_dir = matches[0]
    model_ckp = pdb_dir.name
    pdb_paths = sorted(str(p) for p in pdb_dir.glob("*.pdb"))

    output_dir = Path(FLOWBACK_OUTPUTS) / "energies"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"energies_{ff_label}_{args.data}_{model_ckp}.npy"
    run_energy_pipeline(pdb_paths, str(output_file), ff=args.ff, charmm_ff=args.charmm_ff)


if __name__ == "__main__":
    main()

