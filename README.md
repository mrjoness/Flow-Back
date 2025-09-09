# FlowBack & FlowBack-Adjoint

**FlowBack**: Generative backmapping of coarse-grained molecular systems using an equivariant graph neural network and a flow-matching objective. Implementation of https://openreview.net/forum?id=mhUasr0j5X.

**FlowBack-Adjoint**: Extends FlowBack with an **adjoint matching** scheme that adds **physics-aware, energy-guided corrections** during **post-training**. You can post-train the original FlowBack model using **RDKit** energies or **CHARMM** energies (**recommended**). For CHARMM-based post-training you must have a working installation of **GROMACS** and the desired **CHARMM** force field (e.g., CHARMM27/CHARMM36). **Inference/integration is unchanged** relative to the original FlowBack pipeline.

---

## Installation

### Env setup
Clone the repo and run the setup script to create and activate a local virtual environment using `pip`:

    git clone https://github.com/mrjoness/Flow-Back.git
    cd Flow-Back
    source setup.sh

If you prefer Conda, use the provided environment specification instead:

    conda env create -f environment.yml
    conda activate flowback_env

All commands below use the pattern:

    python -m src.scripts.<script> [options] --config configs/<script>.yaml

### Quick test
    python -m src.scripts.eval --load_dir PDB_example --config configs/eval.yaml

---

## Inference (FlowBack & FlowBack-Adjoint)

The **integration (inference)** procedure is identical for base FlowBack and FlowBack-Adjoint models. Edit `configs/eval.yaml` to control parameters such as `n_gens`, `CG_noise`, and clash checking.

Generate samples for coarse-grained traces:

    python -m src.scripts.eval --load_dir PDB_example --config configs/eval.yaml

To compute bond, clash, or diversity metrics, set `retain_AA`, `check_bonds`, `check_clash`, and `check_div` in `configs/eval.yaml` and run the same command.

To increase diversity, adjust `CG_noise` in the config and rerun the command.

Backmap a short (10-frame) CG trajectory:

    python -m src.scripts.eval --load_dir pro_traj_example --config configs/eval.yaml

---

## Pre-Training (base FlowBack)

Download training PDBs, post-training PDBs, force-field parameters, and pre-processed features from:

    https://zenodo.org/records/17081237

Unzip and move `train_features` into the "inputs" folder of the working directory. Move post-training PDBs into the `data` folder. Edit `configs/pre_train.yaml` to specify `load_path` and `top_path` for the feature and topology pickles.

Run pre-training:

    python -m src.scripts.pre_train --config configs/pre_train.yaml

---

## Post-Training with FlowBack-Adjoint (Energy-Guided)

**Goal**: Refine a pre-trained FlowBack model by incorporating **energy terms** in an adjoint matching objective. Edit `configs/post_train.yaml` to choose the energy backend (`ff`), energy-loss weight (`lam`), and other training hyperparameters.

If `ff` is set to `CHARMM`, ensure that both GROMACS and the desired CHARMM force field are installed; they are required to compute CHARMM energies during post-training.

Run post-training:

    python -m src.scripts.post_train --config configs/post_train.yaml

---

## Running `run_energies` on evaluation outputs

After you've generated PDB files via evaluation, compute their CHARMM energies with:
```
python -m src.scripts.run_energies --data PDB_example --model post_train --checkpoint 7000 --noise 0.003
```
The script searches `outputs/PDB_example` for a directory matching
`post_train_ckp-7000_noise-0.003/`, processes all contained .pdb files, and saves the resulting energy array to `outputs/energies/energies_PDB_example_post_train_ckp-7000_noise-0.003.npy`.

## Cite as

    @article{jones2025flowback,
      title={FlowBack: A Generalized Flow-Matching Approach for Biomolecular Backmapping},
      author={Jones, Michael S and Khanna, Smayan and Ferguson, Andrew L},
      journal={Journal of Chemical Information and Modeling},
      year={2025},
      publisher={ACS Publications}
    }

    @misc{berlaga2025flowbackadjointphysicsawareenergyguidedconditional,
      title={FlowBack-Adjoint: Physics-Aware and Energy-Guided Conditional Flow-Matching for All-Atom Protein Backmapping},
      author={Alex Berlaga and Michael S. Jones and Andrew L. Ferguson},
      year={2025},
      eprint={2508.03619},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.03619}
    }

