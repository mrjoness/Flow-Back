# FlowBack & FlowBack-Adjoint

**FlowBack**: Generative backmapping of coarse-grained molecular systems using an equivariant graph neural network and a flow-matching objective. Implementation of https://openreview.net/forum?id=mhUasr0j5X.

**FlowBack-Adjoint**: Extends FlowBack with an **adjoint matching** scheme that adds **physics-aware, energy-guided corrections** during **post-training**. You can post-train the original FlowBack model using **RDKit** energies or **CHARMM** energies (**recommended**). For CHARMM-based post-training you must have a working installation of **GROMACS** and the desired **CHARMM** force field (e.g., CHARMM27/CHARMM36). **Inference/integration is unchanged** relative to the original FlowBack pipeline.

---

## Installation

### Env setup
    git clone https://github.com/mrjoness/Flow-Back.git
    conda create -n flowback python=3.9
    conda activate flowback
    pip install egnn_pytorch   # installs torch-2.3 + cu12-12; verify CUDA/Torch compatibility
    pip install -c conda-forge openmm sidechainnet
    conda install mdtraj matplotlib pandas
    conda install conda-forge::tqdm

### Quick test
    cd ./scripts
    python eval.py --n_gens 1 --load_dir PDB_example --mask_prior

---

## Inference (FlowBack & FlowBack-Adjoint)

The **integration (inference)** procedure is **identical** for base FlowBack and FlowBack-Adjoint models. Any model (with or without adjoint energy guidance) can be evaluated with the same commands.

Generate 5 samples per CG trace in `./data/PDB_test_CG`:
    
    python eval.py --n_gens 5 --load_dir PDB_example --mask_prior

Compute bonds, clashes, and diversity vs AA references:
    
    python eval.py --n_gens 5 --load_dir PDB_example --retain_AA --check_bonds --check_clash --check_div --mask_prior

Increase diversity by using a noisier initial distribution:
    
    python eval.py --n_gens 5 --load_dir PDB_example --mask_prior --CG_noise 0.005

Backmap a short (10-frame) CG trajectory containing only C\-alpha atoms:
    
    python eval.py --n_gens 3 --load_dir pro_traj_example --mask_prior

---

## Training (base FlowBack)

Download training PDBs and pre-processed features from:
    
    https://zenodo.org/records/13375392

Unzip and move `train_features` into the working directory.

Train protein model with default parameters:
    
    python train.py --system pro --load_path ./train_features/feats_pro_0-1000_all_max-8070.pkl

Train DNA–protein model:
    
    python train.py --system DNApro --load_path ./train_features/feats_DNAPro_DNA-range_10-120_pro-range_10-500.pkl

Re-train with new PDBs:
    
    cd scripts
    python featurize_pro.py --pdb_dir ../train_PDBs/ --save_name pro-train

---

## Post-Training with FlowBack-Adjoint (Energy-Guided)

**Goal**: Refine a pre-trained FlowBack model by incorporating **energy terms** in an adjoint matching objective.

**Energy backends**:
- **RDKit**: lightweight, good for rapid prototyping.
- **CHARMM (recommended)**: biomolecularly realistic. **Requires**:
  - A working **GROMACS** installation.
  - Access to the desired **CHARMM force field** (e.g., CHARMM27/CHARMM36).

During post-training, energies are computed for generated structures and used as additional loss terms to bias the learned distribution toward **physically consistent** ensembles.

### Configuration files

Editable configs live in:
    
    configs/

You can change:
- Energy backend: `rdkit` or `charmm`
- Energy-loss weight in the adjoint objective
- Training hyperparameters (learning rate, batch size, etc.)

### Example usage
    
    python post_train.py --config configs/charmm.yaml

---

## Cite as

    @inproceedings{jones24flowback,
      title={FlowBack: A Flow-matching Approach for Generative Backmapping of Macromolecules},
      author={Jones, Michael and Khanna, Smayan and Ferguson, Andrew},
      booktitle={ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications}
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

