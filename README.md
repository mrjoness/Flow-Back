# FlowBack
Generative backmapping of coarse-grained molecular systems using an equivariant graph neural network and a flow-matching objective. Implementation of https://openreview.net/forum?id=mhUasr0j5X.

# Installation
### Env setup instruction
```
git clone https://github.com/mrjoness/Flow-Back.git
conda create -n flowback python=3.9
conda activate flowback 
pip install egnn_pytorch   # automatically installs torch-2.3 + cu12-12 (make sure these are compatible)
pip install -c conda-forge openmm sidechainnet
conda install mdtraj matplotlib pandas
conda install conda-forge::tqdm
```

### Testing:
```
cd ./scripts
python eval.py --n_gens 1 --load_dir PDB_example --mask_prior
```

# Inference

Generates 5 samples of each CG trace in ./data/PDB_test_CG directory
```
python eval.py --n_gens 5 --load_dir PDB_example --mask_prior
```
Generates samples and computes bond, clash, and diversity score with respect to AA references
```
python eval.py --n_gens 5 --load_dir PDB_example --retain_AA --check_bonds --check_clash --check_div  --mask_prior
```
Backmap samples using noisier initial distribution to increase diversity
```
python eval.py --n_gens 5 --load_dir PDB_example --mask_prior --CG_noise 0.005
```
Backmap short (10 frame) CG trajectory containing only C-alphas atoms
```
python eval.py --n_gens 3 --load_dir pro_traj_example --mask_prior
```
Backmap DNA-protein residues --ckp 900 is recommended
```
python eval.py --n_gens 5 --system DNApro --load_dir DNApro_example --model_path ../models/DNAPro_pretrained --ckp 900 --mask_prior --retain_AA
```
Backmapping DNA-protein CG trajectory
```
python eval.py --n_gens 3 --system DNApro --load_dir DNApro_traj_example --model_path ../models/DNAPro_pretrained --ckp 900 --mask_prior
```

# Training

Download training pdbs and pre-processed features from https://zenodo.org/records/13375392.

### Re-train using default parameters

Unzip and move train_features directory in working directory.

Train protein model using default parameters:

```
python train.py --system pro --load_path ./train_features/feats_pro_0-1000_all_max-8070.pkl
```

Train DNA-protein model:

```
python train.py --system DNApro --load_path ./train_features/feats_DNAPro_DNA-range_10-120_pro-range_10-500.pkl'
```

### Re-train with new PDBs

```
cd scripts
python featurize_pro.py --pdb_dir ../train_PDBs/ --save_name pro-train
```


# Cite as
```bibtex
@inproceedings{jones24flowback,
  title={FlowBack: A Flow-matching Approach for Generative Backmapping of Macromolecules},
  author={Jones, Michael and Khanna, Smayan and Ferguson, Andrew},
  booktitle={ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications}
}
```
