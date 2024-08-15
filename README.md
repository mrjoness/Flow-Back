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
Backmap DNA-protein residues --ckp 750 is recommended
```
eval.py --n_gens 5 --system DNApro --load_dir DNApro_example --model_path ../models/DNAPro_pretrained --ckp 750 --mask_prior --retain_AA --check_bonds 
```
Backmapping DNA-protein CG trajectory
```
TODO
```

# Cite as
```bibtex
@inproceedings{jones24flowback,
  title={FlowBack: A Flow-matching Approach for Generative Backmapping of Macromolecules},
  author={Jones, Michael and Khanna, Smayan and Ferguson, Andrew},
  booktitle={ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications}
}
```
