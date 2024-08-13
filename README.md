# FlowBack
Generative backmapping of coarse-grained molecular systems using an equivariant graph neural network and a flow-matching objective. Implementation of https://openreview.net/forum?id=mhUasr0j5X.

# Installation

## Env setup instruction

```
git clone https://github.com/mrjoness/Flow-Back.git
conda create -n flowback python=3.9
conda activate flowback 
pip install egnn_pytorch   # automatically installs torch-2.3 + cu12-12 (make sure these are compatible)
pip install -c conda-forge openmm sidechainnet
conda install mdtraj matplotlib pandas
conda install conda-forge::tqdm
```

## Test with the following
```
cd ./scripts
python eval.py --n_gens 3 --load_dir PDB_test_CG --mask_prior
```

# Cite as
```bibtex
@inproceedings{jones24flowback,
  title={FlowBack: A Flow-matching Approach for Generative Backmapping of Macromolecules},
  author={Jones, Michael and Khanna, Smayan and Ferguson, Andrew},
  booktitle={ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications}
}
```
