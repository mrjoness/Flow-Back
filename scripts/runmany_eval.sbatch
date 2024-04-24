#!/bin/bash
#SBATCH --job-name=runmany
#SBATCH --output=runmany_DES.out
#SBATCH --partition=gm4-pmext
#SBATCH --account=pi-andrewferguson

#SBATCH --nodes=1 # SET NUM NODES
#SBATCH --gres=gpu:1 # SET NUM GPUS
#SBATCH --ntasks-per-node=1 # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=8  # SET NUM THREADS (cores)
#SBATCH --qos=gm4
#SBATCH --mem=20G              # 50G works for 50p-20d, # 60G works for 80p
#SBATCH --gres=gpu:1

### !!! Make sure running this from midway2 and tenv is activated !!! ### 

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load python
module load cuda

# run eval on multiple paths/ckps 

#/scratch/midway2/mikejones/conda_env/torchcfm_11.8/bin/python eval.py --model_path ../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt --check_clash --check_bonds --check_div --load_dir ../sidechainnet_data/PDB --system 1000-full --n_gens 5 --ckp 10 --CG_noise 0.001

#/scratch/midway2/mikejones/conda_env/torchcfm_11.8/bin/python eval.py --model_path ../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt --check_clash --check_bonds --check_div --load_dir ../sidechainnet_data/PDB --system 1000-full --n_gens 5 --ckp 10 --CG_noise 0.003

#/scratch/midway2/mikejones/conda_env/torchcfm_11.8/bin/python eval.py --model_path ../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt --check_clash --check_bonds --check_div --load_dir ../sidechainnet_data/PDB --system 1000-full --n_gens 5 --ckp 10 --CG_noise 0.005

# run DES 

#/scratch/midway2/mikejones/conda_env/torchcfm_11.8/bin/python eval.py --model_path ../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt --check_clash --check_bonds --check_div --load_dir ../sidechainnet_data/DES --save_dir ../sidechainnet_scores/DES  --system 1000-full --n_gens 5 --ckp 10 --CG_noise 0.001 --stride 10

/scratch/midway2/mikejones/conda_env/torchcfm_11.8/bin/python eval.py --model_path ../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt --check_clash --check_bonds --check_div --load_dir ../sidechainnet_data/DES --save_dir ../sidechainnet_scores/DES  --system 1000-full --n_gens 5 --ckp 10 --CG_noise 0.003 --stride 10

#/scratch/midway2/mikejones/conda_env/torchcfm_11.8/bin/python eval.py --model_path ../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt --check_clash --check_bonds --check_div --load_dir ../sidechainnet_data/DES --save_dir ../sidechainnet_scores/DES  --system 1000-full --n_gens 5 --ckp 10 --CG_noise 0.005 --stride 10

# run CG 

#/scratch/midway2/mikejones/conda_env/torchcfm_11.8/bin/python eval.py --model_path ../jobs/time-batch_adamW_1000-full-fix_L1_m-32_clamp-2.0_attn-0_dim-32_nn-15_depth-6_eps-2001_sigma-0.005_batch-1_CG-noise-0.003_lr-0.001_wdecay-0.0_CGadj--1.0_pos-1_bpack-max_lrdecay-0.0_diff-xt --check_clash --check_bonds --check_div --load_dir ../sidechainnet_data/CG-pro --save_dir ../sidechainnet_scores/CG-pro  --system 1000-full --n_gens 5 --ckp 10 --CG_noise 0.003 --stride 10