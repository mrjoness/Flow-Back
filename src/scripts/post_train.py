"""Fine-tuning script implemented with PyTorch Lightning.

This replaces the original imperative training loop with a
`LightningModule` so that training behaviour matches the previous
implementation while benefitting from the Lightning ecosystem.
"""

import argparse
import glob
import os
import shutil
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Tuple

import mdtraj as md
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.file_config import FLOWBACK_BASE, FLOWBACK_DATA, FLOWBACK_JOBDIR
from src.utils.energy_helpers import ensure_charmm_ff
from src.adjoint import (
    adjoint_matching_loss,
    sigma,
    trajectory_and_adjoint,
)
from src.utils.evaluation import (
    load_features_pro,
    load_model,
    process_pro_aa,
    process_pro_cg,
    split_list,
)
from src.utils.model import ModelWrapper, PostTrainDataset, atom_to_steps


def setup_args(parser: ArgumentParser) -> ArgumentParser:
    """Attach script specific arguments."""

    parser.add_argument(
        "--config",
        type=str,
        default=f"{FLOWBACK_BASE}/configs/post_train.yaml",
        help="Path to config file",
    )
    parser.add_argument("--test", action="store_true", help="For testing purposes")
    parser.add_argument(
        "--compare", action="store_true", help="Just for printing energies"
    )
    return parser


def config_to_args(config: dict) -> argparse.Namespace:
    """Convert config dictionary into argparse Namespace."""

    return argparse.Namespace(**config)


def get_args() -> Tuple[Any, Any]:
    """Parse command line arguments and load YAML configuration."""

    parser = ArgumentParser(description="BoltzmannFlow")
    parser = setup_args(parser=parser)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config_args = config_to_args(config)
    return args, config_args


def _select_timesteps(N, split=0.8, num_samples=10, seed=None):
    """Randomly select timesteps from a range [0, N-1]."""

    if seed is not None:
        np.random.seed(seed)

    split_idx = int(split * N)
    first_80_percent = np.arange(0, split_idx + 1)
    last_20_percent = np.arange(split_idx + 1, N + 1)
    selected_from_first = np.random.choice(
        first_80_percent, size=min(num_samples, len(first_80_percent)), replace=False
    )
    selected_timesteps = np.concatenate([selected_from_first, last_20_percent])
    np.random.shuffle(selected_timesteps)
    return torch.tensor(selected_timesteps)


def posttrain_collate(batch):
    """Custom collate function keeping topology objects intact."""

    res_feats, atom_feats, ca_pos, mask, top = zip(*batch)
    res_feats = torch.stack(res_feats)
    atom_feats = torch.stack(atom_feats)
    ca_pos = torch.stack(ca_pos)
    mask = torch.stack(mask)
    # topologies are returned as a list so we can access the first element
    return res_feats, atom_feats, ca_pos, mask, list(top)


class PostTrainModule(pl.LightningModule):
    """Lightning module implementing the post-training objective."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        lr: float,
        wdecay: float,
        lam: float,
        cg_noise: float,
        num_steps: int,
        ff: str,
        charmm_ff: str,
        int_ff: bool,
        max_grad: float,
        t_flip: float,
        job_dir: str,
        selection_split: float,
        compare: bool = False,
        test: bool = False,
        agb: int = 16
    ) -> None:
        super().__init__()
        self.model = deepcopy(base_model)
        # ensure the fine-tuning model has trainable parameters
        self.model.requires_grad_(True)
        self.model.train()
        self.v_base = base_model
        self.lr = lr
        self.wdecay = wdecay
        self.lam = lam
        self.cg_noise = cg_noise
        self.num_steps = num_steps
        self.ff = ff
        self.charmm_ff = charmm_ff
        self.int_ff = int_ff
        self.max_grad = max_grad
        self.t_flip = t_flip
        self.job_dir = job_dir
        self.selection_split = selection_split
        self.compare = compare
        self.test = test
        self.losses_epoch: list[float] = []
        self.all_losses: list[float] = []
        self.agb = agb

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.wdecay
        )
        scheduler = StepLR(optimizer, step_size=80, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        res_feats, atom_feats, ca_pos, mask, top = batch
        topology = top[0] if isinstance(top, list) else top

        model_wrpd = ModelWrapper(
            model=self.v_base,
            feats=res_feats,
            mask=mask,
            atom_feats=atom_feats,
            ca_pos=ca_pos,
        )
        model_ft_wrpd = ModelWrapper(
            model=self.model,
            feats=res_feats,
            mask=mask,
            atom_feats=atom_feats,
            ca_pos=ca_pos,
        )
        kwargs = {
            "job_dir": self.job_dir,
            "topology": topology,
            "device": self.device,
            "lam": self.lam,
            "n_coords": res_feats.shape[1],
            "cg_noise": self.cg_noise,
            "batch_size": res_feats.shape[0],
            "ca_pos": ca_pos.cpu().numpy(),
            "num_steps": self.num_steps,
            "ff": self.ff,
            "charmm_ff": self.charmm_ff,
            "int_ff": self.int_ff,
            "max_grad": self.max_grad,
            "t_flip": self.t_flip,
            "compare": self.compare
        }

        if self.compare:
            try:
                energy = trajectory_and_adjoint(model_wrpd, model_ft_wrpd, **kwargs)
                _ = _select_timesteps(
                    self.num_steps, self.selection_split
                ).to(self.device)
                with open(f'{self.job_dir}/energies_compare.out', 'a') as f:
                    f.write(str(energy) + '\n') 
            except Exception:
                pass
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        elif self.test:
            traj, a_t = trajectory_and_adjoint(
                model_wrpd, model_ft_wrpd, **kwargs
            )
            select_steps = _select_timesteps(
                self.num_steps, self.selection_split
            ).to(self.device)
            sigma_select = sigma(
                select_steps / self.num_steps, 1 / self.num_steps, self.cg_noise
            )
            max_steps = atom_to_steps(res_feats.shape[1])
            if max_steps < 30:
                step_list = split_list(
                    select_steps, int(np.ceil(len(select_steps) / max_steps))
                )
                sigma_list = split_list(
                    sigma_select, int(np.ceil(len(select_steps) / max_steps))
                )
            else:
                step_list = [select_steps]
                sigma_list = [sigma_select]

            loss_total = torch.tensor(0.0, device=self.device)
            for steps, sigmas in zip(step_list, sigma_list):
                loss_total = loss_total + adjoint_matching_loss(
                    traj,
                    model_ft_wrpd,
                    model_wrpd,
                    a_t,
                    steps,
                    sigmas,
                    **kwargs,
                )
            loss = loss_total
        else:
            try:
                traj, a_t = trajectory_and_adjoint(
                    model_wrpd, model_ft_wrpd, **kwargs
                )
                select_steps = _select_timesteps(
                    self.num_steps, self.selection_split
                ).to(self.device)
                sigma_select = sigma(
                    select_steps / self.num_steps, 1 / self.num_steps, self.cg_noise
                )
                max_steps = atom_to_steps(res_feats.shape[1])
                if max_steps < 30:
                    step_list = split_list(
                        select_steps, int(np.ceil(len(select_steps) / max_steps))
                    )
                    sigma_list = split_list(
                        sigma_select, int(np.ceil(len(select_steps) / max_steps))
                    )
                else:
                    step_list = [select_steps]
                    sigma_list = [sigma_select]
    
                loss_total = torch.tensor(0.0, device=self.device)
                for steps, sigmas in zip(step_list, sigma_list):
                    loss_total = loss_total + adjoint_matching_loss(
                        traj,
                        model_ft_wrpd,
                        model_wrpd,
                        a_t,
                        steps,
                        sigmas,
                        **kwargs,
                    )
                loss = loss_total
            except Exception:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.losses_epoch.append(loss.detach().cpu().item())
        if (self.global_step + 1) % 10 == 0 and not self.compare and not self.test:
            torch.save(
                self.model.state_dict(),
                f"{self.job_dir}/state-{self.global_step + 1}.pth",
            )
            np.save(
                f"{self.job_dir}/losses-epoch-{self.current_epoch}-step-{self.global_step + 1}.npy",
                np.array(self.all_losses + [np.mean(self.losses_epoch)]),
            )

        return loss

    def on_train_epoch_end(self):
        if self.losses_epoch:
            epoch_mean = float(np.mean(self.losses_epoch))
        else:
            epoch_mean = 0.0
        self.all_losses.append(epoch_mean)
        np.save(
            f"{self.job_dir}/losses-epoch-{self.current_epoch}.npy",
            np.array(self.all_losses),
        )
        torch.save(
            self.model.state_dict(), f"{self.job_dir}/state-{self.current_epoch}.pth"
        )
        self.losses_epoch = []


if __name__ == "__main__":
    import random
    import yaml

    args, config_args = get_args()

    seed = 234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_yaml = f"{FLOWBACK_BASE}/configs/pre_train.yaml"

    load_dir = config_args.load_dir
    CG_noise = config_args.CG_noise
    Ca_std = CG_noise
    model_path = config_args.model_path
    ckp = config_args.ckp
    n_epochs = config_args.n_epochs
    stride = config_args.stride
    batch_size = config_args.batch
    mask_prior = config_args.mask_prior
    retain_AA = config_args.retain_AA
    nsteps = config_args.nsteps
    vram = config_args.vram
    lr = config_args.lr
    wdecay = config_args.wdecay
    grad_clip = config_args.grad_clip
    lam = config_args.lam
    n_samples = config_args.n_samples
    save_dir = config_args.save_dir
    eval_every = config_args.evalf
    num_steps = config_args.num_steps
    pdb_list = config_args.pdb_list
    restart = config_args.restart
    selection_split = config_args.selection_split

    MAX_GRAD = getattr(config_args, "max_grad", 5e3)
    t_flip = getattr(config_args, "t_flip", 0.55)
    charmm_ff = getattr(config_args, "charmm_ff", "auto")

    job_dir = f"{FLOWBACK_JOBDIR}/{save_dir}_post"
    os.makedirs(job_dir, exist_ok=True)

    shutil.copyfile(config_yaml, f"{job_dir}/config.yaml")

    ff = getattr(config_args, "ff", "RDKit")
    int_ff = getattr(config_args, "int_ff", False)
    if ff == 'CHARMM':
        ensure_charmm_ff(charmm_ff)
    acc_grad_batch = getattr(config_args, "acc_grad_batch", 1)

    all_dirs = load_dir.split()
    full_trj_list = []
    for dir_ in all_dirs:
        ldir = f"{FLOWBACK_DATA}/{dir_}"
        if retain_AA:
            ldir = process_pro_aa(ldir)
        else:
            ldir = process_pro_cg(ldir)
        if pdb_list == "default":
            full_trj_list.extend(sorted(glob.glob(f"{ldir}/*.pdb")))
        else:
            with open(pdb_list, "r") as f:
                full_trj_list.extend([f"{ldir}/{line[:-1]}.pdb" for line in f.readlines()])

    np.random.shuffle(full_trj_list)
    if args.test:
        full_trj_list = full_trj_list[:100]

    res_list = []
    atom_list = []
    mask_list = []
    top_list = []
    ca_pos = []
    for trj_name in tqdm(full_trj_list):
        trj = md.load(trj_name)
        res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_pro(trj)
        res_list.append(res_ohe)
        atom_list.append(atom_ohe)
        mask_list.append(mask)
        top_list.append(top)
        ca_pos.append(xyz[0, aa_to_cg])

    pt_dataset = PostTrainDataset(res_list, atom_list, ca_pos, mask_list, top_list)
    train_loader = DataLoader(
        pt_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        collate_fn=posttrain_collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_model(model_path, ckp, device)
    base_model.eval()
    base_model.requires_grad_(False)

    module = PostTrainModule(
        base_model=base_model,
        lr=lr,
        wdecay=wdecay,
        lam=lam,
        cg_noise=Ca_std,
        num_steps=num_steps,
        ff=ff,
        charmm_ff=config_args.charmm_ff,
        int_ff=int_ff,
        max_grad=MAX_GRAD,
        t_flip=t_flip,
        job_dir=job_dir,
        selection_split=selection_split,
        compare=args.compare,
        test=args.test,
        agb=acc_grad_batch
    )
    if restart:
        module.model.load_state_dict(
            torch.load(f"{job_dir}/state-{config_args.restart_ckp}.pth")
        )

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        gradient_clip_val=grad_clip,
        accumulate_grad_batches=acc_grad_batch,
        accelerator="auto",
        strategy='ddp_find_unused_parameters_true'
    )
    trainer.fit(module, train_loader)

