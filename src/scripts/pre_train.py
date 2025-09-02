from src.utils.model import *
from src.conditional_flow_matching import ConditionalFlowMatcher
from file_config import FLOWBACK_JOBDIR, FLOWBACK_INPUTS, FLOWBACK_BASE
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from argparse import ArgumentParser
from typing import Tuple, Any
import argparse
import yaml
import numpy as np
import shutil
import os


def setup_args(parser: ArgumentParser) -> ArgumentParser:
    """Add command line arguments."""
    parser.add_argument('--config', type=str, default=f'{FLOWBACK_BASE}/configs/config.yaml',
                        help='Path to config file')
    return parser


def config_to_args(config):
    """Convert a configuration dictionary into an argparse.Namespace."""
    return argparse.Namespace(**config)


def get_args() -> Tuple[Any, Any]:
    """Parse command line arguments and load configuration from YAML file."""
    parser = ArgumentParser(description='BoltzmannFlow')
    parser = setup_args(parser=parser)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config_args = config_to_args(config)
    return args, config_args


class TrainDataset(Dataset):
    """Dataset that draws a fresh prior noise each time it is indexed."""

    def __init__(self, xyz_diff, res_feats, atom_feats, ca_pos, mask, ca_std):
        self.xyz_diff = xyz_diff
        self.res_feats = res_feats
        self.atom_feats = atom_feats
        self.ca_pos = ca_pos
        self.mask = mask
        self.ca_std = ca_std

    def __len__(self):
        return len(self.xyz_diff)

    def __getitem__(self, index):
        x1 = torch.tensor(self.xyz_diff[index], dtype=torch.float32)
        noise = np.random.randn(*self.xyz_diff[index].shape) * self.ca_std
        x0 = torch.tensor(noise, dtype=torch.float32)
        res = torch.tensor(self.res_feats[index], dtype=torch.int)
        atom = torch.tensor(self.atom_feats[index], dtype=torch.int)
        ca = torch.tensor(self.ca_pos[index], dtype=torch.float32)
        mask = torch.tensor(self.mask[index], dtype=torch.bool)
        return x1, x0, res, atom, ca, mask


class PreTrainModule(pl.LightningModule):
    """PyTorch Lightning module for the pre-training loop."""

    def __init__(self, model, fm, train_dataset, xyz_true, load_dict,
                 top_list, masks, config_args, job_dir, valid_idxs):
        super().__init__()
        self.model = model
        self.fm = fm
        self.train_dataset = train_dataset
        self.xyz_true = xyz_true
        self.load_dict = load_dict
        self.top_list = top_list
        self.masks = masks
        self.config = config_args
        self.job_dir = job_dir
        self.valid_idxs = valid_idxs
        self.loss_list = []
        self.start_epoch = int(config_args.checkpoint)
        self._epoch_losses = []
        if config_args.solver == 'euler_ff':
            self.rtp_data, self.lj_data, self.bond_data = get_ff_data()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, drop_last=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr,
                                      weight_decay=self.config.wdecay)
        if self.config.lrdecay > 0.0:
            scheduler = StepLR(optimizer, step_size=1, gamma=self.config.lrdecay)
            return [optimizer], [scheduler]
        return optimizer

    def forward(self, res_feats, x, t, atom_feats, mask, ca_pos):
        return self.model(res_feats, x, time=t, atom_feats=atom_feats, mask=mask, ca_pos=ca_pos)

    def training_step(self, batch, batch_idx):
        x1, x0, res_feats, atom_feats, ca_p, mask = batch
        device = self.device
        x1 = x1.to(device)
        x0 = x0.to(device)
        res_feats = res_feats.to(device)
        atom_feats = atom_feats.to(device)
        ca_p = ca_p.to(device)
        mask = mask.to(device)

        if self.config.batch_pack == 'max':
            time_batch = (self.config.max_atoms // len(res_feats[0])) * self.config.batch
        else:
            time_batch = self.config.batch

        x1 = x1.repeat(time_batch, 1, 1)
        x0 = x0.repeat(time_batch, 1, 1)
        res_feats = res_feats.repeat(time_batch, 1)
        atom_feats = atom_feats.repeat(time_batch, 1)
        mask = mask.repeat(time_batch, 1)

        t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)

        t_pad = t.reshape(-1, *([1] * (xt.dim() - 1)))
        epsilon = torch.randn_like(xt)
        xt_mask = t_pad * x1 + (1 - t_pad) * x0

        sigma_t = self.config.fmsigma
        extended_mask = torch.unsqueeze(mask.int(), -1).repeat(1, 1, 3)
        xt_mask += sigma_t * epsilon * extended_mask

        _, xt_pred = self(res_feats, xt_mask, t, atom_feats, mask, ca_p)

        if self.config.diff_type == 'xt_mask':
            vt = xt_pred - xt_mask
        elif self.config.diff_type == 'xt':
            vt = xt_pred - xt
        elif self.config.diff_type == 'x0':
            vt = xt_pred - x0

        if self.config.loss == 'L2':
            loss = torch.mean((vt - ut) ** 2)
        else:
            loss = torch.mean(torch.abs(vt - ut))

        self._epoch_losses.append(loss.detach())
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=x1.size(0))
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self._epoch_losses).mean()
        self.loss_list.append(avg_loss.detach().cpu().item())
        self._epoch_losses.clear()
        epoch = self.current_epoch + self.start_epoch
        np.save(f'{self.job_dir}/losses-epoch-{epoch}.npy', np.array(self.loss_list))
        torch.save(self.model.state_dict(), f'{self.job_dir}/state-{epoch}.pth')
        if epoch % self.config.evalf == 0 and epoch > 0:
            self.evaluate(epoch)

    def evaluate(self, epoch):
        bf_list = []
        cls_list = []
        for idx in self.valid_idxs:
            xyz_test_real = [self.xyz_true[idx]]
            map_test = [self.load_dict['map'][idx]]
            ca_test_pos = get_ca_pos(xyz_test_real, map_test)
            xyz_test_diff = np.array([xyz_test_real[0] - ca_test_pos[0]])
            try:
                top = self.top_list[idx]
            except:
                top = self.load_dict['top'][idx]
            prior = [np.random.randn(xyz_test_diff.shape[1], xyz_test_diff.shape[2]) * self.config.CG_noise]
            model_wrpd = ModelWrapper(model=self.model,
                                      feats=torch.tensor(np.array([self.load_dict['res'][idx]])).int().to(self.device),
                                      mask=torch.tensor(np.array([self.masks[idx]])).bool().to(self.device),
                                      atom_feats=torch.tensor(np.array([self.load_dict['atom'][idx]])).to(self.device),
                                      ca_pos=torch.tensor(np.array(ca_test_pos)).to(self.device))
            with torch.no_grad():
                if self.config.solver == 'euler':
                    ode_traj = euler_integrator(model_wrpd,
                                                torch.tensor(prior, dtype=torch.float32).to(self.device))
                elif self.config.solver == 'euler_ff':
                    ode_traj = euler_ff_integrator(model_wrpd,
                                        torch.tensor(prior, dtype=torch.float32).to(self.device),
                                        torch.tensor(ca_test_pos).to(self.device),
                                        self.rtp_data, self.lj_data, self.bond_data, top, self.device)
            xyz_gens = ode_traj[-1] + ca_test_pos[0]
            xyz_ref = self.xyz_true[idx]
            trj_gens = md.Trajectory(xyz_gens[:, :top.n_atoms], top)
            trj_ref = md.Trajectory(xyz_ref[:top.n_atoms], top)
            bf_list += [bond_fraction(trj_ref, trj_gen) for trj_gen in trj_gens]
            try:
                cls_list += [clash_res_percent(trj_gen) for trj_gen in trj_gens]
            except:
                print('Failed', [res for res in top.residues])
            np.save(f'{self.job_dir}/ode-{epoch}_f-{idx}.npy', ode_traj)
        np.save(f'{self.job_dir}/bf-{epoch}.npy', bf_list)
        np.save(f'{self.job_dir}/cls-{epoch}.npy', cls_list)


def main():
    args, config_args = get_args()
    config_yaml = args.config
    job_dir = f'{FLOWBACK_JOBDIR}/{config_args.save_dir}'
    os.makedirs(job_dir, exist_ok=True)

    load_path = config_args.load_path
    top_path = config_args.top_path
    if load_path == 'default':
        load_dict = pkl.load(open(f'{FLOWBACK_INPUTS}/train_features/feats_pro_0-1000_all_max-8070.pkl', 'rb'))
        top_list = pkl.load(open(f'{FLOWBACK_INPUTS}/train_features/tops_pro_0-1000_all.pkl', 'rb'))
    else:
        load_dict = pkl.load(open(load_path, 'rb'))
        top_list = pkl.load(open(top_path, 'rb'))

    max_atoms = config_args.max_atoms
    res_dim = config_args.res_dim
    atom_dim = config_args.atom_dim

    xyz_true = load_dict['xyz']
    if xyz_true[0].shape[0] == 1:
        xyz_true = [xyz[0] for xyz in xyz_true]

    train_idxs = np.load(f'{FLOWBACK_INPUTS}/train_features/idxs_train_pro.npy')[:config_args.max_train]
    valid_idxs = np.load(f'{FLOWBACK_INPUTS}/train_features/idxs_valid_pro.npy')[:config_args.max_val]

    params_dict = {
        'depth': config_args.depth,
        'num_nearest_neighbors': config_args.nneigh,
        'dim': config_args.dim,
        'mdim': config_args.mdim,
        'max_atoms': max_atoms,
        'res_dim': res_dim,
        'atom_dim': atom_dim,
        'sym': config_args.sym,
        'pos_cos': config_args.pos_cos,
        'seq_feats': config_args.seq_feats,
        'seq_decay': config_args.seq_decay,
        'act': config_args.act,
        'grad_clip': config_args.grad_clip,
        'train_idxs': train_idxs,
        'valid_idxs': valid_idxs
    }
    pkl.dump(params_dict, open(f'{job_dir}/params.pkl', 'wb'))
    shutil.copyfile(config_yaml, f'{job_dir}/config.yaml')

    masks = []
    for res, m_idxs in zip(load_dict['res'], load_dict['mask']):
        mask = np.ones(len(res))
        mask[m_idxs] = 0
        masks.append(mask)

    if config_args.pos == 1:
        pos_emb = max_atoms
    else:
        pos_emb = None
    pos_cos = None if config_args.pos_cos < 0.001 else config_args.pos_cos

    model = EGNN_Network_time(
        num_tokens=res_dim,
        num_positions=pos_emb,
        dim=config_args.dim,
        depth=config_args.depth,
        num_nearest_neighbors=config_args.nneigh,
        global_linear_attn_every=config_args.attnevery,
        coor_weights_clamp_value=config_args.clamp,
        m_dim=config_args.mdim,
        fourier_features=4,
        time_dim=0,
        res_dim=res_dim,
        atom_dim=atom_dim,
        sym=config_args.sym,
        emb_cos_scale=pos_cos,
        seq_feats=config_args.seq_feats,
        seq_decay=config_args.seq_decay,
        act=config_args.act
    )

    cur_epoch = int(config_args.checkpoint)
    if hasattr(config_args, 'model_path') and config_args.model_path is not None:
        model.load_state_dict(torch.load(config_args.model_path, map_location=config_args.device))
    elif config_args.checkpoint > 0:
        model.load_state_dict(torch.load(f'{job_dir}/state-{config_args.checkpoint}.pth',
                                         map_location=config_args.device))

    fm = ConditionalFlowMatcher(sigma=config_args.fmsigma)

    ca_pos_all = get_ca_pos(xyz_true, load_dict['map'])
    xyz_diff_train = [xyz_true[i] - ca_pos_all[i] for i in train_idxs]
    res_train = [load_dict['res'][i] for i in train_idxs]
    atom_train = [load_dict['atom'][i] for i in train_idxs]
    ca_train = [ca_pos_all[i] for i in train_idxs]
    mask_train = [masks[i] for i in train_idxs]

    train_dataset = TrainDataset(xyz_diff_train, res_train, atom_train, ca_train,
                                 mask_train, config_args.CG_noise)

    module = PreTrainModule(model, fm, train_dataset, xyz_true, load_dict,
                            top_list, masks, config_args, job_dir, valid_idxs)

    trainer = pl.Trainer(max_epochs=config_args.eps - cur_epoch,
                         gradient_clip_val=config_args.grad_clip,
                         default_root_dir=job_dir)
    trainer.fit(module)


if __name__ == '__main__':
    main()

