from utils import *
from conditional_flow_matching import ConditionalFlowMatcher
import argparse
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils as nn_utils

parser = argparse.ArgumentParser()
parser.add_argument('--fmsigma', default=0.005, type=float, help='Epsilon during FM training')
parser.add_argument('--batch_pack', default='max', type=str, help='Whether to keep uniform batch or maximize (max) based on size')
parser.add_argument('--batch', default=1, type=int, help='Batch size over time (set automatically if using max batch_pack)')
parser.add_argument('--CG_noise', default=0.003, type=float, help='Std of noise on initial CG positions')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--eps', default=101, type=int, help='Number of training epochs')
parser.add_argument('--evalf', default=5, type=int, help='Frequency to evaluate on test data')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--wdecay', default=0.0, type=float, help='Weight decay')
parser.add_argument('--lrdecay', default=0.0, type=float, help='Learning rate decay')
parser.add_argument('--dim', default=32, type=int, help='Embedding and feature dim at each node')
parser.add_argument('--depth', default=6, type=int, help='Number of EGNN layers')
parser.add_argument('--nneigh', default=15, type=int, help='Max number of neighbors')
parser.add_argument('--loss', default='L1', type=str, help='How to calculate loss')
parser.add_argument('--mdim', default=32, type=int, help='Dimension of hidden model in EGNN')
parser.add_argument('--clamp', default=2., type=float, help='Dimension of hidden model in EGNN')
parser.add_argument('--attnevery', default=0, type=int, help='Max number of neighbors')
parser.add_argument('--system', default='pro', type=str, help='Dataset to train on')
parser.add_argument('--CGadj', default=0.0, type=float, help='Whether to load a CG adjacent matrix')
parser.add_argument('--pos', default=1, type=int, help='Set to 1 is using positional encoding')
parser.add_argument('--solver', default='euler', type=str, help='Type of solver to use (adaptive by default)')
parser.add_argument('--diff_type', default='xt', type=str, help='Find vt by subtracting noised or unnoised')
parser.add_argument('--load_path', default='default', type=str, help='Where to load structures')
parser.add_argument('--top_path', default='default', type=str, help='Where to load topologies (for validation set)')

# add for cross-symmetry comparison
parser.add_argument('--sym', default='e3', type=str, help='Type of group symmetry')
parser.add_argument('--sym_rep', default=1, type=int, help='Reps of different symmetries')
parser.add_argument('--mask_prior', action='store_true', help='Do not noise the CG atoms')
parser.add_argument('--pos_cos', default=0., type=float, help='Scale of sin/cos embedding')
parser.add_argument('--seq_feats', default=0, type=int, help='Number of relative sequence distance features to include')
parser.add_argument('--seq_decay', default=100., type=float, help='Exp decay constant on sig feats')
parser.add_argument('--act', default='SiLU', type=str, help='MLP activation')
parser.add_argument('--grad_clip', default=0.0, type=float, help='Clip exploding grads')

args = parser.parse_args()

device = args.device
sigma = args.fmsigma
batch_size = args.batch
Ca_std = args.CG_noise
n_epochs = args.eps
eval_every = args.evalf
lr = args.lr
wdecay = args.wdecay
lrdecay = args.lrdecay
depth = args.depth
num_nearest_neighbors = args.nneigh
dim = args.dim
loss_type = args.loss
mdim = args.mdim
clamp = args.clamp
attnevery = args.attnevery
CGadj = args.CGadj
system = args.system      
pos = args.pos
solver = args.solver
batch_pack = args.batch_pack
diff_type = args.diff_type
load_path = args.load_path
top_path = args.top_path

# MJ -- add for testing
sym = args.sym
sym_rep = args.sym_rep
mask_prior = args.mask_prior
pos_cos = args.pos_cos
seq_feats = args.seq_feats
seq_decay = args.seq_decay
act = args.act
grad_clip = args.grad_clip

max_train = 100_000
max_val = 100

if mask_prior: msk_txt = '_mskp-inf'
else: msk_txt = ''

# MJ -- added name text since last push
#new_txt = f'{sym}-{sym_rep}{msk_txt}_cos-emb-{pos_cos}'
new_txt = f'{sym}-{sym_rep}_seq-feats-{seq_feats}_seq-decay-{int(seq_decay)}_act-{act}_clip-{grad_clip}'

job_dir = f'./jobs/{system}_{new_txt}_{loss_type}_m-{mdim}_dim-{dim}_nn-{num_nearest_neighbors}_depth-{depth}_eps-{n_epochs}_sigma-{sigma}_CG-noise-{Ca_std}_lr-{lr}'
os.makedirs(job_dir, exist_ok=True)

# load different systems with max_atoms and encoding dim to ensure features will fit

if system == 'pro':
    if load_path == 'default':
        load_dict = pkl.load(open('./train_features/feats_pro_0-1000_all_max-8070.pkl', 'rb')) 
        top_list = pkl.load(open('./train_features/tops_pro_0-1000_all.pkl', 'rb'))
    else:
        load_dict = pkl.load(open(load_path, 'rb')) 
        top_list= pkl.load(open(top_path, 'rb')) 
    
    # standard for 20-residue proteins up to 1000 residues
    max_atoms = 8070  # max atoms in training set
    res_dim = 21
    atom_dim = 37
    
    # load idxs of training and validation pdbs (features)
    train_idxs = np.load(f'./train_features/idxs_train_pro.npy')[:max_train]
    valid_idxs = np.load(f'./train_features/idxs_valid_pro.npy')[:max_val]
    
    #print('train_idx', train_idxs[:10], train_idxs[-10:])
    #print('valid_idxs', valid_idxs)
    
elif system == 'DNApro':
    if load_path == 'default':
        load_dict = pkl.load(open('./train_features/feats_DNAPro_DNA-range_10-120_pro-range_10-500.pkl', 'rb'))
    else:
        load_dict = pkl.load(open(load_path, 'rb')) 
        
    max_atoms = 6299
    res_dim = 25
    atom_dim = 68    # make sure to fit all pro + dna atom types

    # obtained from mmseqs on DNA and pro sequences
    train_idxs = np.load(f'./train_features/idxs_train_DNAPro.npy')[:]
    valid_idxs = np.load(f'./train_features/idxs_valid_DNAPro.npy')

# save hyperparams to pkl to reload model
params_dict = { 'depth': depth,
                'num_nearest_neighbors': num_nearest_neighbors,
                'dim': dim, 
                'mdim': mdim,
                'max_atoms': max_atoms,
                'res_dim': res_dim,
                'atom_dim': atom_dim,
                'sym':sym,
                'pos_cos': pos_cos,
                'seq_feats':seq_feats,
                'seq_decay':seq_decay,
                'act':act,
                'grad_clip':grad_clip
                }

pkl.dump(params_dict, open(f'{job_dir}/params.pkl', 'wb'))
    
# reformat CG mask
masks = []
for res, m_idxs in zip(load_dict['res'], load_dict['mask']):
    mask = np.ones(len(res))
    mask[m_idxs] = 0
    masks.append(mask) 

# whether or not to include a positional embedding
if pos==1:
    pos_emb = max_atoms
elif pos==0:
    pos_emb = None

# whether or no to include sin/cos pos embedding 
if pos_cos < 0.001:
    pos_cos = None

model = EGNN_Network_time(
    num_tokens = res_dim,
    num_positions = pos_emb,
    dim = dim,               
    depth = depth,
    num_nearest_neighbors = num_nearest_neighbors,
    global_linear_attn_every = attnevery,
    coor_weights_clamp_value = clamp,  
    m_dim=mdim,
    fourier_features = 4, 
    time_dim=0,
    res_dim=res_dim,
    atom_dim=atom_dim,
    sym=sym,
    emb_cos_scale=pos_cos,
    seq_feats=seq_feats,
    seq_decay=seq_decay,
    act=act
).to(device)
print('params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# should be able to remove cFM here
FM = ConditionalFlowMatcher(sigma=sigma)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay)
if lrdecay > 0.0:
    scheduler = StepLR(optimizer, step_size=1, gamma=lrdecay)

# set xyz_tru directly
xyz_true = load_dict['xyz']

# fix xyz for dna trajs
if xyz_true[0].shape[0] == 1:
    xyz_true = [xyz[0] for xyz in xyz_true]

loss_list = []
for epoch in range(n_epochs):
        
    # ensures using new noise profile at each epoch
    if mask_prior:
        xyz_prior = get_prior_mask(xyz_true, load_dict['map'], scale=Ca_std, masks=masks)
    else:
        xyz_prior = get_prior_mix(xyz_true, load_dict['map'], scale=Ca_std) # default

    train_dataset = FeatureDataset([xyz_true[i] for i in train_idxs], [xyz_prior[i] for i in train_idxs], 
                                   [load_dict['res'][i] for i in train_idxs], [load_dict['atom'][i] for i in train_idxs], 
                                   [load_dict['res'][i] for i in train_idxs], [masks[i] for i in train_idxs])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    mean_loss = []
    
    for i, (x1, x0, res_feats, atom_feats, adj_mat, mask) in tqdm(enumerate(train_loader)):
        
        optimizer.zero_grad()
        x1 = x1.to(device)
        x0 = x0.to(device)
        res_feats = res_feats.to(device)
        atom_feats = atom_feats.to(device)
        mask = mask.to(device)
        
        # maximize batch size based on molecule size
        if batch_pack == 'max':
            time_batch = (max_atoms // len(res_feats[0])) * batch_size
        elif batch_pack == 'uniform':
            time_batch = batch_size
           
        # repeat values over time batch
        x1 = x1.repeat(time_batch, 1, 1)
        x0 = x0.repeat(time_batch, 1, 1)
        res_feats = res_feats.repeat(time_batch, 1)
        atom_feats = atom_feats.repeat(time_batch, 1)
        mask = mask.repeat(time_batch, 1)

        # replace with FM code
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        
        t_pad = t.reshape(-1, *([1] * (xt.dim() - 1)))
        epsilon = torch.randn_like(xt)
        xt_mask =  t_pad * x1 + (1 - t_pad) * x0
        
        # calculate sigma_t as in stochastic interpolants
        sigma_t = sigma
        
        # only add noise to unmasked values
        extended_mask = torch.unsqueeze(mask.int(), -1)
        extended_mask = torch.repeat_interleave(extended_mask, 3, dim=-1)
        xt_mask += sigma_t * epsilon * extended_mask
        
        # pred the structure
        _, xt_pred = model(res_feats, xt_mask, time=t, atom_feats=atom_feats, mask = mask)

        # variant of FM objective, xt usually works best
        if diff_type == 'xt_mask':
            vt = xt_pred - xt_mask
        elif diff_type == 'xt':
            vt = xt_pred - xt
        elif diff_type == 'x0':
            vt = xt_pred - x0
            
        if loss_type == 'L2':
            loss = torch.mean((vt - ut) ** 2)
        elif loss_type == 'L1':
            loss = torch.mean(torch.abs(vt - ut))
        
        loss.backward()

        # add clipping
        if grad_clip > 0.001:
            nn_utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        mean_loss.append(loss.item())
        
    print(epoch, np.mean(mean_loss))
    loss_list.append(np.mean(mean_loss))
    
    # update lr scheduler if included
    if lrdecay > 0.0:
        scheduler.step()
    
    # get bond quality (and clash) every N epochs
    if epoch%eval_every==0 and epoch>0:
        
        # can iterate over this and test one at a time
        n_gens = 1    
        bf_list = []
        cls_list = []
        
        for idx in valid_idxs:

            xyz_test_real = [xyz_true[idx]]
            map_test = [load_dict['map'][idx]]

            # when using mixed batch needs to be array
            xyz_test_prior = get_prior_mix(xyz_test_real, map_test, scale=Ca_std)

            if mask_prior:
                mask_test = np.ones(len(xyz_test_real[0]))
                mask_test[load_dict['mask'][idx]] = 0
                xyz_test_prior = get_prior_mask(xyz_test_real, map_test, scale=Ca_std, masks=[mask_test])
            else:
                xyz_test_prior = get_prior_mix(xyz_test_real, map_test, scale=Ca_std) # default

            model_wrpd = ModelWrapper(model=model, 
                              feats=torch.tensor(np.array([load_dict['res'][idx]])).int().to(device), 
                              mask=torch.tensor(np.array([masks[idx]])).bool().to(device).to(device), 
                              atom_feats=torch.tensor(np.array([load_dict['atom'][idx]])).to(device), 
                              adj_mat=None)
            
            # adaptive solver (requires torchdyn)
            if solver == 'adapt':
                n_ode_steps = 2  
                tol = 3e-5   
                node = NeuralODE(model_wrpd, solver="dopri5", sensitivity="adjoint", atol=tol, rtol=tol) 
                with torch.no_grad():
                    ode_traj = node.trajectory(
                        torch.tensor(xyz_test_prior, dtype=torch.float32).to(device),
                        t_span=torch.linspace(0, 1, n_ode_steps+1).to(device),
                    )
                ode_traj = ode_traj.cpu().numpy()
                
            elif solver == 'euler':
                with torch.no_grad():

                    # accounts for different diff types
                    ode_traj = euler_integrator(model_wrpd, torch.tensor(xyz_test_prior,
                                                                    dtype=torch.float32).to(device))
                                                                    #diff_type=diff_type)
                       
            # assume we're working with one structure at a time
            xyz_gens = ode_traj[-1]
            xyz_ref = xyz_true[idx]
            
            # if top_list not provided, load from main dict
            try:
                top = top_list[idx]
            except:
                top = load_dict['top'][idx]
            
            print(xyz_gens.shape, xyz_ref.shape, top.n_atoms)
            
            # need n_atoms to account for pro-dna case
            trj_gens = md.Trajectory(xyz_gens[:, :top.n_atoms], top)
            trj_ref = md.Trajectory(xyz_ref[:top.n_atoms], top)
            bf_list += [bond_fraction(trj_ref, trj_gen) for trj_gen in trj_gens]
            
            # only run this clash calculation for proteins
            if system == 'pro':
                try: cls_list += [clash_res_percent(trj_gen) for trj_gen in trj_gens]
                except: print('Failed', [res for res in top.residues])
            
            np.save(f'{job_dir}/ode-{epoch}_f-{idx}.npy', ode_traj)
                
        np.save(f'{job_dir}/bf-{epoch}.npy', bf_list)
        np.save(f'{job_dir}/cls-{epoch}.npy', cls_list)
        
        # save ode outputs for visualization
        torch.save(model.state_dict(), f'{job_dir}/state-{epoch}.pth')
        
  
   

