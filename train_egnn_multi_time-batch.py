from utils import *
import argparse
import time
import glob
import pickle as pkl
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser()
parser.add_argument('--fmsigma', default=0.003, type=float, help='Epsilon during FM training')
parser.add_argument('--batch', default=64, type=int, help='Batch size over time')
parser.add_argument('--CG_noise', default=0.003, type=float, help='Std of noise on initial CG positions')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')

parser.add_argument('--eps', default=10001, type=int, help='Number of training epochs')
parser.add_argument('--evalf', default=1000, type=int, help='Frequency to evaluate on test data')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--wdecay', default=0.0, type=float, help='Weight decay')
parser.add_argument('--lrdecay', default=0.0, type=float, help='Learning rate decay')

parser.add_argument('--dim', default=32, type=int, help='Embedding and feature dim at each node')
parser.add_argument('--depth', default=6, type=int, help='Number of EGNN layers')
parser.add_argument('--nneigh', default=15, type=int, help='Max number of neighbors')
parser.add_argument('--loss', default='L1', type=str, help='How to calculate loss')

parser.add_argument('--mdim', default=16, type=int, help='Dimension of hidden model in EGNN')
parser.add_argument('--clamp', default=2., type=float, help='Dimension of hidden model in EGNN')
parser.add_argument('--attnevery', default=0, type=int, help='Max number of neighbors')

parser.add_argument('--system', default='30max', type=str, help='Dataset to train on')
parser.add_argument('--CGadj', default=0.0, type=float, help='Whether to load a CG adjacent matrix')
parser.add_argument('--pos', default=1, type=int, help='Set to 1 is using positional encoding')
parser.add_argument('--solver', default='euler', type=str, help='Type of solver to use (adaptive by default)')
parser.add_argument('--batch_pack', default='uniform', type=str, help='Whether to keep uniform batch or maximize based on size')
parser.add_argument('--diff_type', default='xt', type=str, help='Find vt by subtracting noised or unnoised')
parser.add_argument('--seed', default=13, type=int, help='Random seed for mmseqs clustering used in DNA-pro training')

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
seed = args.seed

job_dir = f'./jobs/{system}_{loss_type}_m-{mdim}_clamp-{clamp}_attn-{attnevery}_dim-{dim}_nn-{num_nearest_neighbors}_depth-{depth}_eps-{n_epochs}_sigma-{sigma}_batch-{batch_size}_CG-noise-{Ca_std}_lr-{lr}_wdecay-{wdecay}_CGadj-{CGadj}_pos-{pos}_bpack-{batch_pack}_lrdecay-{lrdecay}_diff-{diff_type}'

if seed is not None:
    job_dir += f'_seed-{seed}'

try: os.mkdir(job_dir)
except: pass

# load different systems with max_atoms and encoding dim to ensure features will fit
      
if system == '30max':
    load_dict = pkl.load(open('./save_dict.pkl', 'rb'))
    max_atoms = 270
    n_test = 100
    res_dim = 21
    atom_dim = 5
    
if system == '30max_skip2':
    load_dict = pkl.load(open('./features/30max_skip2.pkl', 'rb'))
    max_atoms = 270
    n_test = 100
    res_dim = 21
    atom_dim = 5
    
if system == '30max_skip3':
    load_dict = pkl.load(open('./features/30max_skip3.pkl', 'rb'))
    max_atoms = 270
    n_test = 100
    res_dim = 21
    atom_dim = 5
    
elif system == '100max':
    load_dict = pkl.load(open('./save_dict-100res.pkl', 'rb'))
    max_atoms = 847   
    n_test = 100
    res_dim = 21
    atom_dim = 5
    
elif system == '200max':
    load_dict = pkl.load(open('./features/pro_200res.pkl', 'rb'))
    max_atoms = 1662   
    n_test = 100
    res_dim = 21
    atom_dim = 5
    
elif system == '200maxaa':
    load_dict = pkl.load(open('./features/pro_200res_aa.pkl', 'rb'))
    max_atoms = 1662   
    n_test = 100
    res_dim = 21
    atom_dim = 37

elif system == '200_40k':
    load_dict = pkl.load(open('./features/pro_n-39390_max-1745_0-200res.pkl', 'rb'))
    max_atoms = 1745   
    n_test = 100
    res_dim = 21
    atom_dim = 37
    
elif system == '500-600':
    load_dict = pkl.load(open('./features/pro-big_500-600res.pkl', 'rb'))
    max_atoms = 4900   
    n_test = 0
    res_dim = 21
    atom_dim = 5

elif system == '600max':
    load_dict = pkl.load(open('./features/pro-10k_0-600res.pkl', 'rb'))
    max_atoms = 5000 
    n_test = 100
    res_dim = 21
    atom_dim = 37
    
elif system == '600all':
    load_dict = pkl.load(open('./features/pro_n-30000_max-4945_0-600res.pkl', 'rb'))
    max_atoms = 4945
    n_test = 100
    res_dim = 21
    atom_dim = 37
    
# all < 600 trajs combined -- load tops seperately
elif system == '600comb':
    load_dict = pkl.load(open('./features/pro-comb_0-600_N-60000.pkl', 'rb'))
    top_list = pkl.load(open('./features/pro-5k_0-600_top_val.pkl', 'rb'))
    max_atoms = 4945
    n_test = 100
    res_dim = 21
    atom_dim = 37
    
elif system == '1000-bins':
    load_dict = pkl.load(open('./features/0-1000_bins.pkl', 'rb')) 
    max_atoms = 9000
    n_test = 1     # change back to 100
    res_dim = 21
    atom_dim = 37  # make sure to fit all pro + dna atom types
    
elif system == '1000-full':
    load_dict = pkl.load(open('./features/pro_0-1000_all_max-8070.pkl', 'rb')) 
    top_list = pkl.load(open('./features/pro-org_0-1000_all_tops.pkl', 'rb'))  # need to regen this
    max_atoms = 8070
    n_test = 100     # change back to 100
    res_dim = 21
    atom_dim = 37  # make sure to fit all pro + dna atom types
    
elif system == '1000-full-no-d':
    load_dict = pkl.load(open('./features/pro-no-d_0-1000_all_max-8070.pkl', 'rb')) 
    top_list = pkl.load(open('./features/pro_0-1000_all_tops.pkl', 'rb'))  
    max_atoms = 8070
    n_test = 100     # change back to 100
    res_dim = 21
    atom_dim = 37  # make sure to fit all pro + dna atom types
    
elif system == 'dna-pro':
    #load_dict = pkl.load(open('./features/dna-pro_kmeans-400_pdbs-1000.pkl', 'rb'))
    load_dict = pkl.load(open('./features/dna-pro_kmeans-400_pdbs-200_samples-2171.pkl', 'rb')) 
    max_atoms = 569
    n_test = 100
    res_dim = 25
    atom_dim = 9
    
elif system == 'dna-pro-inter':
    load_dict = pkl.load(open('./features/dna-pro_interface-select-20p-10d_pdbs-200_samples-1865.pkl', 'rb')) 
    max_atoms = 425
    n_test = 100
    res_dim = 25
    atom_dim = 9
    
elif system == 'dna-pro-inter-500':
    load_dict = pkl.load(open('./features/dna-pro_interface-select-20p-10d_pdbs-500_samples-4707.pkl', 'rb')) 
    max_atoms = 438
    n_test = 100
    res_dim = 25
    atom_dim = 9
    
elif system == 'dna-pro-inter-50p-15d':
    load_dict = pkl.load(open('./features/dna-pro_interface-select-50p-15d_pdbs-500_samples-4375.pkl', 'rb')) 
    max_atoms = 812
    n_test = 100
    res_dim = 25
    atom_dim = 9

elif system == 'dna-pro-inter-50p-20d':
    load_dict = pkl.load(open('./features/dna-pro_interface-select-50p-20d_pdbs-1000_samples-8166_max-945.pkl', 'rb')) 
    max_atoms = 945
    n_test = 100
    res_dim = 25
    atom_dim = 9
    
elif system == 'dna-pro-AA-50-15d':
    load_dict = pkl.load(open('./features/dna-pro-AA_interface-setorder-50p-15d_pdbs-500_samples-4159.pkl', 'rb')) 
    max_atoms = 812
    n_test = 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-pro-AA-80-20d':
    load_dict = pkl.load(open('./features/dna-pro-AA_interface-setorder-80p-20d_pdbs-500_samples-3655.pkl', 'rb')) 
    max_atoms = 1202
    n_test = 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-pro-100-500':
    load_dict = pkl.load(open('./features/dna-range_10-100_pro-range_10-500_samples-1836_max-6107.pkl', 'rb')) 
    max_atoms = 6107
    n_test = 100  # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-pro-no-rev-100-500':
    load_dict = pkl.load(open('./features/dna-range-no-rev_10-100_pro-range_10-500_samples-1178_max-6006.pkl', 'rb')) 
    max_atoms = 6107
    n_test = 100  # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-range-no-rev-all-frames-100-600':
    load_dict = pkl.load(open('./features/dna-range-no-rev-all-frames_10-100_pro-range_10-600_samples-1309_max-7048.pkl', 'rb')) 
    max_atoms = 7048
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
# pdbs are reformatted before including trajs
elif system == 'dna-range-rev-reformat_100-500':
    load_dict = pkl.load(open('./features/dna-range-rev-reformat_10-100_pro-range_10-500_samples-1778_max-6107.pkl', 'rb')) 
    max_atoms = 6107
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
# exclude 100 sequences based on potentially bad ordering
elif system == 'dna-range-rev-reformat-matching_100-500':
    load_dict = pkl.load(open('./features/dna-range-rev-reformat-matching_10-100_pro-range_10-500_samples-1676_max-6107.pkl', 'rb')) 
    max_atoms = 6107
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
# exclude 100 sequences based on potentially bad ordering
elif system == 'dna-range-rev-reformat-matching_150-600':
    load_dict = pkl.load(open('./features/dna-range-rev-reformat-matching_10-150_pro-range_10-600_samples-1912_max-7637.pkl', 'rb')) 
    max_atoms = 7637
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-range-rev-reformat-P035_100-500':
    load_dict = pkl.load(open('./features/dna-range-rev-reformat-P035_10-100_pro-range_10-500_samples-1664_max-6110.pkl', 'rb'))
    max_atoms = 6110
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-range-rev-reformat-P035_150-600':
    load_dict = pkl.load(open('./features/dna-range-rev-reformat-P035_10-150_pro-range_10-600_samples-1898_max-7640.pkl', 'rb'))
    max_atoms = 7640
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68     # make sure to fit all pro + dna atom types
    
elif system == 'dna-range-rev-reformat-P035-remap-psb-fix_100-500':
    load_dict = pkl.load(open('./features/dna-range-rev-reformat-P035-remap-psb_10-100_pro-range_10-500_samples-1656_max-6107.pkl', 'rb'))
    max_atoms = 6107
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-clean-mis-0_120-500':
    load_dict = pkl.load(open('./features/dna-clean-mis-0_10-120_pro-range_10-500_samples-1544_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-clean-mis-3_120-500':
    load_dict = pkl.load(open('./features/dna-clean-mis-3_10-120_pro-range_10-500_samples-1720_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-clean-original-mis-0_120-500':
    load_dict = pkl.load(open('./features/dna-clean-original-mis-0_10-120_pro-range_10-500_samples-1544_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-clean-original-mis-0_100-500':
    load_dict = pkl.load(open('./features/dna-clean-original-mis-0_10-100_pro-range_10-500_samples-1533_max-6107.pkl', 'rb'))
    max_atoms = 6107
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-clean-fix-mis-0_120-500':
    load_dict = pkl.load(open('./features/dna-clean-fix-mis-0_10-120_pro-range_10-500_samples-1544_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
elif system == 'dna-clean-fix-mis-0_120-500_mmseqs':
    load_dict = pkl.load(open('./features/dna-clean-fix-mis-0_10-120_pro-range_10-500_samples-1544_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
    train_idxs = np.load('./mmseqs/dna-clean-mis-0_10-120_pro-range_10-500_samples-1544_max-6299_train_seed-10.npy')
    test_idxs = np.load('./mmseqs/dna-clean-mis-0_10-120_pro-range_10-500_samples-1544_max-6299_test_seed-10.npy')
    
elif system == 'dna-clean-fix-mis-0_120-500_mmseqs-reidx':
    load_dict = pkl.load(open('./features/dna-clean-fix-mis-0_10-120_pro-range_10-500_samples-1544_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68  # make sure to fit all pro + dna atom types
    
    train_idxs = np.load('./mmseqs/dna-clean-mis-0_10-120_pro-range_10-500_samples-1544_max-6299_train_seed-2.npy')
    test_idxs = np.load('./mmseqs/dna-clean-mis-0_10-120_pro-range_10-500_samples-1544_max-6299_valid_seed-2.npy')
    
elif system == 'dna-clean-no-rev-mis-0_120-500_mmseqs-reidx':
    load_dict = pkl.load(open('./features/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68    # make sure to fit all pro + dna atom types
    
    train_idxs = np.load(f'./mmseqs/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299_train_seed-{seed}.npy')
    test_idxs = np.load(f'./mmseqs/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299_test_seed-{seed}.npy')
    
elif system == 'dna-clean-no-rev-mis-0_120-500_mmseqs-reidx-valid':
    load_dict = pkl.load(open('./features/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299.pkl', 'rb'))
    max_atoms = 6299
    n_test = 100      # change back to 100
    res_dim = 25
    atom_dim = 68    # make sure to fit all pro + dna atom types
    
    train_idxs = np.load(f'./mmseqs/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299_train_seed-{seed}.npy')
    test_idxs = np.load(f'./mmseqs/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299_valid_seed-{seed}.npy')
    
elif system == 'dna-allatom-0605_120-500_mmseqs-reidx-valid':
    load_dict = pkl.load(open('./features/dna-allatom-0605-0_10-120_pro-range_10-500_samples-1547_max-6299.pkl', 'rb'))
    n_test = 100
    max_atoms = 6299
    res_dim = 25
    atom_dim = 68    # make sure to fit all pro + dna atom types
    
    train_idxs = np.load(f'./mmseqs/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299_train_seed-{seed}.npy')
    test_idxs = np.load(f'./mmseqs/dna-clean-no-rev-mis-0_10-120_pro-range_10-500_samples-1547_max-6299_valid_seed-{seed}.npy')

if 'mmseqs' not in system:
    n_trajs = len(load_dict['xyz'])
    train_idxs = np.arange(n_test, n_trajs)  # change this back to n_trajs
    test_idxs = np.arange(n_test) 

# preprocess mask only -- could just save it in this format?
masks = []
for res, m_idxs in zip(load_dict['res'], load_dict['mask']):
    mask = np.ones(len(res))
    mask[m_idxs] = 0
    masks.append(mask) 

# whether or not to include a positional embedding (else should be unordered sets)
if pos==1:
    pos_emb = max_atoms
elif pos==0:
    pos_emb = None

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
).to(device)

FM = ConditionalFlowMatcher(sigma=sigma)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay)

if lrdecay > 0.0:
    scheduler = StepLR(optimizer, step_size=1, gamma=lrdecay)

loss_list = []

# set xyz_tru directly
xyz_true = load_dict['xyz']

# fix xyz for dna trajs
if xyz_true[0].shape[0] == 1:
    xyz_true = [xyz[0] for xyz in xyz_true]

def get_prior_mix(xyz, aa_to_cg, mask_idxs=None, scale=1.0, frames=None):
    '''Normally distribute around respective Ca center of mass'''
    
    # set center of distribution to each CA and use uniform scale
    xyz_prior = []
    
    for xyz_ref, map_ref in zip(xyz, aa_to_cg):
    
        xyz_ca = xyz_ref[map_ref]
        xyz_prior.append(np.random.normal(loc=xyz_ca, scale=scale * np.ones(xyz_ca.shape), size=xyz_ca.shape))

        # Need to account for variable masking procedure
        #if mask_idxs is not None:
        #    xyz_prior[:, mask_idxs] = xyz[:, mask_idxs]
    
    return np.array(xyz_prior)


for epoch in range(n_epochs):
        
    # this probably taking a while to set up each epoch
    xyz_prior = get_prior_mix(xyz_true, load_dict['map'], scale=Ca_std)
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
            time_batch = batch_sizet = model(t, x.detach())
           
        # repeat values over time batch
        x1 = x1.repeat(time_batch, 1, 1)
        x0 = x0.repeat(time_batch, 1, 1)
        res_feats = res_feats.repeat(time_batch, 1)
        atom_feats = atom_feats.repeat(time_batch, 1)
        mask = mask.repeat(time_batch, 1)

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

        if diff_type == 'xt_mask':
            vt = xt_pred - xt_mask
        elif diff_type == 'xt':
            vt = xt_pred - xt
        elif diff_type == 'xt_reg':
            vt = (xt_pred - xt) / (-t_pad + 1)
        elif diff_type == 'x0':
            vt = xt_pred - x0
            
        if loss_type == 'L2':
            loss = torch.mean((vt - ut) ** 2)
        elif loss_type == 'L1':
            loss = torch.mean(torch.abs(vt - ut))
        
        loss.backward()
        optimizer.step()
        
        mean_loss.append(loss.item())
        
    print(epoch, np.mean(mean_loss))
    loss_list.append(np.mean(mean_loss))
    
    # update lr scheduler if included
    if lrdecay > 0.0:
        scheduler.step()
    
    # get bond quality (and clash) every 100 epochs -- just check one structure with N gens for now
    if epoch%eval_every==0 and epoch>0:
        
        # can iterate over this and test one at a time
        test_idx_list = np.arange(n_test)
        n_gens = 1 
                
        bf_list = []
        cls_list = []
        
        for idx in test_idx_list:
        
            test_idxs = np.repeat(idx, n_gens)
            xyz_test_real = [xyz_true[i] for i in test_idxs]
            map_test = [load_dict['map'][i] for i in test_idxs]

            # when using mixed batch needs to be array
            xyz_test_prior = get_prior_mix(xyz_test_real, map_test, scale=Ca_std)

            model_wrpd = ModelWrapper(model=model, 
                              feats=torch.tensor(np.array([load_dict['res'][i] for i in test_idxs])).int().to(device), 
                              mask=torch.tensor(np.array([masks[i] for i in test_idxs])).bool().to(device).to(device), 
                              atom_feats=torch.tensor(np.array([load_dict['atom'][i] for i in test_idxs])).to(device), 
                              adj_mat=None)
            
            # adaptive solver (requires torchdyn)
            if solver == 'adapt':
                n_ode_steps = 5  # save frequency for visualizing
                tol = 3e-5       # optimized rtol and atol values 
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
                                                                    dtype=torch.float32).to(device),
                                                                    diff_type=diff_type)
                       
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
            
            # don't run clash score for dna-proteins
            if 'dna-pro' not in system:
                try: cls_list += [clash_res_percent(trj_gen) for trj_gen in trj_gens]
                except: print('Failed', [res for res in top.residues])
                   
            print('\nN res: ', top.n_residues)
            print('bf: ', np.mean(bf_list[-5:]).round(2))
            print('cls: ', np.mean(cls_list[-5:]).round(2))
            
            np.save(f'{job_dir}/ode-{epoch}_f-{idx}.npy', ode_traj)
                
        np.save(f'{job_dir}/bf-{epoch}.npy', bf_list)
        np.save(f'{job_dir}/cls-{epoch}.npy', cls_list)
        
        # save ode outputs for visualization
        torch.save(model.state_dict(), f'{job_dir}/state-{epoch}.pth')
        
  
   

