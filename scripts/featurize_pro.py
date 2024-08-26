import glob
import mdtraj as md
import pickle as pkl
import argparse
from utils.egnn_utils import get_pro_ohes, get_aa_to_cg

parser = argparse.ArgumentParser()
parser.add_argument('--pdb_dir', type=str, help='Load pdbs from this dir')
parser.add_argument('--save_name', type=str, help='Save name for output features and tops')
parser.add_argument('--collect_tops', action='store_true',  help='Save topologies (only needed for validation set)')
args = parser.parse_args()

pdb_list = glob.glob(f'{args.pdb_dir}/*')
print('N pdbs:', len(pdb_list))

cnt = 0
xyz_list = []
res_list = []
atom_list = []
adj_mat_list = []
aa_to_cg_list = []
mask_list = []
top_list = []

for pdb in pdb_list:

    trj = md.load(pdb)
    non_h_indices = trj.top.select('not (element H)')
    trj = trj.atom_slice(non_h_indices)

    top = trj.top 
    n_res = trj.n_residues
    n_atoms = trj.n_atoms

    try:
        # get ohes
        res_ohe, atom_ohe, allatom_ohe = get_pro_ohes(top)

        # get mapping and CA_idxs
        mask_idxs = top.select('name CA')
        aa_to_cg = get_aa_to_cg(top, mask_idxs)
        xyz = trj.xyz[0]

        res_list.append(res_ohe)
        atom_list.append(allatom_ohe)   
        xyz_list.append(xyz)
        mask_list.append(mask_idxs)
        aa_to_cg_list.append(aa_to_cg)

        # only collect tops for validation set
        if args.collect_tops:
            top_list.append(top)
        cnt += 1

    except Exception as e:
        print(f'Failed {pdb} due to {e}')

    if cnt%100==0 and cnt > 0:
        print(cnt)                    
              
save_dict = {'res':res_list,'atom':atom_list,'xyz':xyz_list,'mask':mask_list,'map':aa_to_cg_list}
pkl.dump(save_dict, open(f'../train_features/feats_{args.save_name}.pkl', 'wb'))

if args.collect_tops:
    pkl.dump(top_list, open(f'../train_features/tops_{args.save_name}.pkl', 'wb'))
