
import os
import matplotlib.pyplot as plt  # removing this thows scipy.optimize gcc error
import numpy as np
import torch
#import torchdyn
#from torchdyn.core import NeuralODE
from Bio.PDB import PDBParser, PDBIO, Select
#from torchcfm.conditional_flow_matching import *
import mdtraj as md
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from itertools import islice, count
import math

#try:
from egnn_pytorch_se3.egnn_pytorch import EGNN_SE3, EGNN
se3_avail = True
    
# except:
#     from egnn_pytorch import EGNN
#     se3_avail = False
#     print('using egnn_pytorch standard')

def exists(val):
    return val is not None

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask = None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask = mask)
        out     = self.attn2(x, induced)

        x =  out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries
    
def generate_cos_pos_encoding(n, dim, device, scale=10000.0):
    '''MJ -- replace pos_emb with sin/cos'''

    assert dim % 2 == 0, "dim must be even for alternating sin/cos encoding"
    
    pos_enc = torch.zeros(n, dim, device=device)
    position = torch.arange(0, n, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(scale) / dim))
    
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc

# adapted from https://github.com/lucidrains/egnn-pytorch with added time conditioning
class EGNN_Network_time(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        num_tokens = None,
        num_edge_tokens = None,
        num_positions = None,
        emb_cos_scale = None,
        edge_dim = 0,
        num_adj_degrees = None,
        adj_dim = 0,
        global_linear_attn_every = 0,
        global_linear_attn_heads = 8,
        global_linear_attn_dim_head = 64,
        num_global_tokens = 4,
        time_dim=0, 
        res_dim=20,  # change to 21
        atom_dim=3,  # change to 5
        sym='e3',
        **kwargs    # MJ -- include seq_features, and seq_decay in kwargs
    ):
        super().__init__()
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'
        self.num_positions = num_positions
        self.emb_cos_scale = emb_cos_scale
        
        self.res_emb = nn.Embedding(res_dim, dim)
        self.atom_emb = nn.Embedding(atom_dim, dim)
        #self.time_emb = nn.Embedding(1, dim)

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = edge_dim > 0

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        edge_dim = edge_dim if self.has_edges else 0
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        if has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_global_layer = has_global_attn and (ind % global_linear_attn_every) == 0

            if sym=='e3':
                self.layers.append(nn.ModuleList([
                    GlobalLinearAttention(dim = dim, heads = global_linear_attn_heads, dim_head = global_linear_attn_dim_head) if is_global_layer else None,
                    EGNN(dim = dim, edge_dim = (edge_dim + adj_dim), norm_feats = True, **kwargs),
                ]))
                
            elif sym=='se3' and se3_avail:
                self.layers.append(nn.ModuleList([
                    GlobalLinearAttention(dim = dim, heads = global_linear_attn_heads, dim_head = global_linear_attn_dim_head) if is_global_layer else None,
                    EGNN_SE3(dim = dim, edge_dim = (edge_dim + adj_dim), norm_feats = True, **kwargs),
                ]))

        # MJ -- add an MLP to encode time
        self.time_dim = time_dim
        if self.time_dim > 0:
            self.time_net = torch.nn.Sequential(
                torch.nn.Linear(1, self.time_dim),
                torch.nn.SELU(),
                torch.nn.Linear(self.time_dim, self.time_dim),
                torch.nn.SELU(),
                torch.nn.Linear(self.time_dim, dim),
            )
    
        #self.generate_cos_pos_encoding = generate_cos_pos_encoding

    def forward(self, x):
        return self.net(x)

    def forward(
        self,
        feats,
        coors,
        time,
        atom_feats=None,
        adj_mat = None,
        edges = None,
        mask = None,
        return_coor_changes = False
    ):
        b, a, device = feats.shape[0], feats.shape[1], feats.device

        if exists(self.token_emb):
            feats = self.token_emb(feats)
        
        if atom_feats != None:
            feats += self.atom_emb(atom_feats)

        if exists(self.pos_emb):
            n = feats.shape[1]
            assert n <= self.num_positions, f'given sequence length {n} must be less than the number of positions {self.num_positions} set at init'
            pos_emb = self.pos_emb(torch.arange(n, device = device))
            feats += rearrange(pos_emb, 'n d -> () n d')

        # don't use both the linear and cos embeddings
        elif exists(self.emb_cos_scale):
            n, dim = feats.shape[1], feats.shape[2]
            pos_emb_cos = generate_cos_pos_encoding(n, dim, device=device, scale=self.emb_cos_scale)
            feats += rearrange(pos_emb_cos, 'n d -> () n d')
            
        else:
            pass
            #print('No absolute pos encoding')
          

        # if time passed as single float or dim 0 tensor
        if isinstance(time, float) or time.dim()==0:
            time = time*torch.ones((b, a, 1)).to(device)
            
        # if time is passed as a tensor of size b
        elif time.dim() == 1:
            time = time[:, None].repeat(1, a)[:, :, None]
            
        #else:
        #    feats += time[:, None, None]
        
        # use a time embedding MLP
        if self.time_dim > 0:
            time = self.time_net(time)
            
        feats += time

        if exists(edges) and exists(self.edge_emb):
            edges = self.edge_emb(edges)
        
        # create N-degrees adjacent matrix from 1st degree connections
        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        # setup global attention

        global_tokens = None
        if exists(self.global_tokens):
            global_tokens = repeat(self.global_tokens, 'n d -> b n d', b = b)
            
        # go through layers

        coor_changes = [coors]

        for global_attn, egnn in self.layers:
            if exists(global_attn):
                feats, global_tokens = global_attn(feats, global_tokens, mask = mask)

            feats, coors = egnn(feats, coors, adj_mat = adj_mat, edges = edges, mask = mask)
            coor_changes.append(coors)

        if return_coor_changes:
            return feats, coors, coor_changes

        return feats, coors
    
def get_adj_mat(top):

    num_atoms = top.n_atoms
    
    bonded_pairs = []
    for bond in top.bonds:
        bonded_pairs.append((bond[0].index, bond[1].index))

    adj_mat = np.zeros((num_atoms, num_atoms), dtype=int)

    # Fill the adjacency matrix based on bonded pairs
    for pair in bonded_pairs:
        adj_mat[pair[0], pair[1]] = 1
        adj_mat[pair[1], pair[0]] = 1

    adj_mat = torch.tensor(adj_mat, dtype=torch.bool)
    return adj_mat

def get_adj_CG(xyz, mask_idxs, cut=1.0):
    '''Directly connect all CG atoms only'''

    num_atoms = xyz.shape[1]
    adj_mat = np.zeros((num_atoms, num_atoms), dtype=int)
    
    for i in range(num_atoms):
        if i in mask_idxs:
            dists = np.sqrt(((xyz[i] - xyz[mask_idxs])**2).sum(axis=-1))
            include_idxs = np.where(dists < cut)[0]
            adj_mat[i, mask_idxs[include_idxs]] = 1
        
    adj_mat = torch.tensor(adj_mat, dtype=torch.bool)
    return adj_mat

    
def get_aa_to_cg(top, msk):
    '''Mapping between AA and CG
       Assign to Ca positions for now with mask, but will need to generalize this'''
    
    aa_to_cg = []
    for atom_idx, atom in enumerate(top.atoms):
        res_idx = atom.residue.index
        aa_to_cg.append(msk[res_idx])
        
    return np.array(aa_to_cg)


def get_prior(xyz, aa_to_cg, mask_idxs=None, scale=1.0, frames=None):
    '''Normally distribute around respective Ca center of mass'''
    
    # set center of distribution to each CA and use uniform scale
    xyz_ca = xyz[:, aa_to_cg]
    scale = scale * np.ones(xyz_ca.shape)
    xyz_prior = np.random.normal(loc=xyz_ca, scale=scale, size=xyz.shape)
    
    # don't add noise to masked values
    if mask_idxs is not None:
        xyz_prior[:, mask_idxs] = xyz[:, mask_idxs]
    
    return xyz_prior


def get_prior_mix(xyz, aa_to_cg, scale=1.0):
    '''Normally distribute around respective Ca center of mass'''
    
    # set center of distribution to each CA and use uniform scale
    xyz_prior = []
    
    for xyz_ref, map_ref in zip(xyz, aa_to_cg):
    
        xyz_ca = xyz_ref[map_ref]
        xyz_prior.append(np.random.normal(loc=xyz_ca, scale=scale * np.ones(xyz_ca.shape), size=xyz_ca.shape))
    
    return xyz_prior #np.array(xyz_prior, dtype=object)  # fix ragged nest warning

def get_prior_mask(xyz, aa_to_cg, masks=None, scale=1.0):
    '''Normally distribute around respective masked coordinates
       Optionally mask out CG values so they are identical in CG and AA traces'''
    
    # set center of distribution to each CA and use uniform scale
    xyz_prior = []
    for i, (xyz_ref, map_ref) in enumerate(zip(xyz, aa_to_cg)):
    
        xyz_ca = xyz_ref[map_ref]
        xyz_ca = np.random.normal(loc=xyz_ca, scale=scale * np.ones(xyz_ca.shape), size=xyz_ca.shape)

        # ensure masked values are not noised at all
        if masks is not None:
            mask = ~masks[i].astype(bool)
            xyz_ca[mask] = xyz_ref[mask]
        xyz_prior.append(xyz_ca)
    
    return xyz_prior

    
def str_to_ohe(string_list):
    unique_strings = list(set(string_list))
    string_to_index = {string: index for index, string in enumerate(unique_strings)}
    indices = [string_to_index[string] for string in string_list]
    return np.array(indices)

def get_node_ohes(top):
    '''get one-hot encodings of residue and atom element identities'''
    
    res_list, atom_list = [], []

    for a in top.atoms:
        if a.element.name != 'hydrogen': 
            res_list.append(a.residue.name)
            atom_list.append(a.element.name)

    res_ohe = str_to_ohe(res_list)
    atom_ohe = str_to_ohe(atom_list)
    
    return res_ohe, atom_ohe
    
# load data set
class CustomDataset(Dataset):
    def __init__(self, data_list1, data_list2):
        self.data_list1 = data_list1
        self.data_list2 = data_list2

    def __len__(self):
        return len(self.data_list1)  # Assuming both lists have the same length

    def __getitem__(self, index):
        sample1 = torch.tensor(self.data_list1[index], dtype=torch.float32)
        sample2 = torch.tensor(self.data_list2[index], dtype=torch.float32)

        return sample1, sample2
    
# load data set
class FeatureDataset(Dataset):
    def __init__(self, data_list1, data_list2, data_list3, data_list4, data_list5, data_list6):
        self.data_list1 = data_list1
        self.data_list2 = data_list2
        self.data_list3 = data_list3
        self.data_list4 = data_list4
        self.data_list5 = data_list5
        self.data_list6 = data_list6

    def __len__(self):
        return len(self.data_list1)  # Assuming both lists have the same length

    def __getitem__(self, index):
        sample1 = torch.tensor(self.data_list1[index], dtype=torch.float32)
        sample2 = torch.tensor(self.data_list2[index], dtype=torch.float32)
        sample3 = torch.tensor(self.data_list3[index], dtype=torch.int)
        sample4 = torch.tensor(self.data_list4[index], dtype=torch.int)
        sample5 = torch.tensor(self.data_list5[index], dtype=torch.bool)
        sample6 = torch.tensor(self.data_list6[index], dtype=torch.bool)

        return sample1, sample2, sample3, sample4, sample5, sample6
    
    
def pro_res_to_ohe(string_list):
    
    amino_acids = {
    "ALA": 1,  # Alanine
    "ARG": 2,  # Arginine
    "ASN": 3,  # Asparagine
    "ASP": 4,  # Aspartic acid
    "CYS": 5,  # Cysteine
    "GLU": 6,  # Glutamic acid
    "GLN": 7,  # Glutamine
    "GLY": 8,  # Glycine
    "HIS": 9,  # Histidine
    "ILE": 10, # Isoleucine
    "LEU": 11, # Leucine
    "LYS": 12, # Lysine
    "MET": 13, # Methionine
    "PHE": 14, # Phenylalanine
    "PRO": 15, # Proline
    "SER": 16, # Serine
    "THR": 17, # Threonine
    "TRP": 18, # Tryptophan
    "TYR": 19, # Tyrosine
    "VAL": 20  # Valine
    }

    indices = [amino_acids[string] for string in string_list]
    return np.array(indices)

def pro_atom_to_ohe(string_list):
    
    atom_types = {
    "carbon": 1,
    "oxygen": 2, 
    "nitrogen": 3, 
    "sulfur": 4,
    }

    indices = [atom_types[string] for string in string_list]
    return np.array(indices)

def pro_allatom_to_ohe(string_list):
    '''List all possible atom names'''
    
    atom_types = {
        'C':1,
        'CA':2,
        'CB':3,
        'CD':4,
        'CD1':5,
        'CD2':6,
        'CE':7,
        'CE1':8,
        'CE2':9,
        'CE3':10,
        'CG':11,
        'CG1':12,
        'CG2':13,
        'CH2':14,
        'CZ':15,
        'CZ2':16,
        'CZ3':17,
        'N':18,
        'ND1':19,
        'ND2':20,
        'NE':21,
        'NE1':22,
        'NE2':23,
        'NH1':24,
        'NH2':25,
        'NZ':26,
        'O':27,
        'OD1':28,
        'OD2':29,
        'OE1':30,
        'OE2':31,
        'OG':32,
        'OG1':33,
        'OH':34,
        'SD':35,
        'SG':36
    }
  
    indices = [atom_types[string] for string in string_list]
    return np.array(indices)


def get_pro_ohes(top):
    '''get one-hot encodings of residue and atom element identities'''
    
    res_list, atom_list, allatom_list = [], [], []

    for a in top.atoms:
        if a.element.name != 'hydrogen': 
            res_list.append(a.residue.name) 
            atom_list.append(a.element.name) 
            allatom_list.append(a.name)       
    
    res_ohe = pro_res_to_ohe(res_list)
    atom_ohe = pro_atom_to_ohe(atom_list)
    allatom_ohe = pro_allatom_to_ohe(allatom_list)
    
    return res_ohe, atom_ohe, allatom_ohe


def dna_res_to_ohe(string_list):
    
    bases = {
    "DA": 1,  
    "DC": 2,  
    "DG": 3,  
    "DT": 4,
    }

    indices = [bases[string] for string in string_list]
    return np.array(indices)

# should change this to be consistent with proteins atom types
def dna_atom_to_ohe(string_list):
    
    atom_types = {
    "carbon": 1,
    "oxygen": 2, 
    "nitrogen": 3, 
    "phosphorus": 5,  # change to 5 and scale COMs by 1
    "BCOM":6,
    "SCOM":7,
    "PCOM":8, 
    }

    indices = [atom_types[string] for string in string_list]
    return np.array(indices)

def dna_allatom_to_ohe(string_list):
    '''List all possible atom names'''
    
    atom_types = {
        "O4": 1,
        "O2": 2,
        "C1'": 3,
        "N4": 4,
        "O6": 5,
        "N7": 6,
        "C3'": 7,
        "C4'": 8,
        "N1": 9,
        "C7": 10,
        "C2'": 11,
        "C2": 12,
        "C8": 13,
        "C5'": 14,
        "N3": 15,
        "N9": 16,
        "N6": 17,
        "P": 18,
        "OP1": 19,
        "C6": 20,
        "O4'": 21,
        "O5'": 22,
        "C4": 23,
        "OP2": 24,
        "N2": 25,
        "O3'": 26,
        "C5": 27,
        "BCOM":28,
        "SCOM":29,
        "PCOM":30, 
    }
  
    indices = [atom_types[string] for string in string_list]
    return np.array(indices)


def get_dna_ohes(top):
    '''get one-hot encodings of residue and atom element identities'''
    
    res_list, atom_list, allatom_list = [], [], []

    for a in top.atoms:
        if a.element.name != 'hydrogen': 
            res_list.append(a.residue.name)
            atom_list.append(a.element.name)
            allatom_list.append(a.name)
    
    res_ohe = dna_res_to_ohe(res_list)
    atom_ohe = dna_atom_to_ohe(atom_list)
    allatom_ohe = dna_atom_to_ohe(atom_list)
    
    return res_ohe, atom_ohe, allatom_ohe


# backmap to all heavy atoms given only the Ca positions
def parse_dna_3spn(dna_trj, with_pro=False):
    '''Extract GNN parameters compatible with 3sn2 CG representation of DNA
       Ensure that dna_trj only includes dna residues
       If proteins also included in then need to add constant to ohes'''
    
    print('init trj shape', dna_trj.xyz.shape)
    
    # seperate AA and CG components if both are included
    try:
        cg_idxs = dna_trj.top.select(f"name DS or name DP or name DB")
        all_idxs = range(dna_trj.n_atoms)
        aa_idxs = [idx for idx in all_idxs if idx not in cg_idxs]

        cg_trj = dna_trj.atom_slice(cg_idxs)
        dna_trj = dna_trj.atom_slice(aa_idxs)
        
    except:
        cg_trj = None
        
    # get all 5' and 3' residues idxs 
    ter_res_list = []
    for chain in dna_trj.topology.chains:
        residues = list(chain.residues)
    
        # Determine the site type for each residue in the chain
        for index, residue in enumerate(residues):
            if index == 0:
                ter_res_list.append(5)
            elif index == len(residues) - 1:
                ter_res_list.append(3)
            else:
                ter_res_list.append(0)
    
    dna_top = dna_trj.top
    n_resid = dna_top.n_residues
    xyz = dna_trj.xyz
    n_frames = len(xyz)
    n_atoms = xyz.shape[1]
    
    xyz_com = [xyz]
    aa_to_cg = np.zeros(n_atoms)
    mask_idxs = []
    cg_atom_list = []
    cg_res_list = []

    for n in range(0, n_resid):

        # make sure to collect O3 from the previous residue
        res_idxs = dna_top.select(f'resid {n} and not name "O3\'"')
        chain_id = dna_top.atom(res_idxs[0]).residue.chain.index
        
        # if not a 5' end then include the O3'
        if ter_res_list[n] != 5:
            O3_prev = dna_top.select(f'resid {n-1} and name "O3\'"')
            res_idxs = np.concatenate([res_idxs, O3_prev])
            
        # if a 3' end then incldue terminal 03' in mapping but not in com
        if ter_res_list[n] == 3:
            O3_curr = dna_top.select(f'resid {n} and name "O3\'"')[0]
        else:
            O3_curr = None
            
        # get names of all atoms in resid
        atom_list = [next(islice(dna_top.atoms, idx, None)).name for idx in res_idxs]

        # get res name
        res_name = next(islice(dna_top.atoms, res_idxs[0], None)).residue.name

        # break if hit a CG coord:
        if 'DS' in atom_list:
            continue

        # passing each res to each chain type
        b_idxs, s_idxs, p_idxs = [], [], []
        b_names, s_names, p_names = [], [], []
        
        for idx, name in zip(res_idxs, atom_list):
            
            # need to get exact lists here and verify against 3spn code -- eg 3 
            if name in ['P', 'OP2', 'OP1', 'O5\'', 'O3\'']:
                p_idxs.append(idx)
                p_names.append(name)
            elif "'" in name: 
                s_idxs.append(idx)
                s_names.append(name)
            else: 
                b_idxs.append(idx)
                b_names.append(name)
                
        # compute center of mass for each
        b_coms = md.compute_center_of_mass(dna_trj.atom_slice(b_idxs)).reshape((n_frames, 1, 3))
        s_coms = md.compute_center_of_mass(dna_trj.atom_slice(s_idxs)).reshape((n_frames, 1, 3))
        
        # append terminal 03' after calculating coms (part of mask but not COM)
        if O3_curr is not None:
            s_idxs.append(O3_curr)

        # check if any phosphates in the residue (don't want to group based on 05' alone)
        if len(p_idxs) > 1:
            p_coms = md.compute_center_of_geometry(dna_trj.atom_slice(p_idxs)).reshape((n_frames, 1, 3))
            xyz_com.append(p_coms)
            xyz_com.append(s_coms)
            xyz_com.append(b_coms)
            
            # check why not getting any b_idxs -- residue has phosphat but no b or s?
            #print('b s p', len(b_idxs), len(s_idxs), len(p_idxs))

            # map to b, s, or p coms
            aa_to_cg[np.array(p_idxs)] = n_atoms + len(xyz_com) - 4
            aa_to_cg[np.array(s_idxs)] = n_atoms + len(xyz_com) - 3
            aa_to_cg[np.array(b_idxs)] = n_atoms + len(xyz_com) - 2

            cg_atom_list += ['PCOM', 'SCOM', 'BCOM']  #['BCOM', 'SCOM', 'PCOM']
            cg_res_list += [res_name]*3

        else:
            # map any missing atoms to the sugar
            xyz_com.append(s_coms)
            xyz_com.append(b_coms)
            
            s_idxs += p_idxs
            aa_to_cg[np.array(s_idxs)] = n_atoms + len(xyz_com) - 3
            aa_to_cg[np.array(b_idxs)] = n_atoms + len(xyz_com) - 2

            cg_atom_list += ['SCOM', 'BCOM'] #['BCOM', 'SCOM']
            cg_res_list += [res_name]*2

    # a lot easier to append everything at the end the xyz
    xyz_com = np.concatenate(xyz_com, axis=1)
    n_atoms_com = xyz_com.shape[1]

    # if cg coords exist, replace the xyz_com values with these
    # need to change order from B-S-P to P-S-B
    if cg_trj is not None:
        print('xyz_com', xyz_com.shape, cg_trj.xyz.shape)
        xyz_com[:, -cg_trj.xyz.shape[1]:] = cg_trj.xyz

    # set mask values to COMs only
    mask_idxs = np.arange(n_atoms, n_atoms_com)

    # set CG mask values to themselves 
    aa_to_cg = np.concatenate([aa_to_cg, np.arange(n_atoms, n_atoms_com)])
    aa_to_cg = np.array(aa_to_cg, dtype=int)
    
    # get res and atom feats for standard atoms
    res_ohe, atom_ohe, all_atom_ohe = get_dna_ohes(dna_top)

    # manually add com encodings for now -- based on pos encoding this might work better dispersed in sequence
    res_ohe = np.concatenate([res_ohe, dna_res_to_ohe(cg_res_list)])
    atom_ohe = np.concatenate([atom_ohe, dna_atom_to_ohe(cg_atom_list)])
    all_atom_ohe = np.concatenate([all_atom_ohe, dna_allatom_to_ohe(cg_atom_list)])
    
    # ensure no overlap with pro encoding
    if with_pro:
        res_ohe = res_ohe + 20
        atom_ohe = atom_ohe + 5
        all_atom_ohe = all_atom_ohe + 36
    
    return xyz_com, mask_idxs, aa_to_cg, res_ohe, all_atom_ohe


# set up alagous protein parse function
def parse_pro_CA(pro_trj):
    '''Extract GNN parameters compatible with 3sn2 CG representation of DNA
       Ensure that pro_trj only includes dna residues'''
    
    res_ohe, atom_ohe, all_atom_ohe = get_pro_ohes(pro_trj.top)
    mask_idxs = pro_trj.top.select('name CA')
    aa_to_cg = get_aa_to_cg(pro_trj.top, mask_idxs)
    xyz = pro_trj.xyz
    
    return xyz, mask_idxs, aa_to_cg, res_ohe, all_atom_ohe
    
class ModelWrapper(EGNN_Network_time):
    def __init__(
        self,
        model,
        feats,
        mask,
        atom_feats=None,
        adj_mat=None
    ):
        super(EGNN_Network_time, self).__init__()  # Call the nn.Module constructor
    
        self.model = model
        self.feats = feats
        self.mask = mask
        self.atom_feats = atom_feats
        self.adj_mat = adj_mat
        
    def forward(self, t, x, y=None, *args, **kwargs):
        
        feats_out, coors_out = self.model(self.feats, x, mask=self.mask, time=t,
                                    atom_feats=self.atom_feats, adj_mat=self.adj_mat)
        return coors_out - x
    
# add custom integrators
def euler_integrator(model, x, nsteps=100, mask=None, noise=0.0):
    
    # try adding small amounts of noise during integration
    
    ode_list = []
    dt = 1./(nsteps-1)
    
    for t in np.linspace(0, 1, nsteps):
        
        # Evaluate dx/dt using the model for the current state and time
        with torch.no_grad():
            dx_dt = model(t, x.detach())

        # Compute the next state using Euler's formula
        x = (x + dx_dt * dt).detach()
        
        # add some noise to model sde (except for masked values)
        if mask != None:
            masked_noise = mask*noise*torch.randn(mask.shape).to(device)
            masked_noise = masked_noise.unsqueeze(-1).expand(-1, -1, 3)
            x += masked_noise
        
        # track each update to show diffusion path
        ode_list.append(x.cpu().numpy())

    return np.array(ode_list)

def runge_kutta_integrator(model, x, nsteps=50):
    '''Runge Kutta 4th order solver (takes ~4x longer per-step compated to Euler)'''
    
    ode_list = []
    dt = 1./(nsteps-1)
    
    for t in np.linspace(0, 1, nsteps):
        
        k1 = model(t, x).detach()
        k2 = model(t + 0.5*dt, x + 0.5*dt*k1).detach()
        k3 = model(t + 0.5*dt, x + 0.5*dt*k2).detach()
        k4 = model(t + dt, x + dt*k3).detach()
    
        # Compute the next state using the RK4 formula
        x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # track each update to show diffusion path
        ode_list.append(x.cpu().numpy())

    return np.array(ode_list)

    
def bond_fraction(trj_ref, trj_gen, fraction=0.1):
    '''Fraction of bonds within X percent of the reference'''

    bond_pairs = [[b[0].index, b[1].index] for b in trj_ref.top.bonds]
    ref_dist = md.compute_distances(trj_ref, bond_pairs)
    gen_dist = md.compute_distances(trj_gen, bond_pairs)

    bond_frac = np.sum((gen_dist < (1+fraction)*ref_dist) & 
                       (gen_dist > (1-fraction)*ref_dist))

    bond_frac = bond_frac / np.size(ref_dist)
    
    return bond_frac

def get_res_idxs_cut(trj, thresh=0.12, Ca_cut=2.0):
    Ca_idxs = []
    for i, atom in enumerate(trj.top.atoms):
        if 'CA' in atom.name:
            Ca_idxs.append(i)
    Ca_idxs = np.array(Ca_idxs)
    Ca_xyzs = trj.xyz[0, Ca_idxs]
    n_res = trj.n_residues
    pairs = []
    for i in range(n_res):
        for j in range(i-1):
            if np.linalg.norm(Ca_xyzs[i]-Ca_xyzs[j]) < Ca_cut:
                pairs.append((i, j))
                
    dist, pairs = md.compute_contacts(trj, contacts=pairs, scheme='closest')
    # look at sidechain-heavy only
    
    neighbor_pairs = [(i, i+1) for i in range(trj.n_residues-1) if (
        trj.top.residue(i).name != 'GLY' and trj.top.residue(i+1).name != 'GLY')]
    
    neighbor_dist, neighbor_pairs = md.compute_contacts(trj, contacts=neighbor_pairs, scheme='sidechain-heavy')
    dist = np.concatenate([dist, neighbor_dist], axis=-1)
    pairs = np.concatenate([pairs, neighbor_pairs], axis=0)
    res_closes = list()
    for n_res in range(trj.top.n_residues):
        pair_mask = np.array([n_res in i for i in pairs])
        res_close = np.any(dist[0, pair_mask] < thresh)
        res_closes.append(res_close)
    res_closes = np.array(res_closes)
    return res_closes

def clash_res_percent(viz_gen, thresh=0.12, Ca_cut=2.0):
    all_res_closes = list()
    #for n in tqdm(range(len(viz_gen))):
    for n in range(len(viz_gen)):
        res_closes = get_res_idxs_cut(viz_gen[n], thresh=thresh, Ca_cut=Ca_cut)
        all_res_closes.append(res_closes)
    return 100 * sum([sum(i) for i in all_res_closes]) / sum([i.shape[0] for i in all_res_closes])
    
    
# for calculating generative diversity
def ref_rmsd(trj_ref, trj_sample_list):
    
    rmsd_list = []
    for i, trj_i in enumerate(trj_sample_list):
        
        print(trj_i.xyz.shape, trj_ref.xyz.shape)

        for k, (trj_if, trj_rf) in enumerate(zip(trj_i, trj_ref)):
            rmsd = md.rmsd(trj_if, trj_rf)*10
            rmsd_list.append(rmsd)
            #frame_rmsds.append(rmsd)
        #rmsd_list.append(np.mean(frame_rmsds))

    return np.mean(rmsd_list), np.std(rmsd_list)

def sample_rmsd(trj_sample_list):
    
    rmsd_list = []
    for i, trj_i in enumerate(trj_sample_list):
        for j, trj_j in enumerate(trj_sample_list[:i]):
            
            # need to compare per frames rmsds or else will be relative to first frame
            #frame_rmsds = []
            for k, (trj_if, trj_jf) in enumerate(zip(trj_i, trj_j)):
                rmsd = md.rmsd(trj_if, trj_jf)*10
                rmsd_list.append(rmsd)
                #frame_rmsds.append(rmsd)
            #rmsd_list.append(np.mean(frame_rmsds))

    return np.mean(rmsd_list), np.std(rmsd_list)

def sample_rmsd_percent(trj_ref, trj_sample_list):
    
    R_ref, S_ref = ref_rmsd(trj_ref, trj_sample_list)
    R_sam, S_sam = sample_rmsd(trj_sample_list)
    
    R_per = (R_ref-R_sam) / R_ref
    S_per = np.sqrt( (S_sam/R_ref)**2 + ((R_sam*S_ref)/(R_ref)**2)**2 )
    
    return R_per, S_per

# get uncertainties on diversity scores 
def jackknife_div(trj_ref, trj_sample_list):
    gen_ref = get_ref_gen_rmsds(trj_ref, trj_gens)
    gen_gen = get_sample_rmsds(trj_gens) 
    
    assert len(gen_ref) == len(gen_gen)
    
    div_mat = np.zeros((trj_ref.n_frames, len(trj_sample_list)))
    for frame_idx, (gen_gen_i, gen_ref_i) in enumerate(zip(gen_gen, gen_ref)):
        for targ in range(len(trj_sample_list)):
            gen_gen_mean = np.mean([v for i,v in gen_gen_i.items() if targ not in i])
            gen_ref_mean = np.mean(np.delete(gen_ref_i, targ))
            div_mat[frame_idx][targ] = 1 - (gen_gen_mean / gen_ref_mean)
    return div_mat.mean(0)

# Define a custom selector for filtering
class ProteinDNASelect(Select):
    def accept_residue(self, residue):
        # Accept only protein and DNA residues
        if "CA" in residue or residue.id[0] == " " and residue.resname.strip() in ["DA", "DC", "DG", "DT"]:
            return 1  # Return 1 to indicate acceptance
        else:
            return 0  # Return 0 to exclude everything else

def reformat_pro_dna(load_name, save_name):
    '''Removes any solvent or non-DNA-PRO atoms and re-orders so that protein always comes first'''

    # Load the PDB file
    parser = PDBParser()
    structure = parser.get_structure('temp', load_name)

    # Separate protein and DNA chains, maintaining original check for inclusion
    protein_chains = []
    dna_chains = []
    for model in structure:
        for chain in model:
            is_protein = False
            is_dna = False
            for residue in chain:
                if "CA" in residue:  # Check for C-alpha atom as a proxy for protein
                    is_protein = True
                if residue.resname.strip() in ["DA", "DC", "DG", "DT"]:  # Common DNA residues
                    is_dna = True
            if is_protein or is_dna:
                # Instead of immediately appending, copy the chain to filter non-protein/DNA atoms
                filtered_chain = chain.copy()
                for residue in list(filtered_chain.get_residues()):
                    if not ProteinDNASelect().accept_residue(residue):
                        filtered_chain.detach_child(residue.id)
                # Now, append the filtered chain
                if is_protein:
                    protein_chains.append(filtered_chain)
                elif is_dna:
                    dna_chains.append(filtered_chain)

    # Assuming you want to reorder at the model level, for simplicity
    model = next(structure.get_models())  # Get the first model to work with

    # Clear the model
    for chain in list(model.get_chains()):
        model.detach_child(chain.id)

    # Reattach chains in the desired order: proteins first, then DNA
    for chain in protein_chains + dna_chains:
        model.add(chain)

    # Save the reordered PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(save_name)
    
    
   
    