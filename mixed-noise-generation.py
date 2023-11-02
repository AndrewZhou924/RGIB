import os
import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, AttributedGraphDataset
import random
import argparse
import networkx as nx
from utils import *

'''
Uasge of this python scipt, e.g.,
python3 mixed-noise-generation.py --dataset Cora       --noise_ratio 0.2
python3 mixed-noise-generation.py --dataset Citeseer   --noise_ratio 0.2
python3 mixed-noise-generation.py --dataset Pubmed     --noise_ratio 0.2
python3 mixed-noise-generation.py --dataset chameleon  --noise_ratio 0.2
python3 mixed-noise-generation.py --dataset squirrel   --noise_ratio 0.2
python3 mixed-noise-generation.py --dataset facebook   --noise_ratio 0.2
'''

def getDistanceMatrix(G):
    ''' return a matrix with shape of (num_nodes, num_nodes) '''
    num_nodes = G.number_of_nodes()
    distance = np.zeros((num_nodes, num_nodes))
    distance[:,:] = num_nodes + 1
    for h in range(num_nodes):
        distance[h,h] = 0
    for (h,t) in G.edges():
        distance[h,t] = 1
        distance[t,h] = 1
    return distance

def generateNoisyEdges(distance_matrix, num_noisy_edges, filter_mask=None):
    select_edges = []
    n_node = distance_matrix.shape[0]
    while len(select_edges) < num_noisy_edges:
        head, tail = np.random.randint(n_node, size=2)
        if distance_matrix[head, tail] >= 2 and [head, tail] not in select_edges:
            select_edges.append([head, tail])
    heads = [h for [h,t] in select_edges]
    tails = [t for [h,t] in select_edges]
    # double direction
    noisy_edges = [heads + tails, tails + heads] 
    noisy_edges = torch.Tensor(noisy_edges).int().to(device)
    return noisy_edges

### Parse args ###
parser = argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()
print(args)
assert args.noise_ratio >= 0

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])

# load dataset
if args.dataset in ['Cora', "Citeseer", 'Pubmed']:
    path = osp.join(args.rel_path, 'Planetoid')
    dataset = Planetoid(path, name=args.dataset, transform=transform)
elif args.dataset in ["chameleon", "squirrel"]:
    path = osp.join(args.rel_path, 'WikipediaNetwork')
    dataset = WikipediaNetwork(path, name=args.dataset, transform=transform)
elif args.dataset in ["facebook"]:    
    path = osp.join(args.rel_path, 'AttributedGraphDataset')
    dataset = AttributedGraphDataset(path, name=args.dataset, transform=transform)
else:
    exit()

# check saving folder
saving_folder = f'{path}/{args.dataset}/processed/'
if not os.path.exists(saving_folder): os.mkdir(saving_folder)

# load distance matrix
distance_matrix_path = osp.join(path, args.dataset, 'distance.npy')
if os.path.exists(distance_matrix_path):
    distance_matrix = np.load(distance_matrix_path)
else:
    print('==> Generating the distance matrix...')
    G = nx.Graph()
    num_nodes = dataset.data.x.shape[0]
    for node_idx in range(num_nodes):
        G.add_node(node_idx)
    G.add_edges_from(dataset.data.edge_index.T.tolist())
    distance_matrix = getDistanceMatrix(G)
    np.save(open(f'{path}/{args.dataset}/distance.npy', 'wb'), distance_matrix)
    print('==> Finished.')
            
for idx in range(args.repeat_times):            
    # save train_data, val_data, val_data
    savePath = f'{path}/{args.dataset}/processed/mixed_noise_ratio_{args.noise_ratio}_repeat_{idx+1}.pt'
    print('==> Save generated data to:', savePath)
    if os.path.exists(savePath):
        print(f'==> File already exists, skip path: {savePath}')
        continue
        
    # copy the original dataset
    train_data, val_data, test_data = [copy.deepcopy(d) for d in dataset[0]]
    if args.noise_ratio > 0:
        # label noise
        num_noisy_edges = int(args.noise_ratio * train_data.edge_label.shape[0] / 2)
        noisy_index = generateNoisyEdges(distance_matrix, num_noisy_edges)
        train_data.edge_label_index = torch.cat([train_data.edge_label_index, noisy_index], dim=1)
        noisy_edge_label = torch.ones(num_noisy_edges * 2).cuda()
        train_data.edge_label = torch.cat([train_data.edge_label, noisy_edge_label], dim=0)
        # input noise 
        num_noisy_edges = int(args.noise_ratio * len(train_data.edge_index[0]) / 2)
        noisy_index = generateNoisyEdges(distance_matrix, num_noisy_edges)
        train_data.edge_index = torch.cat([train_data.edge_index, noisy_index],dim=1)
        val_data.edge_index = torch.cat([val_data.edge_index, noisy_index],dim=1)
        test_data.edge_index = torch.cat([test_data.edge_index, noisy_index],dim=1)   
        
    torch.save((train_data, val_data, test_data), savePath)
