import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, AttributedGraphDataset
import random
from tqdm import tqdm
import models
import scipy.stats
import copy
import os

def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--rel_path', type=str, default='./data')
    parser.add_argument('--repeat_times', type=int, default=5)
    parser.add_argument('--noise_type', type=str, default='mixed')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--noise_ratio', type=float, default=0.4)
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--scheduler_param', type=float, default=1.0)
    parser.add_argument('--search_scheduler',  action='store_true')
    parser.add_argument('--search_iteration', type=int, default=0)
    return
        
def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def getDataset(dataset_name, device, rel_path='./data'):
    assert dataset_name in ['Cora','Citeseer','Pubmed','chameleon','squirrel','facebook']
    transform = T.Compose([
                    T.NormalizeFeatures(),
                    T.ToDevice(device),
                    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                                    add_negative_train_samples=False),])
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        path = osp.join(rel_path, 'Planetoid')
        dataset = Planetoid(path, name=dataset_name, transform=transform)
    elif dataset_name in ['chameleon', 'squirrel']:
        path = osp.join(rel_path, 'WikipediaNetwork')
        dataset = WikipediaNetwork(path, name=dataset_name, transform=transform)
    elif dataset_name in ["facebook"]:
        path = osp.join(rel_path, 'AttributedGraphDataset')
        dataset = AttributedGraphDataset(path, name=dataset_name, transform=transform)
    else:
        exit()
    return path, dataset
        
def getGNNArch(GNN_name):
    assert GNN_name in ['GCN', 'GAT', 'SAGE', 'MLP']
    if GNN_name == 'GCN':
        return models.GCN
    elif GNN_name == 'GAT':
        return models.GAT
    elif GNN_name == 'SAGE':
        return models.SAGE
    elif GNN_name == 'MLP':
        return models.MLP

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)
    # calculate m
    m = (p + q) / 2
    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)
    return distance

def calculateDistSim(res, savePath=None):
    r_edge, r_node, label, predict = res
    label = label.int().tolist()
    cos = torch.nn.CosineSimilarity(dim=0)
    pos_sim, neg_sim = [], []
    for idx in range(r_node[0].shape[0]):
        label_idx = label[idx]
        sim = float(cos(r_node[0][idx], r_node[1][idx]))
        if label_idx == 1:
            pos_sim.append(sim+1)
        else:
            neg_sim.append(sim+1)
    js_dis = jensen_shannon_distance(pos_sim, neg_sim)
    ks_dis = scipy.stats.kstest(pos_sim, neg_sim).statistic
    kl_dis = np.mean(scipy.special.kl_div(sorted(pos_sim), sorted(neg_sim)))
    return [np.mean(pos_sim), np.mean(neg_sim), ks_dis]