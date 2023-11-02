import random
import math
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, MLP
from utils import *
from loss import *
from hyperopt import hp, fmin, tpe

'''
Uasge of this python scipt, e.g.,
python3 RGIB-rep-training.py --gnn_model GCN --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2 --scheduler constant --scheduler_param 1.0
python3 RGIB-rep-training.py --gnn_model GCN --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2 --search_scheduler --search_iteration 50
'''

SAMPLING_RATIO = 1.0
def sampling_MI(prob, tau=0.8, reduction='none'):
    prob = prob.clamp(1e-4, 1-1e-4)
    entropy1 = prob * torch.log(prob / tau)
    entropy2 = (1-prob) * torch.log((1-prob) / (1-tau))
    res = entropy1 + entropy2
    if reduction == 'none':
        return res
    elif reduction == 'mean':
        return torch.mean(res)
    elif reduction == 'sum':
        return torch.sum(res)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def encode(self, x, edge_index):
        # get edge logits from original X
        z = x.clone()
        for conv in self.convs[:-1]:
            z = conv(z, edge_index).relu()
        self.tmp_z = self.convs[-1](z, edge_index)
        edge_logit = (self.tmp_z[edge_index[0]] * self.tmp_z[edge_index[1]]).sum(dim=-1)
        edge_weight = torch.nn.Sigmoid()(edge_logit)
        if self.training: self.encode_edge_weight = edge_weight
        # edge sampling
        sampled_index = (edge_weight > SAMPLING_RATIO * torch.rand_like(edge_weight)).detach()
        new_edge_index = edge_index[:, sampled_index]
        # forward with sampled edges
        for conv in self.convs[:-1]:
            x = conv(x, new_edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x
    
    def decode(self, x, z, edge_label_index):
        edge_logit = (self.tmp_z[edge_label_index[0]] * self.tmp_z[edge_label_index[1]]).sum(dim=-1)
        pos_weight = torch.nn.Sigmoid()(edge_logit)
        neg_weight = torch.ones_like(pos_weight)
        edge_weight = torch.cat([neg_weight.unsqueeze(1), pos_weight.unsqueeze(1)], dim=1)
        if self.training: self.decode_edge_weight = pos_weight

        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits, edge_weight

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        # get edge logits from original X
        z = x.clone()
        for conv in self.convs[:-1]:
            z = conv(z, edge_index).relu()
        self.tmp_z = self.convs[-1](z, edge_index)
        edge_logit = (self.tmp_z[edge_index[0]] * self.tmp_z[edge_index[1]]).sum(dim=-1)
        edge_weight = torch.nn.Sigmoid()(edge_logit)
        if self.training: self.encode_edge_weight = edge_weight
        # edge sampling
        sampled_index = (edge_weight > SAMPLING_RATIO * torch.rand_like(edge_weight)).detach()
        new_edge_index = edge_index[:, sampled_index]
        # forward with sampled edges
        for conv in self.convs[:-1]:
            x = conv(x, new_edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, x, z, edge_label_index):
        edge_logit = (self.tmp_z[edge_label_index[0]] * self.tmp_z[edge_label_index[1]]).sum(dim=-1)
        pos_weight = torch.nn.Sigmoid()(edge_logit)
        # neg_weight = 1 - pos_weight
        neg_weight = torch.ones_like(pos_weight)
        edge_weight = torch.cat([neg_weight.unsqueeze(1), pos_weight.unsqueeze(1)], dim=1)
        if self.training: self.decode_edge_weight = pos_weight
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits, edge_weight

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8, att_dropout=0):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
        for i in range(0, num_layers-2):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, dropout=att_dropout))

    def encode(self, x, edge_index):
        # get edge logits from original X
        z = x.clone()
        for conv in self.convs[:-1]:
            z = conv(z, edge_index).relu()
        self.tmp_z = self.convs[-1](z, edge_index)
        edge_logit = (self.tmp_z[edge_index[0]] * self.tmp_z[edge_index[1]]).sum(dim=-1)
        edge_weight = torch.nn.Sigmoid()(edge_logit)
        if self.training: self.encode_edge_weight = edge_weight
        # edge sampling
        sampled_index = (edge_weight > SAMPLING_RATIO * torch.rand_like(edge_weight)).detach()
        new_edge_index = edge_index[:, sampled_index]
        # forward with sampled edges
        for conv in self.convs[:-1]:
            x = conv(x, new_edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, x, z, edge_label_index):
        edge_logit = (self.tmp_z[edge_label_index[0]] * self.tmp_z[edge_label_index[1]]).sum(dim=-1)
        pos_weight = torch.nn.Sigmoid()(edge_logit)
        neg_weight = torch.ones_like(pos_weight)
        edge_weight = torch.cat([neg_weight.unsqueeze(1), pos_weight.unsqueeze(1)], dim=1)
        if self.training: self.decode_edge_weight = pos_weight
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits, edge_weight

def getGNNArch_tmp(GNN_name):
    assert GNN_name in ['GCN', 'GAT', 'SAGE']
    if GNN_name == 'GCN':
        return GCN
    elif GNN_name == 'GAT':
        return GAT
    elif GNN_name == 'SAGE':
        return SAGE

def standard_train_trial(rel_path, dataset_name, noise_type, noise_ratio, model_name, num_gnn_layers, device, repeat_times, verbose=True):
    path, dataset = getDataset(dataset_name, device, rel_path)
    Net = getGNNArch_tmp(model_name)
    test_auc_list, val_auc_list, best_epoch_list = [], [], []
    
    MAX_EPOCH = 1000
    if verbose: print(f'==> schedule={args.scheduler}, param={args.scheduler_param}')
    assert args.scheduler in ['linear', 'exp', 'sin', 'cos', 'constant']
    if args.scheduler == 'linear':
        lamb_scheduler = np.linspace(0, 1, MAX_EPOCH) * args.scheduler_param
    elif args.scheduler == 'exp':
        lamb_scheduler = np.array([math.exp(-t/MAX_EPOCH) for t in range(MAX_EPOCH)]) * args.scheduler_param
    elif args.scheduler == 'sin':
        lamb_scheduler = np.array([math.sin(t/MAX_EPOCH * math.pi * 0.5) for t in range(MAX_EPOCH)]) * args.scheduler_param
    elif args.scheduler == 'cos':
        lamb_scheduler = np.array([math.cos(t/MAX_EPOCH * math.pi * 0.5) for t in range(MAX_EPOCH)]) * args.scheduler_param
    elif args.scheduler == 'constant':
        lamb_scheduler = np.array([args.scheduler_param] * MAX_EPOCH)
    
    for idx in tqdm(range(repeat_times), ncols=50, leave=False): 
        savePath = f'{path}/{dataset_name}/processed/{noise_type}_noise_ratio_{noise_ratio}_repeat_{idx+1}.pt'
        data = torch.load(savePath)
        (train_data, val_data, test_data) = data
        model = Net(dataset.num_features, 128, 64, num_gnn_layers).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        def train(epoch):
            model.train()
            optimizer.zero_grad()
            z = model.encode(train_data.x, train_data.edge_index)

            # a new round of negative sampling for every training epoch
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
            edge_label_index = torch.cat(
                [train_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                train_data.edge_label,
                train_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            sample_idx = torch.arange(len(edge_label))
            hidden, out, weight = model.decode(train_data.x, z, edge_label_index)
            out = out.view(-1)

            # loss1: supervised loss
            tmp_loss = criterion(out, edge_label)
            sample_weigtht = weight[sample_idx, edge_label.long()]
            sample_weigtht = sample_weigtht.detach()
            if args.gnn_model in ['GCN', 'SAGE']:
                sup_loss = torch.mean(tmp_loss * sample_weigtht)
            elif args.gnn_model == 'GAT':
                sup_loss = torch.mean(tmp_loss)

            # loss2: information regularizer 
            regu_A = sampling_MI(model.encode_edge_weight, reduction='mean')
            regu_Y = sampling_MI(model.decode_edge_weight, reduction='mean')

            # final objective of RGIB-REP
            lamb = lamb_scheduler[epoch]
            loss = lamb * sup_loss + (1-lamb) * regu_A + (1-lamb) * regu_Y
            
            loss.backward()
            optimizer.step()
            return

        @torch.no_grad()
        def test(data):
            model.eval()
            z = model.encode(data.x, data.edge_index)
            hidden, out, weight = model.decode(data.x, z, data.edge_label_index)
            out = out.view(-1).sigmoid()
            return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

        best_val_auc = best_test_auc = 0
        best_epoch = 0
        time_list = []
        for epoch in range(MAX_EPOCH):
            train(epoch)
            val_auc = test(val_data)
            test_auc = test(test_data)
            
            if val_auc > best_val_auc:
                best_epoch = epoch
                best_val_auc = val_auc
                best_test_auc = test_auc

        # print(best_test_auc, best_val_auc, best_epoch)
        test_auc_list.append(best_test_auc)
        val_auc_list.append(best_val_auc)
        best_epoch_list.append(best_epoch)

    # verbose the training results
    if verbose:
        print(f'==> data={dataset_name}, type={noise_type}, ratio={noise_ratio}, model={model_name}, num_gnn_layers={num_gnn_layers}, repeat_time={repeat_times}')
        print(f'==> VAL:   mean={np.mean(val_auc_list)}, std={np.std(val_auc_list)}, max={np.max(val_auc_list)}, min={np.min(val_auc_list)}')
        print(f'==> TEST:  mean={np.mean(test_auc_list)}, std={np.std(test_auc_list)}, max={np.max(test_auc_list)}, min={np.min(test_auc_list)}')
        print('*'*50)
        
    return np.mean(val_auc_list)
    
### Parse args ###
parser = argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()
print(args)
print('*'*50)

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic =True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.search_scheduler:
    # mutiple trials to search the optimal scheduler
    assert args.search_iteration > 0
    candidate_schduler = ['linear', 'exp', 'sin', 'cos', 'constant']
    schedule_search_space = {
        'scheduler': hp.choice('scheduler', candidate_schduler), 
        'scheduler_param': hp.uniform('scheduler_param', 0, 1),
        }

    def run_model(params):
        args.scheduler = params['scheduler']
        args.scheduler_param = params['scheduler_param']
        mean_val_auc = standard_train_trial(args.rel_path, args.dataset, args.noise_type, args.noise_ratio, args.gnn_model, args.num_gnn_layers, device, args.repeat_times)
        return (1-mean_val_auc)
        
    # searching
    best = fmin(run_model, schedule_search_space, algo=tpe.suggest, max_evals=int(args.search_iteration), verbose=False)
    print(f'==> optimal scheduler:{best}')
    
    # final trial
    args.scheduler = candidate_schduler[best['scheduler']]
    args.scheduler_param = best['scheduler_param']
    standard_train_trial(args.rel_path, args.dataset, args.noise_type, args.noise_ratio, args.gnn_model, args.num_gnn_layers, device, args.repeat_times)
    
else:
    # single trial
    standard_train_trial(args.rel_path, args.dataset, args.noise_type, args.noise_ratio, args.gnn_model, args.num_gnn_layers, device, args.repeat_times)
