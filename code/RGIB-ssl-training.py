import random
import math
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
import GCL.augmentors as A
from GCL.augmentors.functional import add_edge
import models
from loss import *
from utils import *
from hyperopt import hp, fmin, tpe

'''
Uasge of this python scipt, e.g.,
python3 RGIB-ssl-training.py --gnn_model GCN --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2 --scheduler linear --scheduler_param 1.0
python3 RGIB-ssl-training.py --gnn_model GCN --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2 --search_scheduler --search_iteration 50
'''

def generate_augmentation_operator(n=2):
    search_space = [
    (A.Identity, ()),
    (A.FeatureMasking, (0.0, 0.3)),
    (A.FeatureDropout, (0.0, 0.3)),
    (A.EdgeRemoving, (0.0, 0.5))
    ]
    
    operator_list = []
    index = list(range(len(search_space)))
    random.shuffle(index)
    sampled_index = index[:n]
    for idx in sampled_index:
        opt, hp_range = search_space[idx]
        if hp_range == ():
            operator_list.append(opt())
        else:
            sampled_hp = random.uniform(hp_range[0], hp_range[1])
            operator_list.append(opt(sampled_hp))

    aug = A.Compose(operator_list)
    return aug

def standard_train_trial(rel_path, dataset_name, noise_type, noise_ratio, model_name, num_gnn_layers, device, repeat_times, verbose=True):
    path, dataset = getDataset(dataset_name, device, rel_path)
    Net = getGNNArch(model_name)
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
        cl_criterion = SelfAdversarialClLoss()
        margin, use_robust_loss, start_fine = 5, 0, True
        
        def train(epoch_idx):
            aug1 = generate_augmentation_operator()
            aug2 = generate_augmentation_operator()
            
            lamb1 = lamb_scheduler[epoch_idx]
            lamb2, lamb3 = 1-lamb1, 1-lamb1
            
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

            model.train()
            optimizer.zero_grad()

            # forward with original graph
            z = model.encode(train_data.x, train_data.edge_index)
            hidden, out = model.decode(z, edge_label_index)
            out = out.view(-1)

            # forward with original augmented graph1
            x1, edge_index1, _ = aug1(train_data.x, train_data.edge_index)
            z1 = model.encode(x1, edge_index1)
            hidden1, out1 = model.decode(z1, edge_label_index)
            out1 = out1.view(-1)

            # forward with original augmented graph2
            x2, edge_index2, _ = aug2(train_data.x, train_data.edge_index)
            z2 = model.encode(x2, edge_index2)
            hidden2, out2 = model.decode(z2, edge_label_index)
            out2 = out2.view(-1)

            # loss1: supervised loss with original graph
            sup_loss_ori = criterion(out, edge_label).mean()
 
            # loss2: supervised loss with augmentated graphs
            sup_loss_aug = (criterion(out1, edge_label) + criterion(out2, edge_label)).mean() if lamb1 > 0 else 0
            
            # loss3: self-supervised loss with original graphs
            h1 = torch.cat([hidden, hidden], dim=0)
            h2 = torch.cat([hidden, hidden.flip([0])], dim=0)
            cl_label = torch.cat([torch.ones(hidden.size(0)), torch.zeros(hidden.size(0))], dim=0).to(device)
            pair_dist = F.pairwise_distance(h1, h2)
            ssl_loss_ori = cl_criterion(pair_dist, cl_label, margin, use_robust_loss, start_fine) if lamb2 > 0 else 0

            # loss4: self-supervised loss with augmentated graphs
            h1 = torch.cat([hidden1, hidden1], dim=0)
            h2 = torch.cat([hidden2, hidden2.flip([0])], dim=0)
            cl_label = torch.cat([torch.ones(hidden1.size(0)), torch.zeros(hidden1.size(0))], dim=0).to(device)
            pair_dist = F.pairwise_distance(h1, h2)
            ssl_loss_aug = cl_criterion(pair_dist, cl_label, margin, use_robust_loss, start_fine) if lamb2 > 0 else 0
            
            # loss5: uniformity I(H, H')
            batchsize = 1024
            tmp_cand = torch.randperm(len(edge_label))
            sampled_index = tmp_cand[:batchsize] # boost uniformity with data sampling
            uniformity_loss = lunif(hidden[sampled_index, :]) if lamb3 > 0 else 0 
            
            # final objective of RGIB-SSL
            sup_loss = lamb1 * (sup_loss_ori + sup_loss_aug)
            align_loss = lamb2 * (ssl_loss_ori + ssl_loss_aug) 
            uni_loss = lamb3 * uniformity_loss
            loss = sup_loss + align_loss + uni_loss
            
            loss.backward()
            optimizer.step()
            return

        @torch.no_grad()
        def test(data):
            model.eval()
            z = model.encode(data.x, data.edge_index)
            hidden, out = model.decode(z, data.edge_label_index)
            out = out.view(-1).sigmoid()
            return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

        best_val_auc = best_test_auc = 0
        best_epoch = 0
        time_list = []
        for epoch in range(MAX_EPOCH):
            train(epoch_idx=epoch)
            val_auc = test(val_data)
            test_auc = test(test_data)
            
            if val_auc > best_val_auc:
                best_epoch = epoch
                best_val_auc = val_auc
                best_test_auc = test_auc
            
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
