import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import negative_sampling
import random
import argparse
from tqdm import tqdm
from models import *
from utils import *

'''
Uasge of this python scipt, e.g.,
python3 standard-training.py --gnn_model GCN  --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2
python3 standard-training.py --gnn_model GAT  --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2
python3 standard-training.py --gnn_model SAGE --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2
'''

def standard_train_trial(rel_path, dataset_name, noise_type, noise_ratio, model_name, num_gnn_layers, device, repeat_times):
    path, dataset = getDataset(dataset_name, device, rel_path)
    Net = getGNNArch(model_name)
    test_auc_list, val_auc_list, best_epoch_list = [], [], []

    for idx in tqdm(range(repeat_times), ncols=50, leave=False): 
        savePath = f'{path}/{dataset_name}/processed/{noise_type}_noise_ratio_{noise_ratio}_repeat_{idx+1}.pt'
        data = torch.load(savePath)
        (train_data, val_data, test_data) = data
        model = Net(dataset.num_features, 128, 64, num_gnn_layers).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        def train():
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

            hidden, out = model.decode(z, edge_label_index)
            out = out.view(-1)
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()
            return loss

        @torch.no_grad()
        def test(data):
            model.eval()
            z = model.encode(data.x, data.edge_index)
            hidden, out = model.decode(z, data.edge_label_index)
            out = out.view(-1).sigmoid()
            return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

        best_val_auc = best_test_auc = 0
        best_epoch = 0
        for epoch in range(1, 1001):
            loss = train()
            val_auc = test(val_data)
            test_auc = test(test_data)
            
            if val_auc > best_val_auc:
                best_epoch = epoch
                best_val_auc = val_auc
                best_test_auc = test_auc

        test_auc_list.append(best_test_auc)
        val_auc_list.append(best_val_auc)
        best_epoch_list.append(best_epoch)

    # verbose results
    print(f'==> data={dataset_name}, type={noise_type}, ratio={noise_ratio}, model={model_name}, num_gnn_layers={num_gnn_layers}, repeat_time={repeat_times}')
    print(f'==> VAL:   mean={np.mean(val_auc_list)}, std={np.std(val_auc_list)}, max={np.max(val_auc_list)}, min={np.min(val_auc_list)}')
    print(f'==> TEST:  mean={np.mean(test_auc_list)}, std={np.std(test_auc_list)}, max={np.max(test_auc_list)}, min={np.min(test_auc_list)}')
    print('*'*50)
    return
    

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

standard_train_trial(args.rel_path, args.dataset, args.noise_type, args.noise_ratio, args.gnn_model, args.num_gnn_layers, device, args.repeat_times)
