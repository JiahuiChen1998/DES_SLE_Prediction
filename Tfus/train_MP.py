
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import yaml
import numpy as np
import torch
import shutil
import os
import time
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import random
import pandas as pd
from model import Transformer,AttentiveFP
from dataset import SMILES_dataset,Graph_dataset
from tokenizer import SMILESTokenizer




def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def load_pre_trained_weights(model):
    try:
        state_dict = torch.load(config['fine_tune_from'], map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
    return model


def train(model, loader, optimizer,norm=False):
    model.train()
    loss_all = []
    all_predictions = []  
    all_targets = []  
    
    if norm:
        with torch.set_grad_enabled(True):
            for datas,label in loader:
                optimizer.zero_grad()

                if  config['model_type'] == 'gnn':
                    datas,label = datas.to(device),label.to(device)
                    label = normalizer.norm(label)
                    output = model(datas)

                elif config['model_type'] == 'tf':

                    data = [data.to(device) for data in datas]

                    label = label.to(device)
                    label = normalizer.norm(label)
                    output = model(data)

                label = torch.squeeze(normalizer.denorm(label))
                output = torch.squeeze(normalizer.denorm(output))

                loss = F.mse_loss(output, label)

                loss.backward()
                optimizer.step()

                loss_all.append(loss.item())
                all_predictions.append(output.detach().cpu())
                all_targets.append(label.detach().cpu())

    else:
        with torch.set_grad_enabled(True):
            for datas,label in loader:
                optimizer.zero_grad()

                if  config['model_type'] == 'gnn':
                    datas,label = datas.to(device),torch.squeeze(label.to(device))
                    output = torch.squeeze(model(datas))

                elif config['model_type'] == 'tf':
                    data = [data.to(device) for data in datas] 


                    label = torch.squeeze(label.to(device))
                    output = torch.squeeze(model(data))

                loss = F.mse_loss(output, label)

                loss.backward()
                optimizer.step()

                
                loss_all.append(loss.item())
                all_predictions.append(output.detach().cpu())
                all_targets.append(label.detach().cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mae_all = F.l1_loss(all_predictions, all_targets).item()
    r2_all = r2_score(all_predictions.numpy(), all_targets.numpy())
    mse_all = F.mse_loss(all_predictions, all_targets).item()

    loss = np.average(loss_all)
    mae = mae_all
    r2 = r2_all
    rmse = math.sqrt(mse_all)

    return loss, mae, r2, rmse


def save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('config_fusH.yaml', os.path.join(model_checkpoints_folder, 'config_fusH.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def evaluate_dataset(model, dataset, shuffle,name=None, norm=False, save=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

    y, pred = [], []

    if norm:
        with torch.no_grad():
            for datas, label in loader:
                if config['model_type'] == 'gnn':
                    datas, label = datas.to(device), label.to(device)
                    label = normalizer.norm(label)
                    output = model(datas)

                elif config['model_type'] == 'tf':
                    data = [data.to(device) for data in datas]
                    label = label.to(device)
                    label = normalizer.norm(label)
                    output = model(data)

                label = normalizer.denorm(label)
                output = normalizer.denorm(output)

                y.extend(label.detach().cpu().numpy())
                pred.extend(output.detach().cpu().numpy())

    else:
        with torch.no_grad():
            for datas, label in loader:
                if config['model_type'] == 'gnn':
                    datas, label = datas.to(device), label.to(device)
                    output = model(datas)

                elif config['model_type'] == 'tf':
                    data = [data.to(device) for data in datas]
                    label = label.to(device)
                    output = model(data)

                y.extend(label.detach().cpu().numpy())
                pred.extend(output.detach().cpu().numpy())


    y, pred = np.array(y).flatten(), np.array(pred).flatten()
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    mse = mean_squared_error(y, pred)

    if save:
        df = pd.DataFrame({'Experimental Value': y, 'Predicted Value': pred})
        df.to_csv(f'training_results/{name}_predictions_{model_type}.csv', index=False)

    return mae, rmse, r2, mse


def calculate_metrics(csv_path):

    data = pd.read_csv(csv_path)
    experimental_values = []
    predicted_values = []

    for i in range(0, len(data), 10):
        predicted_value = data['Predicted Value'][i:i+10]
        experimental_value = data['Experimental Value'][i:i+10]

        if len(set(experimental_value)) == 1:
            experimental_value = experimental_value.sample(n=1).values[0]
        else:
            print("Error: Experimental values are not the same in this group!")
            continue

        predicted_value_mean = np.mean(predicted_value)

        experimental_values.append(experimental_value)
        predicted_values.append(predicted_value_mean)

    mae = mean_absolute_error(experimental_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(experimental_values, predicted_values))
    r2 = r2_score(experimental_values, predicted_values)
    return mae, rmse, r2


def train_and_test(model, train_dataset,test_dataset, config):

    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(test_dataset))


    if config['model_type'] == 'gnn':
        model = AttentiveFP(**config["gnn"]).to(device)
    elif config['model_type'] == 'tf':
        model = Transformer(**config["transformer"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    early_stopping_count = 0
    epoch_times = []

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        model.train()
        train_loss, _, train_r2, _ = train(model, train_loader, optimizer, norm=config['norm'])

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        print(f"Epoch{epoch:3d},Time:{epoch_time:.2f}s,TrainLoss:{train_loss:.6f},TrainR2:{train_r2:.6f}")

    model.load_state_dict(torch.load(f'model_weight/{model_type}/best_{model_type}_model.pth'))
    train_mae, train_rmse, train_r2,_= evaluate_dataset(model, train_dataset, name='traindata',shuffle=False,norm=config['norm'],save=True)
    test_mae, test_rmse, test_r2,_= evaluate_dataset(model, test_dataset, name='testdata',shuffle=False,norm=config['norm'],save=True)


    if config['DA'] == True:
        train_mae,train_rmse,train_r2=calculate_metrics(f'training_results/traindata_predictions_{model_type}.csv')
        test_mae, test_rmse, test_r2 = calculate_metrics(f'training_results/testdata_predictions_{model_type}.csv')
        print(f'Best_Epoch:{epoch - early_stopping_count},Train_MAE: {train_mae:.6f}, Train_RMSE: {train_rmse:.6f},Train_R2: {train_r2:.6f},Test_MAE: {test_mae:.6f}, Test_RMSE: {test_rmse:.6f}, Test_R2: {test_r2:.6f}')
    else:
        print(f'Best_Epoch:{epoch - early_stopping_count},Train_MAE: {train_mae:.6f}, Train_RMSE: {train_rmse:.6f},Train_R2: {train_r2:.6f},Test_MAE: {test_mae:.6f}, Test_RMSE: {test_rmse:.6f}, Test_R2: {test_r2:.6f}')



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on:',device)
    print('---loading dataset---')

    config = yaml.load(open("config_MP.yaml", "r"), Loader=yaml.FullLoader)
    setup_seed(config['seed'])
    print(config)

    target = 'MP_K'
    if config['DA'] == True:
        df1 = pd.read_csv("MP_data/expanded_train_data.csv")
        df2 = pd.read_csv("MP_data/expanded_test_data.csv")
        print("Using Data Augmentation!")
    else:
        df1 = pd.read_csv("MP_data/train_MP.csv")
        df2 = pd.read_csv("MP_data/test_MP.csv")
        print("Not Using Data Augmentation!")


    if  config['model_type'] == 'gnn':

        train_dataset = Graph_dataset(df1, target=target)
        test_dataset = Graph_dataset(df2, target=target)

        model = AttentiveFP(**config["gnn"]).to(device)
        model = load_pre_trained_weights(model)
        model_type = config['model_type']
        print('model_type:', 'GNN')

    elif config['model_type'] == 'tf':

        train_dataset = SMILES_dataset(df = df1, tokenizer = SMILESTokenizer(config['vocab_path'], model_max_length = 128, padding_side='right'),target=target)
        test_dataset = SMILES_dataset(df = df2, tokenizer = SMILESTokenizer(config['vocab_path'], model_max_length = 128, padding_side='right'),target=target)

        model = Transformer(**config["transformer"]).to(device)
        model = load_pre_trained_weights(model)
        model_type = config['model_type']
        print('model_type:', 'Transformer')


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    
    labels = []
    for data,label in loader:
        labels.append(label)
    labels = torch.cat(labels)
    normalizer = Normalizer(labels)
    print(normalizer.mean, normalizer.std, labels.shape)

    train_and_test(model,train_dataset,test_dataset,config)


