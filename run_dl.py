import numpy as np
import pandas as pd
import cfg_target
import pickle
from random import randint
from random import seed
import torch
import model
from joblib import Parallel, delayed
from utils import *
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model_name')
parser.add_argument('--year', type=int, help='year')
parser.add_argument('--month', type=int, help='month')

args = parser.parse_args()
model_name = args.model_name
year = args.year
month_id = args.month


def forecast_dl(month_id, year, rootpath, param_path, device, model_name):
    results = {}
    results['prediction_train'] = []
    results['prediction_test'] = []
    if model_name in ['CNN_LSTM', 'CNN_FNN']:
        train_X = load_results(rootpath + 'train_X_map_{}_forecast{}.pkl'.format(year, month_id))
        test_X = load_results(rootpath + 'test_X_map_{}_forecast{}.pkl'.format(year, month_id))
        train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(year, month_id))
        test_y = load_results(rootpath + 'test_y_pca_{}_forecast{}.pkl'.format(year, month_id))
        output_dim = train_y.shape[-1]
    else:
        train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(year, month_id))
        test_X = load_results(rootpath + 'test_X_pca_{}_forecast{}.pkl'.format(year, month_id))
        train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(year, month_id))
        test_y = load_results(rootpath + 'test_y_pca_{}_forecast{}.pkl'.format(year, month_id))
        # set input and output dim
        input_dim = train_X.shape[-1]
        output_dim = train_y.shape[-1]
    best_parameter = load_results(param_path + '{}_forecast{}.pkl'.format(model_name, month_id))
    if model_name == 'FNN':
        train_X = np.reshape(train_X, (train_X.shape[0], -1))
        test_X = np.reshape(test_X, (test_X.shape[0], -1))
        input_dim = input_dim = train_X.shape[-1]
        train_dataset = model.MapDataset(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)
        curr_hidden_dim = best_parameter['hidden_dim']
        curr_num_layers = best_parameter['num_layers']
        mdl = model.ReluNet(input_dim=input_dim, output_dim=output_dim,
                            hidden_dim=curr_hidden_dim, num_layers=curr_num_layers,
                            threshold=0.1, num_epochs=1000)
    elif model_name == 'CNN_FNN':
        train_dataset = model.MapDataset_CNN(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=False)
        curr_stride = best_parameter['stride']
        curr_kernel_size = best_parameter['kernel_size']
        curr_hidden_dim = best_parameter['hidden_dim']
        curr_num_layers = best_parameter['num_layers']
        num_var = len(train_X)
        input_dim = model.get_input_dim(train_X, num_var, curr_stride, curr_kernel_size)
        mdl = model.CnnFnn(num_var, input_dim, output_dim, kernel_size=curr_kernel_size,
                           stride=curr_stride, hidden_dim=curr_hidden_dim,
                           num_layers=curr_num_layers, num_epochs=100)
    elif model_name == 'CNN_LSTM':
        train_dataset = model.MapDataset_CNN(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=False)
        curr_stride = best_parameter['stride']
        curr_kernel_size = best_parameter['kernel_size']
        curr_hidden_dim = best_parameter['hidden_dim']
        curr_num_layers = best_parameter['num_layers']
        curr_lr = best_parameter['learning_rate']
        curr_num_epochs = best_parameter['num_epochs']
        num_var = len(train_X)
        input_dim = model.get_input_dim(train_X, num_var, curr_stride, curr_kernel_size)
        mdl = model.CnnLSTM(num_var, input_dim, output_dim, kernel_size=curr_kernel_size,
                            stride=curr_stride, hidden_dim=curr_hidden_dim,
                            num_lstm_layers=curr_num_layers, num_epochs=curr_num_epochs,
                            learning_rate=curr_lr)
    model.init_weight(mdl)
    # send model to gpu
    mdl.to(device)
    mdl.fit(train_loader, device)
    state = {'state_dict': mdl.state_dict()}
    torch.save(state, rootpath + 'model/{}_{}_{}.t7'.format(model_name, year, month_id))
    pred_train = mdl.predict(train_X, device)
    pred_test = mdl.predict(test_X, device)
    results['target_train'] = train_y
    results['prediction_train'] = pred_train
    results['target_test'] = test_y
    results['prediction_test'] = pred_test

    save_results(rootpath + 'forecast_results/results_{}_{}_{}.pkl'.format(model_name, year, month_id), results)


# set device for running the code
num_rep = cfg_target.num_rep
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rootpath = cfg_target.forecast_rootpath
param_path = cfg_target.param_path
one_day = cfg_target.one_day

forecast_dl(month_id=month_id, year=year, rootpath=rootpath, param_path=param_path, device=device, model_name=model_name)
# Parallel(n_jobs=12)(delayed(forecast_rep)(month_id,rootpath=rootpath,param_path=param_path, device= device, model_name=model_name,folder_name=folder_name, num_rep= num_rep) for month_id in range(1,13))
