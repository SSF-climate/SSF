#!/usr/bin/env python
# coding: utf-8

# need a function for clarify the spatial range
import os
import sys
os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF/')
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import math
import argparse
import numpy as np
import pandas as pd
import math
import pickle
from random import randint
from random import seed
import torch
import model
import cfg_target
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import argparse
import torch
import torch.nn as nn
from utils import *


def compute_cosine(a, b):
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))


def random_cv(cv_index, cv_year, roothpath, param_grid, num_random, model_name, device, one_day):
    # load data
    if model_name in ['CNN_LSTM', 'CNN_FNN']:
        train_X = load_results(rootpath + 'train_X_map_{}_forecast{}.pkl'.format(cv_year, cv_index))
        valid_X = load_results(rootpath + 'val_X_map_{}_forecast{}.pkl'.format(cv_year, cv_index))
        train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
        valid_y = load_results(rootpath + 'val_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
        output_dim = train_y.shape[-1]
    else:
        train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
        valid_X = load_results(rootpath + 'val_X_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
        train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
        valid_y = load_results(rootpath + 'val_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
        # set input and output dim
        input_dim = train_X.shape[-1]
        output_dim = train_y.shape[-1]

    if model_name == 'EncoderFNN_AllSeq_AR_CI' or model_name == 'EncoderFNN_AllSeq_AR':
        hidden_dim = param_grid['hidden_dim']
        num_layers = param_grid['num_layers']
        lr = param_grid['learning_rate']
        threshold = param_grid['threshold']
        num_epochs = param_grid['num_epochs']
        seq_len = param_grid['seq_len']
        linear_dim = param_grid['linear_dim']
        drop_out = param_grid['drop_out']
        if model_name == 'EncoderFNN_AllSeq_AR_CI':
            ci_dim = param_grid['ci_dim']

        train_y_ar = load_results(rootpath + 'train_y_pca_ar_{}_forecast{}.pkl'.format(cv_year, cv_index))
        valid_y_ar = load_results(rootpath + 'val_y_pca_ar_{}_forecast{}.pkl'.format(cv_year, cv_index))
        train_dataset = model.MapDataset_ar(train_X, train_y_ar, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)

    elif model_name == 'EncoderDecoder' or model_name == 'EncoderFNN_AllSeq' or model_name == 'EncoderFNN':
        train_dataset = model.MapDataset(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)
        hidden_dim = param_grid['hidden_dim']
        num_layers = param_grid['num_layers']
        lr = param_grid['learning_rate']
        threshold = param_grid['threshold']
        num_epochs = param_grid['num_epochs']
        if model_name == 'EncoderDecoder':
            decoder_len = param_grid['decoder_len']
        elif model_name == 'EncoderFNN':
            last_layer = param_grid['last_layer']
            seq_len = param_grid['seq_len']
        elif model_name == 'EncoderFNN_AllSeq':
            seq_len = param_grid['seq_len']
            linear_dim = param_grid['linear_dim']
            drop_out = param_grid['drop_out']
    elif model_name == 'XGBoost':
        if one_day = True:
            train_X = train_X[:,-1,:] # one day
            valid_X = valid_X[:,-1,:] # one day
        train_X = np.reshape(train_X,(train_X.shape[0],-1))
        valid_X = np.reshape(valid_X,(valid_X.shape[0],-1))
        max_depth = param_grid['max_depth']
        colsample_bytree = param_grid['colsample_bytree']
        gamma = param_grid['gamma']
        n_estimators = param_grid['n_estimators']
        lr = param_grid['learning_rate']
    elif model_name == 'Lasso':
        if one_day = True:
            train_X = train_X[:,-1,:] # one day
            valid_X = valid_X[:,-1,:] # one day
        train_X = np.reshape(train_X,(train_X.shape[0],-1))
        valid_X = np.reshape(valid_X,(valid_X.shape[0],-1))
        alphas = param_grid['alpha']
    elif model_name == 'FNN':
        # train_X = train_X[:,-1,:] # one day
        # valid_X = valid_X[:,-1,:] # one day
        train_X = np.reshape(train_X,(train_X.shape[0],-1))
        valid_X = np.reshape(valid_X,(valid_X.shape[0],-1))
        train_dataset = model.MapDataset(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)
        hidden_dim = param_grid['hidden_dim']
        num_layers = param_grid['num_layers']
    elif model_name == 'CNN_FNN':
        train_dataset = model.MapDataset_CNN(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=False)
        stride = param_grid['stride']
        kernel_size = param_grid['kernel_size']
        hidden_dim = param_grid['hidden_dim']
        num_layers = param_grid['num_layers']
    elif model_name == 'CNN_LSTM':
        train_dataset = model.MapDataset_CNN(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=False)
        stride = param_grid['module__stride']
        kernel_size = param_grid['module__kernel_size']
        hidden_dim = param_grid['module__hidden_dim']
        num_lstm_layers = param_grid['module__num_lstm_layers']
        lr = param_grid['lr']
        num_epochs = param_grid['module__num_epochs']
    else:
        print('the model name is not in the list')


    history_all = []
    score = []
    parameter_all = []
    for i in range(num_random):
        # set model
        if model_name == 'EncoderDecoder':
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layer = num_layers[randint(0, len(num_layers) - 1)]
            curr_decoder_len = decoder_len[randint(0, len(decoder_len) - 1)]
            curr_threshold = threshold[randint(0, len(threshold) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            curr_num_epochs = num_epochs[randint(0, len(num_epochs) - 1)]
            parameters = {'hidden_dim': curr_hidden_dim, 'num_layers': curr_num_layer, 'decoder_len': curr_decoder_len, 'threshold': curr_threshold, 'learning_rate': curr_lr, 'num_epochs': curr_num_epochs}
            parameter_all.append(parameters)
            mdl = model.EncoderDecoder(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim, num_layers=curr_num_layer, learning_rate=curr_lr, decoder_len=curr_decoder_len, threshold=curr_threshold, num_epochs=curr_num_epochs)

            # initialize the model
            model.init_weight(mdl)

            # send model to gpu
            mdl.to(device)
            # fit the model
            history = mdl.fit_cv(train_loader, valid_X, valid_y, device)
            # compute the prediction of validation set
            pred_y = mdl.predict(valid_X, device)
        elif model_name == 'EncoderFNN':
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layer = num_layers[randint(0, len(num_layers) - 1)]
            curr_seq_len = seq_len[randint(0, len(seq_len) - 1)]
            curr_threshold = threshold[randint(0, len(threshold) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            curr_num_epochs = num_epochs[randint(0, len(num_epochs) - 1)]
            curr_last_layer = last_layer[randint(0, len(last_layer) - 1)]
            parameters = {'hidden_dim': curr_hidden_dim, 'num_layers': curr_num_layer, 'last_layer': curr_last_layer, 'threshold': curr_threshold, 'learning_rate': curr_lr, 'num_epochs': curr_num_epochs, 'seq_len': curr_seq_len}
            parameter_all.append(parameters)
            mdl = model.EncoderFNN(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim, num_layers=curr_num_layer, last_layer=curr_last_layer, seq_len=curr_seq_len, learning_rate=curr_lr, threshold=curr_threshold, num_epochs=curr_num_epochs)
            # initialize the model
            model.init_weight(mdl)

            # send model to gpu
            mdl.to(device)
            # fit the model
            history = mdl.fit_cv(train_loader, valid_X, valid_y, device)
            # compute the prediction of validation set
            pred_y = mdl.predict(valid_X, device)
        elif model_name == 'EncoderFNN_AllSeq':
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layer = num_layers[randint(0, len(num_layers) - 1)]
            curr_seq_len = seq_len[randint(0, len(seq_len) - 1)]
            curr_threshold = threshold[randint(0, len(threshold) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            curr_num_epochs = num_epochs[randint(0, len(num_epochs) - 1)]
            curr_linear_dim = linear_dim[randint(0, len(linear_dim) - 1)]
            curr_drop_out = drop_out[randint(0, len(drop_out) - 1)]
            parameters = {'hidden_dim': curr_hidden_dim, 'num_layers': curr_num_layer, 'linear_dim': curr_linear_dim, 'threshold': curr_threshold,
                          'learning_rate': curr_lr, 'num_epochs': curr_num_epochs, 'seq_len': curr_seq_len, 'drop_out': curr_drop_out}
            parameter_all.append(parameters)

            mdl = model.EncoderFNN_AllSeq(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim, num_layers=curr_num_layer,
                                          seq_len=curr_seq_len, linear_dim=curr_linear_dim, learning_rate=curr_lr, dropout=curr_drop_out,
                                          threshold=curr_threshold, num_epochs=curr_num_epochs)
            # initialize the model
            model.init_weight(mdl)

            # send model to gpu
            mdl.to(device)
            # fit the model
            history = mdl.fit_cv(train_loader, valid_X, valid_y, device)
            # compute the prediction of validation set
            pred_y = mdl.predict(valid_X, device)
        elif model_name == 'EncoderFNN_AllSeq_AR':
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layer = num_layers[randint(0, len(num_layers) - 1)]
            curr_seq_len = seq_len[randint(0, len(seq_len) - 1)]
            curr_threshold = threshold[randint(0, len(threshold) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            curr_num_epochs = num_epochs[randint(0, len(num_epochs) - 1)]
            curr_linear_dim = linear_dim[randint(0, len(linear_dim) - 1)]
            curr_drop_out = drop_out[randint(0, len(drop_out) - 1)]
            parameters = {'hidden_dim': curr_hidden_dim, 'num_layers': curr_num_layer, 'linear_dim': curr_linear_dim, 'threshold': curr_threshold,
                          'learning_rate': curr_lr, 'num_epochs': curr_num_epochs, 'seq_len': curr_seq_len, 'drop_out': curr_drop_out}
            parameter_all.append(parameters)

            mdl = model.EncoderFNN_AllSeq_AR(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim, num_layers=curr_num_layer,
                                             seq_len=curr_seq_len, linear_dim=curr_linear_dim, learning_rate=curr_lr, dropout=curr_drop_out,
                                             threshold=curr_threshold, num_epochs=curr_num_epochs)
            # initialize the model
            model.init_weight(mdl)

            # send model to gpu
            mdl.to(device)
            # fit the model
            history = mdl.fit_cv(train_loader, valid_X, valid_y_ar, valid_y, device)
            # compute the prediction of validation set
            pred_y = mdl.predict(valid_X, valid_y_ar, device)
        elif model_name == 'EncoderFNN_AllSeq_AR_CI':
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layer = num_layers[randint(0, len(num_layers) - 1)]
            curr_seq_len = seq_len[randint(0, len(seq_len) - 1)]
            curr_threshold = threshold[randint(0, len(threshold) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            curr_num_epochs = num_epochs[randint(0, len(num_epochs) - 1)]
            curr_linear_dim = linear_dim[randint(0, len(linear_dim) - 1)]
            curr_drop_out = drop_out[randint(0, len(drop_out) - 1)]
            parameters = {'hidden_dim': curr_hidden_dim, 'num_layers': curr_num_layer, 'linear_dim': curr_linear_dim, 'threshold': curr_threshold,
                          'learning_rate': curr_lr, 'num_epochs': curr_num_epochs, 'seq_len': curr_seq_len, 'drop_out': curr_drop_out, 'ci_dim': ci_dim}
            parameter_all.append(parameters)

            mdl = model.EncoderFNN_AllSeq_AR_CI(input_dim=input_dim - ci_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim, num_layers=curr_num_layer,
                                                seq_len=curr_seq_len, linear_dim=curr_linear_dim, ci_dim=ci_dim, learning_rate=curr_lr, dropout=curr_drop_out,
                                                threshold=curr_threshold, num_epochs=curr_num_epochs)
            # initialize the model
            model.init_weight(mdl)

            # send model to gpu
            mdl.to(device)
            # fit the model
            history = mdl.fit_cv(train_loader, valid_X, valid_y_ar, valid_y, device)
            pred_y = mdl.predict(valid_X, valid_y_ar, device)
        elif model_name == 'XGBoost':
            curr_max_depth = max_depth[randint(0, len(max_depth) - 1)]
            curr_colsample_bytree = colsample_bytree[randint(0, len(colsample_bytree) - 1)]
            curr_gamma = gamma[randint(0, len(gamma) - 1)]
            curr_n_estimators = n_estimators[randint(0, len(n_estimators) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            parameters = {'max_depth': curr_max_depth, 'colsample_bytree': curr_colsample_bytree,
                          'gamma': curr_gamma, 'n_estimators': curr_n_estimators,
                          'learning_rate': curr_lr}
            parameter_all.append(parameters)
            mdl = model.XGBMultitask(num_models=output_dim, colsample_bytree=curr_colsample_bytree,
                                     gamma=curr_gamma, learning_rate=curr_lr, max_depth=curr_max_depth,
                                     n_estimators=curr_n_estimators, objective='reg:squarederror')
            # history = mdl.fit_cv(train_X, train_y, valid_X, valid_y)
            mdl.fit(train_X, train_y)
            pred_y = mdl.predict(valid_X)
            history = None
        elif model_name == 'Lasso':
            curr_alpha = alphas[randint(0, len(alphas) - 1)]
            parameter = {'alpha': curr_alpha}
            parameter_all.append(parameter)
            mdl = model.LassoMultitask(alpha=curr_alpha, fit_intercept=False)
            mdl.fit(train_X, train_y)
            pred_y = mdl.predict(valid_X)
            history = None
        elif model_name == 'FNN':
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layers = num_layers[randint(0, len(num_layers) - 1)]
            parameters = {'hidden_dim': curr_hidden_dim, 'num_layers': curr_num_layers}
            parameter_all.append(parameters)
            mdl = model.ReluNet(input_dim=input_dim, output_dim=output_dim,
                                hidden_dim=curr_hidden_dim, num_layers=curr_num_layers,
                                threshold=0.1, num_epochs=1000)
            model.init_weight(mdl)
            mdl.to(device)
            history = mdl.fit_cv(train_loader, valid_X, valid_y, device)
            pred_y = mdl.predict(valid_X, device)
        elif model_name == 'CNN_FNN':
            curr_stride = stride[randint(0, len(stride) - 1)]
            curr_kernel_size = kernel_size[randint(0, len(kernel_size) - 1)]
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layers = num_layers[randint(0, len(num_layers) - 1)]
            parameters = {'stride': curr_stride, 'kernel_size': curr_kernel_size,
                          'hidden_dim': curr_hidden_dim, 'num_layers': curr_num_layers}
            parameter_all.append(parameters)
            num_var = len(train_X)
            input_dim = model.get_input_dim(train_X, num_var, curr_stride, curr_kernel_size)
            mdl = model.CnnFnn(num_var, input_dim, output_dim, kernel_size=curr_kernel_size,
                                stride=curr_stride, hidden_dim=curr_hidden_dim,
                                num_layers=curr_num_layers, num_epochs=100)
            mdl.to(device)
            history = mdl.fit_cv(train_loader, valid_X, valid_y, device)
            pred_y = mdl.predict(valid_X, device)
        elif model_name == 'CNN_LSTM':
            curr_stride = stride[randint(0, len(stride) - 1)]
            curr_kernel_size = kernel_size[randint(0, len(kernel_size) - 1)]
            curr_hidden_dim = hidden_dim[randint(0, len(hidden_dim) - 1)]
            curr_num_layers = num_lstm_layers[randint(0, len(num_lstm_layers) - 1)]
            curr_lr = lr[randint(0, len(lr) - 1)]
            curr_num_epochs = num_epochs[randint(0, len(num_epochs) - 1)]
            parameters = {'stride': curr_stride, 'kernel_size': curr_kernel_size, 'hidden_dim': curr_hidden_dim,
                          'num_layers': curr_num_layers, 'learning_rate': curr_lr, 'num_epochs': curr_num_epochs}
            parameter_all.append(parameters)
            num_var = len(train_X)
            input_dim = model.get_input_dim(train_X, num_var, curr_stride, curr_kernel_size)
            mdl = model.CnnLSTM(num_var, input_dim, output_dim, kernel_size=curr_kernel_size,
                                 stride=curr_stride, hidden_dim=curr_hidden_dim,
                                 num_lstm_layers=curr_num_layers, num_epochs=curr_num_epochs,
                                 learning_rate=curr_lr)
            mdl.to(device)
            history = mdl.fit_cv(train_loader, valid_X, valid_y, device)
            pred_y = mdl.predict(valid_X, device)

        history_all.append(history)
        test_rmse = np.sqrt(((valid_y - pred_y)**2).mean())
        test_cos = np.asarray([compute_cosine(valid_y[i, :], pred_y[i, :]) for i in range(len(valid_y))]).mean()
        score.append([test_rmse, test_cos])

    cv_results = {'score': score, 'parameter_all': parameter_all, 'history_all': history_all}
    save_results(rootpath + 'cv_results_test/cv_results_' + model_name + '_{}_{}.pkl'.format(cv_year, cv_index), cv_results)


# set device for running the code

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


seed(314)

# param_grid = cfg_target.param_grid_en_de
month_range = cfg_target.month_range
val_years = cfg_target.val_years
num_random = cfg_target.num_random
rootpath = cfg_target.rootpath_cv
one_day = cfg_target.one_day

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2012, help='the year selected for hyper parameter tuning')
parser.add_argument('--model_name', type=str, default='EncoderFNN', help='the model used for hyper parameter tuning')

args = parser.parse_args()
year = args.year
model_name = args.model_name
# print(model_name)
if model_name == 'XGBoost':
    param_grid = cfg_target.param_grid_xgb
elif model_name == 'Lasso':
    param_grid = cfg_target.param_grid_lasso
elif model_name == 'FNN':
    param_grid = cfg_target.param_grid_fnn
elif model_name == 'CNN_FNN':
    param_grid = cfg_target.param_grid_cnn_fnn
elif model_name == 'CNN_LSTM':
    param_grid = cfg_target.param_grid_cnn_lstm
elif model_name in ['EncoderFNN_AllSeq_AR_CI', 'EncoderFNN_AllSeq_AR','EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']:
    param_grid = cfg_target.param_grid_en_de
else:
    print('can not find the model')

#
Parallel(n_jobs=12)(delayed(random_cv)(cv_index, cv_year=year, roothpath=rootpath, param_grid=param_grid, num_random=num_random, model_name=model_name, device=device, one_day=one_day) for cv_index in month_range)
