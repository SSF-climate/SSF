#!/usr/bin/env python
# coding: utf-8

# need a function for clarify the spatial range
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
import os

import model
# from multitask import *
import cfg_target
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import argparse
import torch
import torch.nn as nn


def load_results(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save_results(filename, results):
    with open(filename, 'wb') as fh:
        pickle.dump(results, fh)


def compute_cosine(a, b):
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))


def random_cv(cv_index, cv_year, roothpath, param_grid, num_random, model_name, device):
    # load data
    train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
    valid_X = load_results(rootpath + 'val_X_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
    train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))
    valid_y = load_results(rootpath + 'val_y_pca_{}_forecast{}.pkl'.format(cv_year, cv_index))

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
    # set input and output dim
    input_dim = train_X.shape[-1]
    output_dim = train_y.shape[-1]
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

        history_all.append(history)

        test_rmse = np.sqrt(((valid_y - pred_y)**2).mean())
        test_cos = np.asarray([compute_cosine(valid_y[i, :], pred_y[i, :]) for i in range(len(valid_y))]).mean()
        score.append([test_rmse, test_cos])

    cv_results = {'score': score, 'parameter_all': parameter_all, 'history_all': history_all}
    save_results(rootpath + 'cv_results_test/cv_results_' + model_name + '_{}_{}.pkl'.format(cv_year, cv_index), cv_results)


def best_hyperparameter(val_years, month_range, eval_metrics, model_name, rootpath):
    for month in month_range:
        score_all = []
        for year in val_years:
            cv_results = load_results(rootpath + 'cv_results_test/cv_results_' + model_name + '_{}_{}.pkl'.format(year, month))
            score = cv_results['score']
            score_all.append(score)
        score_all = np.asarray(score_all).squeeze()

        if eval_metrics == 'cos':
            best_score = score_all[:, :, 1].mean(axis=0)
            best_id = np.where(best_score == best_score.max())
        elif eval_metrics == 'rmse':
            best_score = score_all[:, :, 0].mean(axis=0)
            best_id = np.where(best_score == best_score.min())

        best_parameter = cv_results['parameter_all'][best_id[0][0]]

        save_results(rootpath + 'cv_results_test/best_parameter/{}_forecast{}.pkl'.format(model_name, month), best_parameter)


# set device for running the code

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


seed(314)
# for cross-validation
# param_grid = {'hidden_dim': [10, 20, 40, 60, 150, 200],
#               'num_layers': [2, 3, 4, 5, 6],
#               'learning_rate': [0.05, 0.01, 0.005, 0.001],
#               'threshold': [0.5, 0.6],
#               'num_epochs': [20, 50],  # [100, 200, 300],
#               'decoder_len': [4, 11, 18],
#               'last_layer': [True, False],
#               'seq_len': [4, 11, 18],
#               'linear_dim': [50, 100, 200],
#               'drop_out': [0.1, 0.2]}

param_grid = cfg_target.param_grid
month_range = cfg_target.month_range
val_years = cfg_target.val_years
num_random = cfg_target.num_random
rootpath = cfg_target.rootpath_cv


parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2012, help='the year selected for hyper parameter tuning')
parser.add_argument('--model_name', type=str, default='EncoderFNN', help='the model used for hyper parameter tuning')

args = parser.parse_args()
year = args.year
model_name = args.model_name

# model_names = ['EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']
# model_names = ['EncoderFNN_AllSeq']
# num_random = 4
# year = 2012
# val_years = [2013]
# month_range = [1]

Parallel(n_jobs=12)(delayed(random_cv)(cv_index, cv_year=year, roothpath=rootpath, param_grid=param_grid, num_random=num_random, model_name=model_name, device=device) for cv_index in month_range)
