import sys
import os
os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF/')
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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model_name')
parser.add_argument('--year', type=int, help='year')
parser.add_argument('--month', type=int, help='month')

args = parser.parse_args()
model_name = args.model_name
year = args.year
month_id = args.month


def forecast_non_dl(month_id, year, rootpath, param_path, model_name, one_day):
    results = {}
    train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_X = load_results(rootpath + 'test_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_y = load_results(rootpath + 'test_y_pca_{}_forecast{}.pkl'.format(year, month_id))
    if one_day is True:
        train_X = train_X[:, -1, :]
        test_X = test_X[:, -1, :]
    train_X = np.reshape(train_X, (train_X.shape[0], -1))
    test_X = np.reshape(test_X, (test_X.shape[0], -1))

    input_dim = train_X.shape[-1]
    output_dim = train_y.shape[-1]
    # ar_dim = train_X.shape[1]
    best_parameter = load_results(param_path + '{}_forecast{}.pkl'.format(model_name, month_id))
    if model_name == 'XGBoost':
        curr_max_depth = best_parameter['max_depth']
        curr_colsample_bytree = best_parameter['colsample_bytree']
        curr_gamma = best_parameter['gamma']
        curr_n_estimators = best_parameter['n_estimators']
        curr_lr = best_parameter['learning_rate']
        mdl = model.XGBMultitask(num_models=output_dim, colsample_bytree=curr_colsample_bytree,
                                 gamma=curr_gamma, learning_rate=curr_lr, max_depth=curr_max_depth,
                                 n_estimators=curr_n_estimators, objective='reg:squarederror')

    elif model_name == 'Lasso':
        curr_alpha = best_parameter['alpha']
        mdl = model.LassoMultitask(alpha=curr_alpha, fit_intercept=False)

    mdl.fit(train_X, train_y)
    # send model to gpu
    pred_train = mdl.predict(train_X)
    pred_test = mdl.predict(test_X)
    results['target_train'] = train_y
    results['prediction_train'] = pred_train
    results['target_test'] = test_y
    results['prediction_test'] = pred_test
    save_results(rootpath + 'forecast_results/results_{}_{}_{}.pkl'.format(model_name, year, month_id), results)


# set device for running the code
rootpath = cfg_target.forecast_rootpath
param_path = cfg_target.param_path
one_day = cfg_target.one_day

forecast_non_dl(month_id=month_id, year=year, rootpath=rootpath, param_path=param_path, model_name=model_name, one_day=one_day)
# Parallel(n_jobs=12)(delayed(forecast_rep)(month_id,rootpath=rootpath,param_path=param_path, device= device, model_name=model_name,folder_name=folder_name, num_rep= num_rep) for month_id in range(1,13))
