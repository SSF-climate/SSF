#!/usr/bin/env python
# coding: utf-8

#need a function for clarify the spatial range
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import math
from itertools import chain
import config
#from utils import *
import argparse
import numpy as np
import pandas as pd
import math
import pickle
#import xarray as xr
from random import randint
from random import seed
#from Data_load_year import dataloader

import torch

import os

import model
#from multitask import *
import cfg
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from joblib import Parallel,delayed

import torch
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model_name')
parser.add_argument('--year', type=int, help='year')
parser.add_argument('--month', type=int, help='month')
args=parser.parse_args()

model_name=args.model_name
year=args.year
month_id=args.month
#model_name='mulitask_encoder_fnn'

def load_results(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f,encoding='bytes')
    return data


def save_results(filename,results):

    with open(filename, 'wb') as fh:
        pickle.dump(results, fh)



def forecast_rep(month_id,year,rootpath,param_path,device,model_name,num_rep):
    results={}
    results['prediction_train']=[]
    results['prediction_test']=[]
    results['history']=[]
    train_X = load_results(rootpath+'train_X_pca_{}_forecast{}.pkl'.format(year,month_id))
    test_X = load_results(rootpath+'test_X_pca_{}_forecast{}.pkl'.format(year,month_id))
    train_y= load_results(rootpath+'train_y_pca_{}_forecast{}.pkl'.format(year,month_id))
    test_y= load_results(rootpath+'test_y_pca_{}_forecast{}.pkl'.format(year,month_id))

    input_dim = train_X.shape[-1]
    output_dim = train_y.shape[-1]
    ar_dim = train_X.shape[1]
    for rep in range(num_rep):
        if model_name == 'mulitask_encoder_fnn':
            best_parameter=load_results(param_path+'/encoder_fnn_multitask_forecast{}.pkl'.format(month_id))
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            learning_rate = best_parameter['learning_rate']
            mdl = model.Encoder_fnn_multitask_gpu(input_dim = input_dim, output_dim= output_dim, hidden_dim = curr_hidden_dim, num_layers = curr_num_layer,last_layer=True, learning_rate = learning_rate, threshold=0.5 ,num_epochs = 300)
            #initialize the model
            init_weight(mdl)
        elif model_name == 'mulitask_encoder_fnn_all_seq':
            best_parameter=load_results(param_path+'/encoder_fnn_all_seq_multitask_forecast{}.pkl'.format(month_id))
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            curr_lr = 0.001#best_parameter['learning_rate']
            curr_linear_dim = best_parameter['linear_dim']
            curr_dropout = best_parameter['dropout']
            mdl = model.Encoder_fnn_all_seq_multitask_gpu(input_dim = input_dim, output_dim=output_dim, hidden_dim = curr_hidden_dim, num_layers = curr_num_layer,seq_len=train_X.shape[1],linear_dim=curr_linear_dim,dropout=curr_dropout,learning_rate = curr_lr, threshold=0.5 ,num_epochs = 300)
            #initialize the model
            init_weight(mdl)


            #set data for training
        train_dataset = model.MapDataset(train_X,train_y)
        train_loader = DataLoader(dataset=train_dataset,batch_size=512,shuffle=False)

        #send model to gpu
        mdl.to(device)
        #fit the model
        history = mdl.fit_cv(train_loader,test_X,test_y,device)

        state = {'state_dict': mdl.state_dict()}

        torch.save(state, 'forecast_results_sessonal_daily_nao_nino/model/{}_{}_{}.t7'.format(model_name,year,month_id))
        pred_train = mdl.predict(train_X,device)
        pred_y = mdl.predict(test_X,device)

        results['target_train']=train_y
        results['prediction_train'].append(pred_train)
        results['target_test']=test_y
        results['prediction_test'].append(pred_y)
        results['history'].append(history)

    save_results('forecast_results_sessonal_daily_nao_nino/results_{}_{}_{}.pkl'.format(model_name,year,month_id),results)


#set device for running the code
num_rep=10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
year = 2017
month_id = 1
#set the path to train-validation data
rootpath='/home/srivbane/lixx1166/S2S/forecast/data_seasonal_daily_nao_nino_25year/'
param_path = '/home/srivbane/hexxx893/S2S/random_cv/best_params_seasonal_daily_nao_nino/'
train_X = load_results(rootpath+'train_X_pca_{}_forecast{}.pkl'.format(year,month_id))
test_X = load_results(rootpath+'test_X_pca_{}_forecast{}.pkl'.format(year,month_id))
train_y= load_results(rootpath+'train_y_pca_{}_forecast{}.pkl'.format(year,month_id))
train_y_ar = load_results(rootpath+'train_y_pca_ar_{}_forecast{}.pkl'.format(year,month_id))
test_y= load_results(rootpath+'test_y_pca_{}_forecast{}.pkl'.format(year,month_id))
test_y_ar = load_results(rootpath+'test_y_pca_ar_{}_forecast{}.pkl'.format(year,month_id))
input_dim = train_X.shape[-1]
output_dim = train_y.shape[-1]
best_parameter=load_results(param_path+'/encoder_fnn_multitask_forecast{}.pkl'.format(month_id))
curr_hidden_dim = best_parameter['hidden_dim']
curr_num_layer = best_parameter['num_layers']
learning_rate = best_parameter['learning_rate']
#mdl = model.EncoderDecoder(input_dim = input_dim, output_dim= output_dim, hidden_dim = curr_hidden_dim, num_layers = curr_num_layer,#,last_layer=True,
#    learning_rate = learning_rate, threshold=0.5 ,num_epochs = 10)
    
#mdl = model.EncoderFNN(input_dim = input_dim, output_dim= output_dim, hidden_dim = curr_hidden_dim, num_layers = curr_num_layer,last_layer=True,
#    learning_rate = learning_rate, threshold=0.5 ,num_epochs = 10)
#
#mdl = model.EncoderFNN_AllSeq(input_dim = input_dim, output_dim= output_dim, hidden_dim = curr_hidden_dim, num_layers = curr_num_layer, learning_rate = learning_rate, threshold=0.5 ,num_epochs = 30)
#
#mdl = model.EncoderFNN_AllSeq_AR(input_dim = input_dim, output_dim= output_dim, hidden_dim = curr_hidden_dim, num_layers = curr_num_layer,
#learning_rate = learning_rate, threshold=0.5 ,num_epochs = 30)

mdl = model.EncoderFNN_AllSeq_AR_CI(input_dim = 80, output_dim= output_dim, hidden_dim = curr_hidden_dim, num_layers = curr_num_layer,
learning_rate = learning_rate, threshold=0.5 ,num_epochs = 30)

#initialize the model
model.init_weight(mdl)

#train_dataset = model.MapDataset(train_X,train_y)
train_dataset = model.MapDataset_ar(train_X, train_y_ar, train_y)
train_loader = DataLoader(dataset=train_dataset,batch_size=512,shuffle=False)

#send model to gpu
mdl.to(device)
#fit the model
#history = mdl.fit_cv(train_loader,test_X,test_y,device)
history = mdl.fit_cv(train_loader,test_X,test_y_ar,test_y,device)

state = {'state_dict': mdl.state_dict()}

#torch.save(state, 'forecast_results_sessonal_daily_nao_nino/model/{}_{}_{}.t7'.format(model_name,year,month_id))
#pred_train = mdl.predict(train_X,device)
#pred_y = mdl.predict(test_X,device)

pred_train = mdl.predict(train_X,train_y_ar,device)
pred_y = mdl.predict(test_X,test_y_ar,device)

#forecast_rep(month_id=month_id,year=year,rootpath=rootpath,param_path=param_path, device= device, model_name=model_name,num_rep= num_rep)
