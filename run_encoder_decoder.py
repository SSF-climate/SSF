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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model_name')
parser.add_argument('--year', type=int, help='year')
parser.add_argument('--month', type=int, help='month')

args = parser.parse_args()
model_name = args.model_name
year = args.year
month_id = args.month


def forecast_rep(month_id, year, rootpath, param_path, device, model_name, num_rep):
    results = {}
    results['prediction_train'] = []
    results['prediction_test'] = []
    train_X = load_results(rootpath + 'train_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_X = load_results(rootpath + 'test_X_pca_{}_forecast{}.pkl'.format(year, month_id))
    train_y = load_results(rootpath + 'train_y_pca_{}_forecast{}.pkl'.format(year, month_id))
    test_y = load_results(rootpath + 'test_y_pca_{}_forecast{}.pkl'.format(year, month_id))

    if model_name == 'EncoderFNN_AllSeq_AR_CI' or model_name == 'EncoderFNN_AllSeq_AR':
        train_ar = load_results(rootpath + 'train_y_pca_ar_{}_forecast{}.pkl'.format(year, month_id))
        test_ar = load_results(rootpath + 'test_y_pca_ar_{}_forecast{}.pkl'.format(year, month_id))

    input_dim = train_X.shape[-1]
    output_dim = train_y.shape[-1]
    # ar_dim = train_X.shape[1]
    best_parameter = load_results(param_path + '{}_forecast{}.pkl'.format(model_name, month_id))
    for rep in range(num_rep):
        if model_name == 'EncoderDecoder':
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            curr_decoder_len = best_parameter['decoder_len']
            curr_threshold = best_parameter['threshold']
            curr_lr = best_parameter['learning_rate']
            curr_num_epochs = best_parameter['num_epochs']
            mdl = model.EncoderDecoder(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim,
                                       num_layers=curr_num_layer, learning_rate=curr_lr, decoder_len=curr_decoder_len,
                                       threshold=curr_threshold, num_epochs=curr_num_epochs)

            # set data for training
            train_dataset = model.MapDataset(train_X, train_y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)
        elif model_name == 'EncoderFNN':
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            curr_seq_len = best_parameter['seq_len']
            curr_threshold = best_parameter['threshold']
            curr_lr = best_parameter['learning_rate']
            curr_num_epochs = best_parameter['num_epochs']
            curr_last_layer = best_parameter['last_layer']
            mdl = model.EncoderFNN(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim,
                                   num_layers=curr_num_layer, last_layer=curr_last_layer, seq_len=curr_seq_len,
                                   learning_rate=curr_lr, threshold=curr_threshold, num_epochs=curr_num_epochs)
            # set data for training
            train_dataset = model.MapDataset(train_X, train_y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)
        elif model_name == 'EncoderFNN_AllSeq':
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            curr_seq_len = best_parameter['seq_len']
            curr_threshold = best_parameter['threshold']
            curr_lr = best_parameter['learning_rate']
            curr_num_epochs = best_parameter['num_epochs']
            curr_linear_dim = best_parameter['linear_dim']
            curr_drop_out = best_parameter['drop_out']

            mdl = model.EncoderFNN_AllSeq(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim,
                                          num_layers=curr_num_layer, seq_len=curr_seq_len, linear_dim=curr_linear_dim,
                                          learning_rate=curr_lr, dropout=curr_drop_out, threshold=curr_threshold,
                                          num_epochs=curr_num_epochs)
            # set data for training
            train_dataset = model.MapDataset(train_X, train_y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)
        elif model_name == 'EncoderFNN_AllSeq_AR':
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            curr_seq_len = best_parameter['seq_len']
            curr_threshold = best_parameter['threshold']
            curr_lr = best_parameter['learning_rate']
            curr_num_epochs = best_parameter['num_epochs']
            curr_linear_dim = best_parameter['linear_dim']
            curr_drop_out = best_parameter['drop_out']
            mdl = model.EncoderFNN_AllSeq_AR(input_dim=input_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim,
                                             num_layers=curr_num_layer, seq_len=curr_seq_len, linear_dim=curr_linear_dim,
                                             learning_rate=curr_lr, dropout=curr_drop_out, threshold=curr_threshold,
                                             num_epochs=curr_num_epochs)
            train_dataset = model.MapDataset_ar(train_X, train_y_ar, train_y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)
        elif model_name == 'EncoderFNN_AllSeq_AR_CI':
            curr_hidden_dim = best_parameter['hidden_dim']
            curr_num_layer = best_parameter['num_layers']
            curr_seq_len = best_parameter['seq_len']
            curr_threshold = best_parameter['threshold']
            curr_lr = best_parameter['learning_rate']
            curr_num_epochs = best_parameter['num_epochs']
            curr_linear_dim = best_parameter['linear_dim']
            curr_drop_out = best_parameter['drop_out']
            ci_dim = best_parameter['ci_dim']
            mdl = model.EncoderFNN_AllSeq_AR_CI(input_dim=input_dim - ci_dim, output_dim=output_dim, hidden_dim=curr_hidden_dim,
                                                num_layers=curr_num_layer, seq_len=curr_seq_len, linear_dim=curr_linear_dim,
                                                ci_dim=ci_dim, learning_rate=curr_lr, dropout=curr_drop_out,
                                                threshold=curr_threshold, num_epochs=curr_num_epochs)
            train_dataset = model.MapDataset_ar(train_X, train_y_ar, train_y)
            train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=False)

        model.init_weight(mdl)
        # send model to gpu
        mdl.to(device)
        mdl.fit(train_loader, device)
        state = {'state_dict': mdl.state_dict()}
        torch.save(state, rootpath + 'model/{}_{}_{}.t7'.format(model_name, year, month_id))
        pred_train = mdl.predict(train_X, device)
        pred_y = mdl.predict(test_X, device)
        results['target_train'] = train_y
        results['prediction_train'].append(pred_train)
        results['target_test'] = test_y
        results['prediction_test'].append(pred_y)

    save_results(rootpath + 'forecast_results/results_{}_{}_{}.pkl'.format(model_name, year, month_id), results)


# set device for running the code
num_rep = cfg_target.num_rep
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rootpath = cfg_target.forecast_rootpath
param_path = cfg_target.param_path

forecast_rep(month_id=month_id, year=year, rootpath=rootpath, param_path=param_path, device=device, model_name=model_name, num_rep=num_rep)
# Parallel(n_jobs=12)(delayed(forecast_rep)(month_id,rootpath=rootpath,param_path=param_path, device= device, model_name=model_name,folder_name=folder_name, num_rep= num_rep) for month_id in range(1,13))
