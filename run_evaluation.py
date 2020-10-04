import os
import cfg_target
import pickle
import numpy as np
from utils import *
import argparse
import evaluation


model_names = cfg_target.model_names  # ['EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']
test_years = cfg_target.test_years  # [2017, 2018]
month_range = cfg_target.month_range
rootpath = cfg_target.forecast_rootpath
num_rep = cfg_target.num_rep

result_all = {}
for model_name in model_names:
    print('evaluate ',model_name)
    if model_name in ['EncoderFNN_AllSeq_AR_CI', 'EncoderFNN_AllSeq_AR', 'EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']:
        result_train, result_test = evaluation.eval_forecast(model_name, rootpath, test_years, month_range, True, num_rep)
    elif model_name in ['XGBoost', 'Lasso', 'FNN', 'CNN_FNN', 'CNN_LSTM']:
        result_train, result_test = evaluation.eval_forecast(model_name, rootpath, test_years, month_range, False)
    result_all[model_name] = {'train': result_train, 'test': result_test}

save_results(rootpath + 'forecast_results/result_all.pkl', result_all)
