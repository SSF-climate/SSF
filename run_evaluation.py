import os
import cfg_target
import pickle
import numpy as np
from utils import *
import argparse
import evaluation

# from random_cv import best_hyperparameter

model_names = cfg_target.model_names  # ['EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']
test_years = cfg_target.test_years  # [2017, 2018]
month_range = cfg_target.month_range
rootpath = cfg_target.forecast_rootpath
num_rep = cfg_target.num_rep


for model_name in model_names:
    if model_name in ['EncoderFNN_AllSeq_AR_CI', 'EncoderFNN_AllSeq_AR', 'EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']:
        result = evaluation.eval_forecast(model_name, rootpath, test_years, month_range, True, num_rep)
    elif model_name in ['XGBoost', 'Lasso', 'FNN', 'CNN_FNN', 'CNN_LSTM']:
        result = evaluation.eval_non_rep(model_name, rootpath, test_years, month_range, False)
