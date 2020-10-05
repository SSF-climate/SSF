"""
Apply zscoring to spatiotemporal target, e.g., tmp2m
"""
import numpy as np
import pandas as pd
import preprocess
import cfg_target as cfg


# Load data from path
rootpath = cfg.rootpath_data
path_to_save = cfg.savepath_data

filename = 'target.h5'

train_start = pd.Timestamp(cfg.train_start_date)
train_end = pd.Timestamp(cfg.train_end_date)

test_start = pd.Timestamp(cfg.test_start_date)
test_end = pd.Timestamp(cfg.end_date)


target = pd.read_hdf(rootpath + filename)
target_id = cfg.target
date_id = 'start_date'


preprocess.zscore_spatial_temporal(path_to_save,
                                   target, var_id=target_id,
                                   train_start=train_start, train_end=train_end,
                                   test_start=test_start, test_end=test_end,
                                   date_id=date_id)
