"""
Apply scoring to spatiotemporal covariates first,
then convert normalized covariates to squared maps
"""
import argparse
import sys
import os
os.chdir(os.path.join(".."))
sys.path.insert(0, 'S2S/')
import numpy as np
import pandas as pd
import preprocess
import cfg_target as cfg



parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, help='var name')
parser.add_argument('--location', type=str, help='var location')

args = parser.parse_args()
var_location = args.location
var_name = args.var

##idx = pd.IndexSlice
##date_id = 'start_date'

## Load data from path
rootpath = cfg.rootpath_data
path_to_save = cfg.savepath_data



if var_location == 'global':
    filename ='covariates_global.h5'
elif var_location == 'us':
    filename = 'covariates_us.h5'
elif var_location == 'pacific':
    filename = 'covariates_pacific.h5'
elif var_location == 'atlantic':
    filename = 'covariates_atlantic.h5'
else:
    raise ValueError("No such covariate!")



train_start = pd.Timestamp(cfg.train_start_date)
train_end = pd.Timestamp(cfg.train_end_date)

test_start = pd.Timestamp(cfg.test_start_date)
test_end = pd.Timestamp(cfg.end_date)

n_components = cfg.num_pcs

data =  pd.read_hdf(rootpath+filename)


df = preprocess.zscore_spatial_temporal_map(path_to_save,
                                            data,
                                            var_name, var_location,
                                            train_start,train_end,
                                            test_start,test_end,
                                            date_id = 'start_date',
                                            to_save=True)
# To save the intermideta zscored covariates, change "to_save" to True


cov_map = preprocess.convert_covariate_to_map(df, var=var_name + '_zscore', num_cores=24)
preprocess.save_results(path_to_save,'{}_{}_map.pkl'.format(var_name,var_location),cov_map)
















