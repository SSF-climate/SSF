"""
Apply PCA first, and then zscoring to spatiotemporal covariates
"""
import argparse
import sys
import os
os.chdir(os.path.join(".."))
sys.path.insert(0, 'SSF/')
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



preprocess.get_pca_from_covariate(path_to_save,
                                  data,
                                  var_name,var_location,
                                  train_start,train_end,
                                  test_start,test_end,
                                  n_components=n_components)

pca_filename = '{}_{}_pca_all.h5'.format(var_location, var_name)
pca_data = pd.read_hdf(path_to_save + pca_filename)


for i in range(n_components):
        var = '{}_{}_pca_{}'.format(var_location, var_name, i)

        preprocess.zscore_temporal(path_to_save,
                                   pca_data,var,
                                   train_start, train_end,
                                   test_start, test_end)
















##var = '{}_{}_pca_{}'.format(var_location,var_name,0)
##
##
##preprocess.zscore_temporal('test/',data,var,train_start,train_end,test_start,test_end)
##
##preprocess.zscore_spatial_temporal('test/',data,'tmp2m',train_start,train_end,test_start,test_end)
##
##
##
##
##df1,df2 = preprocess.do_pca_on_covariate(train_X,test_X,n_components = 10,location=var_location,var_id = var_name)
##
##
##
##preprocess.get_pca_from_covariate(var_name,var_location,
##                                  data,
##                                  'test/',
##                                  train_start,train_end,
##                                  test_start,test_end,
##                                  n_components=10)
##
##






