#!/usr/bin/env python
# coding: utf-8
"""
Concatenate PCs from all spatiotemporal covariates and temporal covariates
to construct the pandas dataframe for final Covariates
used to train XGBoost, Lasso, Encoder-Decoder, etc. 
"""


import pandas as pd
import cfg_target as cfg
import preprocess


def main():
    rootpath = cfg.savepath_data
    var_names = cfg.vars
    var_locations = cfg.locations
    n_components = cfg.num_pcs

    temporal_vars = cfg.temporal_set

    num_var = 0

    for var_name, var_location in zip(var_names, var_locations):

        print('var:{}, location:{}'.format(var_name, var_location))


        for i in range(n_components):

            print('load data: pc{}'.format(i))

            var = '{}_{}_pca_{}'.format(var_location, var_name, i)
            data = pd.read_hdf(rootpath + '{}_zscore.h5'.format(var))

            if num_var == 0:

                df_all = data[var + '_zscore'].to_frame()

            else:
                df_all = df_all.merge(data[var + '_zscore'].to_frame(), how='left', on=['start_date'])

            num_var += 1


    # add temporal covariates

    data = pd.read_hdf(rootpath + 'temporal_covariates.h5')

    for var in temporal_vars:

        df_all = df_all.merge(data[var].to_frame(), how='left', on=['start_date'])

    df_all.to_hdf(rootpath+'covariates_all_pc{}.h5'.format(n_components), key='df', mode='w')


if __name__ == "__main__":
    main()
