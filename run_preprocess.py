"""
Run preprocess_covariates.py for covariates define in cfg
"""
import cfg_target as cfg
import os


var_locations = cfg.locations
var_names = cfg.vars
exe_cov_file = 'preprocess/preprocess_covariate.py'
exe_target_file = 'preprocess/preprocess_target.py'

# to preprocess spatiotemporal covariates
for var_name, var_location in zip(var_names, var_locations):

    cmd = "{} {} --var {} --location {}".format("python3", exe_cov_file, var_name, var_location)
    print(cmd)
    os.system(cmd)

# to preprocess target variable

cmd = "{} {}".format("python3", exe_target_file)
print(cmd)
os.system(cmd)

