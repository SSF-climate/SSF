"""
Load a subset of data required by configure file (cfg_target)
"""
import cfg_target
from data_load import DataLoader
data = DataLoader(cfg_target)

target = data.data_download_target()
covariates_us, covariates_sea, covariates_global, spatial_covariates, temporal_covariates = data.data_download_cov()
