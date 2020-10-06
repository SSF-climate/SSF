import numpy as np

################### Configuration for Data Loading ################################
path = 'data/'  # need to change to the absolute path of the data files
rootpath_cv = 'SSF/data/random_cv/'
forecast_rootpath = 'SSF/data/forecast/'
param_path = 'SSF/data/random_cv/cv_results_test/best_parameter/'


# target variables
target = 'tmp2m'  # target variable: 'tmp2m' or 'precip'
target_res = 2  # target resolution
target_lat = [25.1, 48.9]  # lat range of the target region
target_lon = [235.7, 292.8]  # lon range of the target region
target_us_all = True
shift_days = 14  # 'days to shift for target variable 14 days means 2-week ahead prediction,
forecast_range = 14  # 'forecast range' - 14 days average or summation
operation = 'mean'  # 'compute the summation or average over the forecast range
# ("mean" for temperature, "sum" for precipitation)
save_target = True  # 'flag to indicate weather to save shifted target


train_start_date = '1990-01-01'   # Set the start date for training
train_end_date = '2016-12-31'     # Set the end date for training set
test_start_date = '2017-01-01'   # Set the end date for training'
end_date = '2018-12-31'   # Set the end date for whole dataset'

# spatial temporal covariate variables
covariates_us = ['tmp2m','precip']
covariates_global = ['hgt500','slp','rhum500'] # spatial-temporal covariates on land.
covariates_sea =  ['sst'] # spatial-temporal covariates over ocean.
pacific_atlantic = True


lat_range_global = [0, 50.0]   # latitude range for covariates
lon_range_global = [120, 340]   # longitude range for covariates

lat_range_us = [25.1, 48.9]  # latitude range for covariates
lon_range_us = [235.7, 292.8]  # longitude range for covariates

lat_range_sea = [0, 50]  # latitude range for covariates
lon_range_sea = [120, 340]  # longitude range for covariates


# spatial variable
# add_spatial = True  # flag to indicate adding spatial features: may not need this flag
spatial_set = ['elevation']  # spatial variables

# temporal variable
# add_temporal = True  # flag to indicate adding temporal features: may not need
temporal_set = ['mei', 'nao', 'mjo_phase', 'mjo_amplitude', 'nino3', 'nino4', 'nino3.4', 'nino1+2']  # temporal variable(s)

save_cov = True    # flag to indicate weather to save covariance


# target_lat = 37.75 # 'latitude range for target variable'
# target_lon = 237.75 #'longitude range for target variable'

################### Configuration for Dataset ################################

# preprocessing
rootpath_data = 'data/'
savepath_data = 'data/'
vars = ['tmp2m', 'sst']
locations = ['us', 'atlantic']

num_pcs = 10

# train-validation split
data_target_file = 'data/target_multitask_zscore.h5'
data_cov_file = 'data/covariates_all_pc10_nao_nino.h5'
target_var = 'tmp2m'

val_years = [2016, 2015, 2014, 2013, 2012] # years to create validation sets

val_train_range = 5 # number of years in the training set (train-val split)

val_range = 28 # number of days to include in the validation set
val_freq = '7D' # frequency to generate validation date

# train-test split
test_years = [2017, 2018]

test_train_range = 5 # number of years in the training set (train-test split)


past_ndays = 28   # number of days to aggaregate in the past: t-n,...,t-1'

past_kyears = 2  # number of years in the past to aggaregate: t-k,...,t year'

# future_mdays = 0

################ Configuration for hyper parameter tuning  ######################
# param_grid for encoder decoder model
param_grid_en_de = {'hidden_dim': [10, 20, 40, 60, 150, 200],
                    'num_layers': [2, 3, 4, 5, 6],
                    'learning_rate': [0.005, 0.001],
                    'threshold': [0.5, 0.6],
                    'num_epochs': [100, 200, 300],
                    'decoder_len': [4, 11, 18],
                    'last_layer': [True, False],
                    'seq_len': [4, 11, 18],
                    'linear_dim': [50, 100, 200],
                    'drop_out': [0.1, 0.2],
                    'ci_dim': 8}

# param_grid for XGBoost
param_grid_xgb = {'max_depth': [3, 5, 7, 9],
                  'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'gamma': [0, 0.25, 0.5, 1.0],
                  'n_estimators': [100, 150, 200],
                  'learning_rate': [0.01, 0.05, 0.1]}

# param_grid for Lasso
param_grid_lasso = {'alpha': np.linspace(0, 1, 1001)}

# param_grid for FNN
param_grid_fnn = {'hidden_dim': [128, 256, 512], 'num_layers': [2, 4, 6, 8]}

# param_grid for CNN-LSTM
param_grid_cnn_lstm = {'lr': [0.01],
                       'module__kernel_size': [9, 13, 15],
                       'module__stride': [5],
                       'module__hidden_dim': [100, 200, 300, 400, 500],
                       'module__num_lstm_layers': [2, 4],
                       'module__num_epochs': [100]}
# param_grid for CNN-FNN
param_grid_cnn_fnn = {'kernel_size': [9, 13, 15],
                      'stride': [5, 7, 9],
                      'hidden_dim': [50, 100, 200],
                      'num_layers': [2, 4]}

num_random = 30

month_range = list(range(1,13))
model_names = ['Lasso', 'FNN', 'XGBoost','CNN_FNN', 'CNN_LSTM', 'EncoderFNN_AllSeq_AR_CI',
               'EncoderFNN_AllSeq_AR','EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']
# ['EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']
cv_metric = 'cos'
one_day = True
num_rep = 10
