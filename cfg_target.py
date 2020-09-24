################### Configuration for Data Loading ################################
path = '../../../../project/banerjee-00/S2S_dataset/data_new/'
#target variables
target = 'tmp2m' #target variable: 'tmp2m' or 'precip'
target_res = 2 #target resolution
target_lat = [25.1,48.9] #lat range of the target region
target_lon = [235.7,292.8] #lon range of the target region
target_us_all = True
shift_days = 14 # 'days to shift for target variable 14 days means 2-week ahead prediction,
forecast_range = 14 # 'forecast range' - 14 days average or summation
operation = 'mean' # 'compute the summation or average over the forecast range
                  #("mean" for temperature, "sum" for precipitation)
save_target = True # 'flag to indicate weather to save shifted target


train_start_date = '2012-01-01' # 'Set the start date for training'
#test_start_date = '2018-01-01' # 'Set the end date for training'
end_date = '2017-12-31' # 'Set the end date for whole dataset'

#spatial temporal covariate variables
covariates_us = []#['tmp2m','precip']
covariates_global = []#['hgt500','slp','rhum500'] #'spatial-temporal covariates on land.'
covariates_sea = []# ['sst'] #'spatial-temporal covariates over ocean.'
pacific_atlantic = True

past_ndays = 0   # number of days to aggaregate in the past: t-n,...,t-1'

past_kyears = 2 # number of years in the past to aggaregate: t-k,...,t year'


lat_range_global = [0,50.0] # 'latitude range for covariates')
lon_range_global = [120,340] #'longitude range for covariates')

lat_range_us = [25.1,48.9] # 'latitude range for covariates')
lon_range_us = [235.7,292.8] #'longitude range for covariates')

lat_range_sea = [0,50] # latitude range for covariates')
lon_range_sea = [120,340] # longitude range for covariates')


#spatial variable
add_spatial = True # 'flag to indicate adding spatial features: may not need this flag
spatial_set=[]#['elevation'] # spatial variables

#temporal variable
add_temporal = True # 'flag to indicate adding temporal features: may not need
temporal_set= ['mei', 'nao', 'nino3', 'nino3.4', 'nino1+2'] # 'temporal variable(s)


save_cov = True    # flag to indicate weather to save covariance


#parameters for spliting - may not necessary
train_range = 1825 #'Size of each training set: number of days'

val_range = 365 # 'Size of each training set'

stride = 280# 'Size to shift train-val window'
#
#target_lat = 37.75 # 'latitude range for target variable'
#target_lon = 237.75 #'longitude range for target variable'
#



################ Configuration for preprocess  ######################


################ Configuration fro training  #########################
inset_gap = 0    # 'sampling gap among training samples' )
#
#past_ndays = 0   # number of days to aggaregate in the past: t-n,...,t-1'
#
future_mdays = 0 # number of days to aggaregate in the future: t+1,...,t+m'
#
#past_kyears = 0 # number of years in the past to aggaregate: t-k,...,t year'



################ Configuration for Evaluation ##########################
skill = 'cosine' # Type of skill: ['cosine', 'rmse','r2']'



################ Configuration for Model Selection #####################
regressor = 'autoencoder_combine'   # Type of regressor. Choose among ['Lasso','xgboost', 'fullconn','random','average','autoencoder_predictor','autoencoder_combine']



# Lasso: The optimization objective for Lasso is: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

lasso_alpha = 0.1  # Constant that multiplies the L1 term. Defaults to 0.1. alpha = 0 is equivalent to an ordinary least square.')

# xgboost: regularization parameters
xg_objective = 'reg:squarederror'# 'specify the loss function to be used like for regression problem.'

##xg_colsample_bytree = [0.3]      #'the subsample ratio of columns when constructing each tree.')
##
##xg_learning_rate = [0.1]         # Step size shrinkage used in update to prevents overfitting.Range is [0,1]')
##
##xg_max_depth = [5]               # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.')
##
##xg_n_estimators = [100,150]      # number of trees you want to build.')
##
##xg_alpha = [0.1,0.5]             # L1 regularization term on weights. Increasing this value will make model more conservative.')
##
##xg_lambda = [0.1]                # L2 regularization term on weights. Increasing this value will make model more conservative.')

xg_colsample_bytree = 0.3      #'the subsample ratio of columns when constructing each tree.')

xg_learning_rate = 0.1        # Step size shrinkage used in update to prevents overfitting.Range is [0,1]')

xg_max_depth = 5              # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.')

xg_n_estimators = 150      # number of trees you want to build.')

xg_alpha = 0.1             # L1 regularization term on weights. Increasing this value will make model more conservative.')

xg_lambda = 0.1                # L2 regularization term on weights. Increasing this value will make model more conservative.')


# LSTM-Autoencoder
# Parameters to create autoencoder model
hid_dim = 64
n_layers = 2
enc_dropout = 0.
dec_dropout = 0.
# parameters related to training
n_epochs = 1000
learning_rate = 1e-3

# CNN
kernel_size = 5
stride = 3
padding = 0


# CNN+attention
head=6
# CNN
att_kernel_size = 5
att_stride = 3
att_padding = 0
att_dropout=0.1
att_num_encoder_layer=2
att_input_dim=36
att_ff_dim=64
