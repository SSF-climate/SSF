"""
To create training-test sets and training - test sets
"""
import pandas as pd
import preprocess
import cfg_target as cfg



#def main():
idx = pd.IndexSlice
date_col = 'start_date'



target = pd.read_hdf(cfg.data_target_file)
data = pd.read_hdf(cfg.data_cov_file)


train_start_date = cfg.train_start_date
end_date = cfg.end_date


time_index = pd.date_range(train_start_date, end_date, freq='1D')

target = target.loc[idx[:, :, time_index], :]
data = data.loc[idx[time_index], :]


cv_path = cfg.rootpath_cv
forecast_path = cfg.forecast_rootpath

target_var = cfg.target_var

val_years = cfg.val_years
test_years = cfg.test_years

val_train_range = cfg.val_train_range
test_train_range = cfg.test_train_range

past_years = cfg.past_kyears

val_range = cfg.val_range
val_freq = cfg.val_freq


test_start_date = cfg.test_start_date
test_time_index_all = pd.date_range(test_start_date, end_date, freq='7D')




# to create train-validation sets

for year in val_years:

    for num_forecast in range(1, 2):

 

        preprocess.train_val_split(cv_path,
                                   data,
                                   target, target_var,
                                   year, num_forecast,
                                   train_range=val_train_range,
                                   past_years=past_years,
                                   test_range=val_range,
                                   test_freq=val_freq,
                                   n_jobs=20)

# to create train-test sets

for year in test_years:

    for num_forecast in range(1, 2):

        preprocess.train_test_split(forecast_path,
                                    data,
                                    target, target_var,
                                    test_time_index_all,
                                    year, num_forecast,
                                    train_range=test_train_range,
                                    past_years=past_years,
                                    n_jobs=20)



##
##if __name__ == "__main__":
##    main()
