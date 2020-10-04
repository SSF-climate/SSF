"""
To create training - validation sets
"""
import pandas as pd
import preprocess
import cfg_target as cfg



def main():
    idx = pd.IndexSlice
    date_col = 'start_date'



    target = pd.read_hdf(cfg.data_target_file)
    data = pd.read_hdf(cfg.data_cov_file)


    start_date = cfg.train_start_date
    end_date = cfg.end_date


    time_index = pd.date_range(start_date, end_date, freq='1D')

    target = target.loc[idx[:, :, time_index], :]
    data = data.loc[idx[time_index], :]


    cv_path = cfg.rootpath_cv
    var = cfg.target_var
    val_years = cfg.val_years
    train_range = cfg.val_train_range
    past_years = cfg.past_kyears
    val_range = cfg.val_range
    val_freq = cfg.val_freq


    for year in val_years:

        for num_forecast in range(1, 12):

     

            preprocess.train_val_split(cv_path,
                                       data,
                                       target, var,
                                       year, num_forecast,
                                       train_range=train_range,
                                       past_years=past_years,
                                       test_range=val_range,
                                       test_freq=val_freq,
                                       n_jobs=20)


if __name__ == "__main__":
    main()
