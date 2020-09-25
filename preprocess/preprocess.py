"""
Functions used for pre-processing
"""


#import math
import pickle
#import copy
#import config
import os

# for multiprocessing
#from functools import partial
#from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA




def load_results(filename):

    """ Load a pickle file

    """

    with open(filename, 'rb') as file_to_load:
        data = pickle.load(file_to_load, encoding='bytes')

    return data


def save_results(rootpath, filename, results):

    """ Save results as a pickle file

    """
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)



    with open(rootpath+filename, 'wb') as file_to_save:
        pickle.dump(results, file_to_save)



########## Code to perform Principal Component Analysis (PCA) on a covariate ###################


def do_pca_on_covariate(df_train, df_test, n_components=10, location='pacific', var_id='sst'):


    """ Do PCA: learn PC loadings from training data,
                and project test data onto corresponding directions

    Args:
            df_train: multi-index (spatial-temporal) pandas dataframe
                      -- Training data used to compute Principal axes in feature space
            df_test:  multi-index pandas dataframe -- Test data
            n_components: int -- Number of components to keep
            location: str -- location indicator of the climate variable
            var_id: str -- climate variable to process
    Returns:
            df1: pandas dataframe -- PCs for training data
            df2: pandas dataframe -- PCs for test data



    """

    # check the legitimate of the given parameters 
    if not isinstance(df_train, pd.DataFrame):

        if isinstance(df_train, pd.Series):
            df_train = df_train.to_frame() # convert pd.series to pd.dataframe
        else:
            raise ValueError("Training data needs to be a pandas dataframe")


    if not isinstance(df_test, pd.DataFrame):

        if isinstance(df_test, pd.Series):
            df_test = df_test.to_frame() # convert pd.series to pd.dataframe
        else:
            raise ValueError("Test data needs to be a pandas dataframe")

    # check dataframe level!

    if len(df_train.index.names) < 3 or len(df_test.index.names) < 3:
        raise ValueError("Multiindex dataframe includes 3 levels: [lat,lon,start_date]")
        
    # flatten the dataframe such that the number of
    # samples equals the number of dates in the dataframe
    # and the number of features equals to lat x lon 
    df_train_flat = df_train.unstack(level=[0, 1])
    df_test_flat = df_test.unstack(level=[0, 1])
    x_train = df_train_flat.to_numpy()
    x_test = df_test_flat.to_numpy()
    

    # Initialize the PCA model such that it will reture the top n_components 
    pca = PCA(n_components=n_components)

    # Fit the model with Xtrain and apply the dimensionality reduction on Xtrain.
    pca_train = pca.fit_transform(x_train)
    # Apply dimensionality reduction to Xtest
    pca_test = pca.transform(x_test)

    # Convert PCs of Xtrain and Xtest to pandas dataframe
    col = ['{}_{}_pca_{}'.format(location, var_id, i) for i in range(n_components)]

    df1 = pd.DataFrame(data=pca_train,
                       columns=col,
                       index=df_train_flat.index)

    df2 = pd.DataFrame(data=pca_test,
                       columns=col,
                       index=df_test_flat.index)

    return(df1, df2)



def get_pca_from_covariate(rootpath,
                           data,
                           var_name, var_location,
                           train_start, train_end,
                           test_start, test_end,
                           n_components=10):



    """ Apply PCA on spatial-temporal Climate variables (Covariates),
        e.g., Sea surface temperature (SST)

    Args:
            data: multi-index pandas dataframe -- raw covariate to apply PCA
            var_name: str -- covariance name
            var_location: str -- covariance location (pacific, atlantic, us, and global)
            rootpath: str -- directory to save the results
            train_start, train_end: pd.Timestamp() -- the start date and the end date of the training set
            test_start, test_end: pd.Timestamp() -- the start date and the end date of the test set


    """
    idx = pd.IndexSlice

    # check the legitimate of the given parameters 
    if not isinstance(data, pd.DataFrame):

        if isinstance(data, pd.Series):
            data = data.to_frame() # convert pd.series to pd.dataframe
        else:
            raise ValueError("Covariate needs to be a pandas multiindex dataframe")

    # check if the train start date and the train end date is out of range

    if train_start < data.index.get_level_values('start_date')[0]:
        raise ValueError("Train start date is out of range!")        


    if train_end > data.index.get_level_values('start_date')[-1]:
        raise ValueError("Train end date is out of range!")


    # check if the test start date and the test end date is out of range
    if test_start < train_start:
        raise ValueError("Test start date is out of range!")

    if test_end < train_end or test_end > data.index.get_level_values('start_date')[-1]:
        raise ValueError("Test end date is out of range!")



    print('create training-test split')
    train_x = data.loc[idx[:, :, train_start:train_end], :]
    test_x = data.loc[idx[:, :, test_start:test_end], :]


    # start PCA
    print('start pca')
    train_x_pca, test_x_pca = do_pca_on_covariate(train_x[var_name], test_x[var_name],
                                                  n_components, var_location, var_name)


    # save PCA data
    all_x_pca = train_x_pca.append(test_x_pca)
    all_x_pca.to_hdf(rootpath + '{}_{}_pca_all.h5'.format(var_location, var_name),
                     key=var_name, mode='w')

########## Code to perform z-score on a time-series using long-term mean and std ############################

def get_mean(df1, var_id='tmp2m', date_id='start_date'):

    """ Compute the mean and standard deviation of a covariate on the given period
    Args:
            d1: multi-index pandas dataframe -- covariate
            var_id: str -- covariate name
            date_id: str -- index column name for date
    Return(s):
            df1: multi-index pandas dataframe -- with month-day-mean-std added

    """

    indexnames = df1.index.names
    idxlevel = indexnames.index(date_id)


    df1 = df1.assign(month=df1.index.get_level_values(idxlevel).month)
    df1 = df1.assign(day=df1.index.get_level_values(idxlevel).day)

    # get mean of each date
    df1['{}_daily_mean'.format(var_id)] = df1.groupby(['month', 'day'])[var_id].transform('mean')
    # get std of each date
    df1['{}_daily_std'.format(var_id)] = df1.groupby(['month', 'day'])[var_id].transform('std')
  
    return df1.fillna(0)

def add_month_day(df1, date_id='start_date'):

    """ Extract the month-of-year and day-of-year from the date index,
        and add it to the datafram
    Args:
            d1: multi-index pandas dataframe -- covariate
            date_id: str -- index column name for date

    """

    indexnames = df1.index.names
    idxlevel = indexnames.index(date_id)

    df1 = df1.assign(month=df1.index.get_level_values(idxlevel).month)
    df1 = df1.assign(day=df1.index.get_level_values(idxlevel).day)

    return(df1)


def zscore_temporal(rootpath,
                    data,
                    var,
                    train_start='1986-01-01', train_end='2016-12-31',
                    test_start='2017-01-01', test_end='2018-12-31',
                    date_id='start_date'):


    """ Do zscore on time series only (no spatial information), e.g., pca of a covariate 
    Args:
            rootpath: directory to save the results
            data: pd.Dataframe -- dataframe contains data that is about to apply zscore
            var: str -- variable name
            train_start, train_end: str -- the start date and the end date of the training set
            test_start, test_end: str -- the start date and the end date of the test set
            date_id: str -- index column name for date

    """

    # check the legitimate of the given parameters 
    if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
        raise ValueError("Data needs to be a pandas dataframe/series.")

    idx = pd.IndexSlice

    target = data[var].to_frame()

    print('pre-process: {}'.format(var))

    
    df1 = target.loc[idx[train_start:train_end], :]# train
    df2 = target.loc[idx[test_start:test_end], :]# test



    df1 = get_mean(df1, var)
    #  get first element of each group: mean for each location each month-day 
    month_day = df1.groupby(['month', 'day']).first() 
    month_day = month_day.reset_index()



    # add month-day column to second dataframe
    df2 = add_month_day(df2)
    df2.reset_index(level=0, inplace=True)

    var_cols = ['{}_daily_{}'.format(var, col_type) for col_type in ['mean', 'std']]

    # add mean and std get from df1
    df2 = df2.merge(month_day[['month', 'day']+var_cols], how='left', on=['month', 'day'])


    df2 = df2.sort_values(by=[date_id])
    df2 = df2.set_index([date_id]) # add multi-index back


    df1[var+'_zscore'] = (df1[var]- df1['{}_daily_mean'.format(var)])/df1['{}_daily_std'.format(var)]
    df2[var+'_zscore'] = (df2[var]- df2['{}_daily_mean'.format(var)])/df2['{}_daily_std'.format(var)]



    df_all = df1.append(df2)

    df_all.to_hdf(rootpath + '{}_zscore.h5'.format(var), key=var, mode='w')



def zscore_spatial_temporal(rootpath,
                            target, var_id='tmp2m',
                            train_start='1986-01-01', train_end='2016-12-31',
                            test_start='2017-01-01', test_end='2018-12-31',
                            date_id='start_date'):

    """ Apply zscore on spatial-temporal climate variable, e.g., the target variable tmp2m

    Args:
            rootpath: directory to save the results
            data: pd.Dataframe -- dataframe contains data that is about to apply zscore
            var_id: variable name
            train_start, train_end: str -- the start date and the end date of the training set
            test_start, test_end: str -- the start date and the end date of the test set
            date_id: column name for time/date

    """

    idx = pd.IndexSlice


    df1 = target.loc[idx[:, :, train_start:train_end], :] # train
    df2 = target.loc[idx[:, :, test_start:test_end], :]# test



    # ---- Day-Month Mean of each location ---- #
    # Add 'month', 'day' column, and get mean and std of each date, each location
    df1 = df1.groupby(['lat', 'lon']).apply(lambda df: get_mean(df, var_id, date_id))

    #  get first element of each group: mean for each location each month-day 
    month_day = df1.groupby(['lat', 'lon', 'month', 'day']).first() 
    month_day = month_day.reset_index()

    # add month-day column to second dataframe
    df2 = df2.groupby(['lat', 'lon']).apply(lambda df: add_month_day(df, date_id))
    df2.reset_index(level=2, inplace=True)

    var_cols = ['{}_daily_{}'.format(var_id, col_type) for col_type in ['mean', 'std']]

    # add mean and std get from df1
    df2 = df2.merge(month_day[['lat', 'lon', 'month', 'day'] + var_cols],
                    how='left', on=['lat', 'lon', 'month', 'day'])


    df2 = df2.sort_values(by=['lat', 'lon', date_id])
    df2 = df2.set_index(['lat', 'lon', date_id]) # add multi-index back


    df1[var_id+'_zscore'] = (df1[var_id] - df1['{}_daily_mean'.format(var_id)])/df1['{}_daily_std'.format(var_id)]
    df2[var_id+'_zscore'] = (df2[var_id] - df2['{}_daily_mean'.format(var_id)])/df2['{}_daily_std'.format(var_id)]


    df_all = df1.append(df2)
    df_all.sort_index(level=['lat', 'lon'], inplace=True)


    df_all.to_hdf(rootpath + 'target_{}_multitask_zscore.h5'.format(var_id), key=var_id, mode='w')

############## train-validation split ##################

def create_sequence_custom(today, time_frame, covariate_map, past_years=2,
                           curr_shift_days=[7, 14, 18], past_shift_days=[7, 14, 28]):

    """ Feature aggregation: add features from past dates

    Args:
            today:  pd.Timestamp() -- the date we want to aggregate feature
            time_frame: pandas dataframe -- corresponding dates for covariate map
            covariate_map: numpy array -- data/feature we use to aggregate
            past_years: int -- number of years in the past to be included
            curr_shift_days: list of int -- past dates/neighbors in the current year/most recent year to be included
            past_shift_days: list of int -- both past and future dates/neighbors in the past year to be included

    Return:
            agg_x: numpy array -- the aggragated feature for the date provided by "today"

    """

    combine = [today] + [today - pd.DateOffset(days=day) for day in curr_shift_days]



    for k in range(past_years): # go to the past k years
        today = today - pd.DateOffset(years=1)

        past = [today - pd.DateOffset(days=day) for day in past_shift_days]
        future = [today+pd.DateOffset(days=day) for day in past_shift_days[::-1]]

        time_index_next = future + [today] + past

        combine = combine + time_index_next#combine.union(time_index_next)

    combine.reverse()# reverse the sequenc from oldest to newest

    location = time_frame.loc[combine]

    agg_x = covariate_map[location.values].squeeze()

    return agg_x 


def get_test_train_index_seasonal(test_start, test_end, train_range=10, past_years=2, gap=28):


    """ Construct train/test time index used to split training and test dataset

    Args:
            test_start, test_end: pd.Timestamp() -- the start date and the end date of the test set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length (years) of features in the past to be included
            gap: int -- number of days between the date in X and date in y

    Return:
            test_start_shift: pd.Timestamp() -- new start date for test
                                                after including # of years in the past
            train_start_shift:pd.Timestamp() -- new start date for training
                                                after including # of years in the past
            train_time_index: list of pd.Timestamp() -- time index for training set
            
    """    

    test_start_shift = test_start - pd.DateOffset(years=train_range + past_years, days=gap)
    

    # handles the train time indices
    # you need to gap 28 days to predict Feb-01 standing on Jan-03
    train_end = test_start - pd.DateOffset(days=gap)
    # train starts 10 years before the end date
    train_start = train_end - pd.DateOffset(years=train_range)
    
    # shift another two years to create the sequence
    train_start_shift = train_start- pd.DateOffset(years=past_years, days=gap) 

    train_time_index = pd.date_range(train_start, train_end) 

    return test_start_shift, train_start_shift, train_time_index


def train_val_split_target(rootpath,
                           target,
                           var_id,
                           val_year, val_month,
                           train_range=10,
                           past_years=2,
                           test_range=28,
                           test_freq='7D'):

    """ Generate Train-validation sets on the target variable tmp2m

    Args:
            rootpath: str -- the directory to save the results
            target: multi-index (spatial-temporal) pandas dataframe -- target data used
                    to construct training-validation set
            var_id: str -- the name of the target variable, e.g., tmp2m and precip
            val_year,val_month: int -- the year and the month for the validation set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length of features in the past to be included
            test_range: int -- the length (days) used in the validation set
            test_freq: str -- the frequency to generate dates in the validtion set

    """

    

    idx = pd.IndexSlice

    # check the legitimate of the given parameters 
    if not isinstance(target, pd.DataFrame):

        if isinstance(target, pd.Series):
            target = target.to_frame() # convert pd.series to pd.dataframe
        else:
            raise ValueError("Dataset needs to be a pandas dataframe")


    # check dataframe level!

    if len(target.index.names) < 3:
        raise ValueError("Multiindex dataframe includes 3 levels: [lat,lon,start_date]")

    # handles the test time indices
    test_start = pd.Timestamp('{}-{:02d}-01'.format(val_year, val_month), freq='D')
    test_end = test_start + pd.DateOffset(days=test_range)
    #[test_start,test_end]
    test_time_index = pd.date_range(test_start, test_end, freq=test_freq)

    test_start_shift, train_start_shift, train_time_index = get_test_train_index_seasonal(test_start,
                                                                                          test_end,
                                                                                          train_range,
                                                                                          past_years)
    train_end = train_time_index[-1]


    train_y = target['{}_zscore'.format(var_id)].to_frame().loc[idx[:, :, train_time_index], :]
    test_y = target['{}_zscore'.format(var_id)].to_frame().loc[idx[:, :, test_time_index], :]

    train_y = train_y.unstack(level=[0, 1]).values
    test_y = test_y.unstack(level=[0, 1]).values

    save_results(rootpath, 'train_y_pca_{}_forecast{}.pkl'.format(val_year, val_month), train_y)
    save_results(rootpath, 'val_y_pca_{}_forecast{}.pkl'.format(val_year, val_month), test_y)


def train_val_split_covariate(rootpath,
                              data,
                              val_year, val_month,
                              train_range=10, past_years=2,
                              test_range=28, test_freq='7D',
                              n_jobs=16):

    # pylint: disable-msg=too-many-locals
    """ Generate Train-validation sets for temporal covariates, e.g., PCs, climate indeces

    Args:
            rootpath: str -- the directory to save the results
            data: pandas dataframe -- covariates used to construct training-validation set
            val_year,val_month: int -- the year and the month for the validation set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length of features in the past to be included
            test_range: int -- the length (days) used in the validation set
            test_freq: str -- the frequency to generate dates in the validtion set
            n_jobs: int -- number of workers for parallel
    """


    idx = pd.IndexSlice

    # check the legitimate of the given parameters 
    if not isinstance(data, pd.DataFrame):

        if isinstance(data, pd.Series):
            data = data.to_frame() # convert pd.series to pd.dataframe
        else:
            raise ValueError("Dataset needs to be a pandas dataframe")


    # check dataframe level!

    if len(data.index.names) > 1:
        raise ValueError("Pandas dataframe for temporal data only!")


    # handles the test time indices
    test_start = pd.Timestamp('{}-{:02d}-01'.format(val_year, val_month), freq='D')
    test_end = test_start + pd.DateOffset(days=test_range)
    #[test_start,test_end]
    test_time_index = pd.date_range(test_start, test_end, freq=test_freq) 

    test_start_shift, train_start_shift, train_time_index = get_test_train_index_seasonal(test_start,
                                                                                          test_end,
                                                                                          train_range,
                                                                                          past_years)
    train_end = train_time_index[-1]

    train_x_norm = data.loc[idx[train_start_shift:train_end], :]
    test_x_norm = data.loc[idx[test_start_shift:test_end], :]
    
    time_index1 = pd.date_range(train_start_shift, train_end) 
    time_index2 = pd.date_range(test_start_shift, test_end) 

    df1 = pd.DataFrame(data={'pos':np.arange(len(time_index1))}, index=time_index1)# training index
    df2 = pd.DataFrame(data={'pos':np.arange(len(time_index2))}, index=time_index2)

    # to aggregate features for covariates
    train_x = np.asarray(Parallel(n_jobs=n_jobs)(delayed(create_sequence_custom)(date, df1['pos'],
                                                                                 train_x_norm.values) for date in train_time_index)) 
    test_x = np.asarray(Parallel(n_jobs=n_jobs)(delayed(create_sequence_custom)(date, df2['pos'],
                                                                                test_x_norm.values) for date in test_time_index)) 

    #print(train_X.shape,test_X.shape)


    save_results(rootpath, 'train_X_pca_{}_forecast{}.pkl'.format(val_year, val_month), train_x)
    save_results(rootpath, 'val_X_pca_{}_forecast{}.pkl'.format(val_year, val_month), test_x)


def train_val_split(rootpath,
                    data,
                    target, var_id,
                    val_year, val_month,
                    train_range=10, past_years=2,
                    test_range=28, test_freq='7D',
                    n_jobs=16):
    # pylint: disable-msg=too-many-locals

    """ Wrapper fucntion: Training and validation split for both temporal covariates and target


    Args:
            rootpath: str -- the directory to save the results
            data: pandas dataframe -- covariates used to construct training-test set
            target: multi-index (spatial-temporal) pandas dataframe -- target data used
                    to construct training-validation set
            var_id: str -- the name of the target variable, e.g., tmp2m and precip
            val_year,val_month: int -- the year and the month for the validation set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length of features in the past to be included
            test_range: int -- the length (days) used in the validation set
            test_freq: str -- the frequency to generate dates in the validtion set
            n_jobs: int -- number of workers for parallel
    """

    train_val_split_covariate(rootpath, data, val_year, val_month,
                              train_range, past_years, test_range, test_freq, n_jobs)
    train_val_split_target(rootpath, target, var_id, val_year, val_month,
                           train_range, past_years, test_range, test_freq)

# train-test split

def train_test_split_target(rootpath,
                            target,
                            var_id,
                            test_time_index_all,
                            test_year,
                            test_month,
                            train_range=24,
                            past_years=2):
    # pylint: disable-msg=too-many-locals

    """ Generate Train-test set on the target variable tmp2m

    Args:
            rootpath: str -- the directory to save the results
            target: multi-index (spatial-temporal) pandas dataframe -- target data used to construct training-validation set
            var_id: str -- the name of the target variable, e.g., tmp2m and precip
            test_time_index_all: list of pd.timestamp() -- the sequence of test dates to mimic the live evaluation,
                                                           test dates are choose from the list based on the month and the year
            test_year,test_month: int -- yyyy-mm the year and the month for the test set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length of features in the past to be included
            test_range: int -- the length (days) used in the validation set

    """



    idx = pd.IndexSlice

    # check the legitimate of the given parameters 
    if not isinstance(target, pd.DataFrame):

        if isinstance(target, pd.Series):
            target = target.to_frame() # convert pd.series to pd.dataframe
        else:
            raise ValueError("Dataset needs to be a pandas dataframe")


    # check dataframe level!

    if len(target.index.names) < 3:
        raise ValueError("Multiindex dataframe includes 3 levels: [lat,lon,start_date]")


    # handles the test time indices
    test_time_index = test_time_index_all[(test_time_index_all.month == test_month)
                                          & (test_time_index_all.year == test_year)]

    test_start = test_time_index[0]
    test_end = test_time_index[-1]

    test_start_shift, train_start_shift, train_time_index = get_test_train_index_seasonal(test_start,
                                                                                          test_end,
                                                                                          train_range=train_range,
                                                                                          past_years=past_years)
    train_end = train_time_index[-1]


    train_y = target['{}_zscore'.format(var_id)].to_frame().loc[idx[:, :, train_time_index], :]
    test_y = target['{}_zscore'.format(var_id)].to_frame().loc[idx[:, :, test_time_index], :]



    train_y = train_y.unstack(level=[0, 1]).values
    test_y = test_y.unstack(level=[0, 1]).values


    #print(train_y.shape,test_y.shape)

    save_results(rootpath, 'train_y_pca_{}_forecast{}.pkl'.format(test_year, test_month), train_y)
    save_results(rootpath, 'test_y_pca_{}_forecast{}.pkl'.format(test_year, test_month), test_y)


def train_test_split_covariate(rootpath,
                               data,
                               test_time_index_all,
                               test_year, test_month,
                               train_range=24, past_years=2,
                               n_jobs=16):


    """ Training and validation split for temporal covariates


    Args:
            rootpath: str -- the directory to save the results
            data: pandas dataframe -- covariates used to construct training-test set
            test_time_index_all: list of pd.timestamp() -- the sequence of test dates to mimic the live evaluation,
                                                           test dates are choose from the list based on the month and the year
            test_year,test_month: int -- yyyy-mm the year and the month for the test set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length of features in the past to be included
            n_jobs: int -- number of workers for parallel
    """


    idx = pd.IndexSlice

    # check the legitimate of the given parameters 
    if not isinstance(data, pd.DataFrame):

        if isinstance(data, pd.Series):
            data = data.to_frame() # convert pd.series to pd.dataframe
        else:
            raise ValueError("Dataset needs to be a pandas dataframe")


    # check dataframe level!

    if len(data.index.names) > 1:
        raise ValueError("Pandas dataframe for temporal data only!")

    # handles the test time indices
    test_time_index = test_time_index_all[(test_time_index_all.month == test_month) & (test_time_index_all.year == test_year)]

    test_start = test_time_index[0]
    test_end = test_time_index[-1]

    test_start_shift, train_start_shift, train_time_index = get_test_train_index_seasonal(test_start,
                                                                                          test_end,
                                                                                          train_range=train_range,
                                                                                          past_years=past_years)
    train_end = train_time_index[-1]
        



    train_x_norm = data.loc[idx[train_start_shift:train_end], :]
    test_x_norm = data.loc[idx[test_start_shift:test_end], :]
    
    time_index1 = pd.date_range(train_start_shift, train_end) 
    time_index2 = pd.date_range(test_start_shift, test_end) 
    df1 = pd.DataFrame(data={'pos':np.arange(len(time_index1))}, index=time_index1)# training index
    df2 = pd.DataFrame(data={'pos':np.arange(len(time_index2))}, index=time_index2)

    # to aggregate features for covariates

    train_x = np.asarray(Parallel(n_jobs=n_jobs)(delayed(create_sequence_custom)(date, df1['pos'], train_x_norm.values)
                                                 for date in train_time_index)) #(3627, 18, 70)
    test_x = np.asarray(Parallel(n_jobs=n_jobs)(delayed(create_sequence_custom)(date, df2['pos'], test_x_norm.values)
                                                for date in test_time_index)) #(5, 18, 70)

    #print(train_X.shape,test_X.shape)

    save_results(rootpath, 'train_X_pca_{}_forecast{}.pkl'.format(test_year, test_month), train_x)
    save_results(rootpath, 'test_X_pca_{}_forecast{}.pkl'.format(test_year, test_month), test_x)





def train_test_split(rootpath,
                     data,
                     target, var_id,
                     test_time_index_all,
                     test_year, test_month,
                     train_range=24, past_years=2,
                     n_jobs=16):

    """ Wrapper fucntion: Training and test split for both temporal covariates and target


    Args:
            rootpath: str -- the directory to save the results
            data: pandas dataframe -- covariates used to construct training-test set
            test_time_index_all: list of pd.timestamp() -- the sequence of test dates to mimic the live evaluation,
                                                           test dates are choose from the list based on the month and the year
            test_year,test_month: int -- yyyy-mm the year and the month for the test set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length of features in the past to be included
            n_jobs: int -- number of workers for parallel
    """

    

    train_test_split_covariate(rootpath, data, test_time_index_all,
                               test_year, test_month, train_range, past_years, n_jobs)

    train_test_split_target(rootpath, target, var_id, test_time_index_all,
                            test_year, test_month, train_range, past_years)


def train_test_split_target_ar(rootpath,
                               target, var_id,
                               test_time_index_all,
                               test_year, test_month,
                               train_range=24, past_years=2):

    """ Training and test split for target variable to build AR model


    Args:
            rootpath: str -- the directory to save the results
            target: multi-index (spatial-temporal) pandas dataframe -- target data used to construct training-validation set
            var_id: str -- the name of the target variable, e.g., tmp2m and precip
            test_time_index_all: list of pd.timestamp() -- the sequence of test dates to mimic the live evaluation,
                                                           test dates are choose from the list based on the month and the year
            test_year,test_month: int -- yyyy-mm the year and the month for the test set
            train_range: int -- the length (years) to be included in the training set
            past_years: int -- the length of features in the past to be included
    """

    idx = pd.IndexSlice


    # handles the test time indices
    test_time_index = test_time_index_all[(test_time_index_all.month == test_month)
                                          & (test_time_index_all.year == test_year)]

    test_start = test_time_index[0]
    test_end = test_time_index[-1]

    test_start_shift, train_start_shift, train_time_index = get_test_train_index_seasonal(test_start,
                                                                                          test_end,
                                                                                          train_range=train_range,
                                                                                          past_years=past_years) 
    train_end = train_time_index[-1]


    train_time_shift_index = pd.date_range(train_start_shift, train_end)
    test_time_shift_index = pd.date_range(test_start_shift, test_end)


    # you have to have all data here, including the past 2 years data, so use shifted index
    train_y_norm = target['{}_zscore'.format(var_id)].to_frame().loc[idx[:, :, train_time_shift_index], :]
    test_y_norm = target['{}_zscore'.format(var_id)].to_frame().loc[idx[:, :, test_time_shift_index], :]

    train_y_norm = train_y_norm.unstack(level=[0, 1])
    test_y_norm = test_y_norm.unstack(level=[0, 1])


    # aggragate data into sequence #
    # multi-index dataframe with lat,lon, start_date
    time_index1 = train_y_norm.index.get_level_values('start_date') 
    time_index2 = test_y_norm.index.get_level_values('start_date')
    # training index
    df1 = pd.DataFrame(data={'pos':np.arange(len(time_index1))}, index=time_index1)
    df2 = pd.DataFrame(data={'pos':np.arange(len(time_index2))}, index=time_index2)


    train_y = np.asarray(Parallel(n_jobs=16)(delayed(create_sequence_custom)(date, df1['pos'], train_y_norm.values, 2, [28, 42, 56, 70])
                                             for date in train_time_index))
    test_y = np.asarray(Parallel(n_jobs=16)(delayed(create_sequence_custom)(date, df2['pos'], test_y_norm.values, 2, [28, 42, 56, 70])
                                            for date in test_time_index)) 

    train_y = np.swapaxes(train_y, 1, 2)
    test_y = np.swapaxes(test_y, 1, 2)

    train_y = train_y[:, :, :-1]
    test_y = test_y[:, :, :-1]

    print(train_y.shape, test_y.shape)
   

    save_results(rootpath, 'train_y_pca_ar_{}_forecast{}.pkl'.format(test_year, test_month), train_y)
    save_results(rootpath, 'test_y_pca_ar_{}_forecast{}.pkl'.format(test_year, test_month), test_y)




