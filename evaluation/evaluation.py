import pickle
import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy import stats
import sys
sys.path.append('../')
from utils import *


def compute_rmse(target, prediction):
    """Compute rmse between the ground truth and forecasts
    Args:
    target: a numpy array with ground truth
    forecasts: a numpy array with forecasted values
    Returns: rmse between the ground truth and forecasts
    """
    return np.sqrt(mean_squared_error(target, prediction))


def compute_cosine(target, prediction):
    """Compute cosine simialrity between the ground truth and forecasts
    Args:
    target: a numpy array with ground truth
    forecasts: a numpy array with forecasted values
    Returns: cosine simialrity between the ground truth and forecasts
    """
    result = np.dot(target, prediction) / (LA.norm(target) * LA.norm(prediction))
    return result


def r_squared(y_true, y_pred, y_mean=None):
    """Compute relative R^2 between the ground truth and forecasts
    Args:
    target: a numpy array with ground truth
    forecasts: a numpy array with forecasted values
    Returns: relative R^2 between the ground truth and forecasts
    """
    if y_mean is None:
        y_mean = np.zeros(y_true.shape[0]) * np.mean(y_true)
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - y_mean)**2)
    rsq = 1 - rss / tss
    return rsq


def print_eval_stats(eval_result):
    """Print the mean(se), median(se), 0.25 quantile(se), and 0.75 quantile (se) of the array, where se represents standard deviation
    Args:
    eval_result: a numpy array with evluation results
    """
    print('mean: {:.4f} ({:.4f}) median {:.4f} ({:.4f})'.format(np.mean(eval_result),
                                                                stats.sem(eval_result),
                                                                np.median(eval_result),
                                                                quantile_se(eval_result, p=50)))
    print('0.25 quantile: {:.4f} ({:.4f}) 0.75 quantile: {:.4f} ({:.4f})'.format(np.quantile(eval_result, 0.25),
                                                                                 quantile_se(eval_result, p=25),
                                                                                 np.quantile(eval_result, 0.75),
                                                                                 quantile_se(eval_result, p=75)))


def quantile_se(x, p=50):
    # compute the standard error for different quantiles
    # Source: Jun Shao, "Mathematical Statistics". Springer Texts in Statistics, 1999. Page 306: Theorem 5.10
    # p: quantile: int between 0-100
    # x: data sequence
    n = len(x)  # number of samples
    q = np.percentile(x, p)
    density = stats.gaussian_kde(x)  # density estimate of x
    Fp = density(q).item()
    p = p / 100.
    sF = np.sqrt(p * (1 - p)) / Fp
    se = sF / np.sqrt(n)
    return se


def eval_forecast(model_name, rootpath, test_years, month_range, rep=False, num_rep=10):
    """Evalute the forecasts on training and test sets
    Args:
    model_name: a string indicating the name of a model
    rootpath: the path where the forecasts are saved
    test_years: a list of years in the test set
    month_range: a list of months in the test set
    rep: True or False, indicating if the reults include repeated runs
    num_rep: the number of repetition
    Returns:
    result_train: the forecasting performance (temporal/spatial cosine/r2) on training set
    result_test: the forecasting performance (temporal/spatial cosine/r2) on test set
    """
    target_train = []
    target_test = []
    prediction_train = []
    prediction_test = []
    for year in test_years:
        for month_id in month_range:
            result_temp = load_results(rootpath + 'forecast_results/results_{}_{}_{}.pkl'.format(model_name, year, month_id))
            target_train.append(result_temp['target_train'])
            target_test.append(result_temp['target_test'])
            if rep is True:
                prediction_train_temp = np.zeros(result_temp['target_train'].shape)
                prediction_test_temp = np.zeros(result_temp['target_test'].shape)
                for i in range(num_rep):
                    prediction_train_temp += result_temp['prediction_train'][i]
                    prediction_test_temp += result_temp['prediction_test'][i]
                prediction_train.append(prediction_train_temp / float(num_rep))
                prediction_test.append(prediction_test_temp / float(num_rep))
            else:
                prediction_train.append(result_temp['prediction_train'])
                prediction_test.append(result_temp['prediction_test'])
    # test set evaluation
    prediction_test = np.concatenate(prediction_test, axis=0)
    target_test = np.concatenate(target_test, axis=0)
    temporal_cos = np.zeros(prediction_test.shape[0])
    spatial_cos = np.zeros(prediction_test.shape[1])
    temporal_r2 = np.zeros(prediction_test.shape[0])
    spatial_r2 = np.zeros(prediction_test.shape[1])
    for i in range(prediction_test.shape[0]):
        temporal_cos[i] = compute_cosine(target_test[i, :], prediction_test[i, :])
        temporal_r2[i] = r_squared(target_test[i, :], prediction_test[i, :])
    for i in range(prediction_test.shape[1]):
        spatial_cos[i] = compute_cosine(target_test[:, i], prediction_test[:, i])
        spatial_r2[i] = r_squared(target_test[:, i], prediction_test[:, i])
    result_test = {}
    result_test['temporal_cos'] = temporal_cos
    result_test['spatial_cos'] = spatial_cos
    result_test['temporal_r2'] = temporal_r2
    result_test['spatial_r2'] = spatial_r2
    # training set evaluation
    prediction_train = np.concatenate(prediction_train, axis=0)
    target_train = np.concatenate(target_train, axis=0)
    temporal_cos_train = np.zeros(prediction_train.shape[0])
    spatial_cos_train = np.zeros(prediction_train.shape[1])
    temporal_r2_train = np.zeros(prediction_train.shape[0])
    spatial_r2_train = np.zeros(prediction_train.shape[1])
    for i in range(prediction_train.shape[0]):
        temporal_cos_train[i] = compute_cosine(target_train[i, :], prediction_train[i, :])
        temporal_r2_train[i] = r_squared(target_train[i, :], prediction_train[i, :])
    for i in range(prediction_train.shape[1]):
        spatial_cos_train[i] = compute_cosine(target_train[:, i], prediction_train[:, i])
        spatial_r2_train[i] = r_squared(target_train[:, i], prediction_train[:, i])
    result_train = {}
    result_train['temporal_cos'] = temporal_cos_train
    result_train['spatial_cos'] = spatial_cos_train
    result_train['temporal_r2'] = temporal_r2_train
    result_train['spatial_r2'] = spatial_r2_train
    return result_train, result_test
