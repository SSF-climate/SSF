"""
Run random_cv.py (in the folder hyperparameter_tuning) for hyper parameter tuning
"""
import os
import cfg_target
import pickle
import numpy as np
from utils import *


model_names = cfg_target.model_names  # ['EncoderFNN_AllSeq', 'EncoderDecoder', 'EncoderFNN']
val_years = cfg_target.val_years  # [2012, 2013, 2014, 2015, 2016]
month_range = cfg_target.month_range
rootpath = cfg_target.rootpath_cv
metric = cfg_target.cv_metric


def best_hyperparameter(val_years, month_range, eval_metrics, model_name, rootpath):
    """Find the best hyper parameters based on the results on validation set

    Args:
    val_years: an array with the years considered for validation set
    month_range: an array with the months for hyperparameter tuning
    eval_metrics: a string indicating the evaluation metric ('cos' or 'rmse')
    model_name: a string, the name of the model for hyperparameter tuning
    rootpath: the path where the validation results are saved
    """
    for month in month_range:
        score_all = []
        for year in val_years:
            cv_results = load_results(rootpath + 'cv_results_test/cv_results_' + model_name + '_{}_{}.pkl'.format(year, month))
            score = cv_results['score']
            score_all.append(score)
        score_all = np.asarray(score_all).squeeze()

        if eval_metrics == 'cos':
            best_score = score_all[:, :, 1].mean(axis=0)
            best_id = np.where(best_score == best_score.max())
        elif eval_metrics == 'rmse':
            best_score = score_all[:, :, 0].mean(axis=0)
            best_id = np.where(best_score == best_score.min())

        best_parameter = cv_results['parameter_all'][best_id[0][0]]

        save_results(rootpath + 'cv_results_test/best_parameter/{}_forecast{}.pkl'.format(model_name, month), best_parameter)


for model_name in model_names:
    for year in val_years:
        cmd = "{} {} --year {} --model_name {}".format("python", 'hyperparameter_tuning/random_cv.py', year, model_name)
        print(cmd)
        os.system(cmd)
    print('find the best hyper parameter for {}'.format(model_name))
    best_hyperparameter(val_years=val_years, month_range=month_range, eval_metrics=metric, model_name=model_name, rootpath=rootpath)
