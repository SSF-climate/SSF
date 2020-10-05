## Sub-Seasonal Climate Forecasting via Machine Learning: Challenges, Analysis, and Advances

Code for data extraction, preprocessing, and SSF model training and evaluation used in He et al. [Sub-Seasonal Climate Forecasting via Machine Learning: Challenges, Analysis, and Advances](https://arxiv.org/abs/2006.07972).  

## Requirments
The code is compatible with Python 3.6 and the following packages:
- numpy: 1.19.0
- pandas: 0.24.2
- joblib: 0.15.1
- pickle: 4.0
- scipy: 1.5.0
- pytorch: 1.2.0
- sklearn: 0.23.1
- xgboost: 1.0.2

## Project Structure
- cfg_target.py: configure file with all parameters from users
- create_covariates_pca.py: script for concatenating PCs from all spatiotemporal covariates and temporal covariates
to construct the pandas dataframe for final covariates
- create_train_test_sets.py: script for creating training and test sets
- create_train_validation_sets.py: script for creating training and validation sets
- data_load: folder contains functions for data loading
- evaluation: folder contains functions for evaluation
- forecasting: folder contains scripts for training forecasting models and evluate on test sets
- hyperparameter_tuning: folder contains functions for random search
- load_data: script for a subset of data required by configure file (cfg_target.py)
- main_experiments: script for runing experiments for all models on training and test sets
- model: collection of models implemented in experiments
- preprocess: folder contains functions for data preprocessing
- preprocess_covariate.py: script for preprocessing covariates
- preprocess_target.py: script for preprocessing target variables
- run_evaluation.py: script for evaluate the performance of forecasting models on training and test sets
- run_preprocess.py: script for data preprocessing
- run_random_cv.py: script for hyperparameter tuning
- utils:utility functions

## Getting started
1. Clone the repo
2. Create virtual environments and install the necessary python packages listed above
3. Load raw data from the google drive folder
4. Revise configure file (cfg_target.py) to adpat to the required settings
5. Data loading and preprocessing:
    1. Execute load_data.py to load the subset of data needed in generating forecasts
    2. Execute run_preprocess.py to preprocess covariates and target variable separately
    3. Execute create_covariates_pca.py to concatenate data
    4. Execute create_train_validation_sets.py and create_train_test_sets.py to create training-validation sets and training-test sets
6. Hyperparameter tuning: execute run_random_cv.py to find the best parameter by random search
7. Generate forecasts: execute main_experiments.py to train all the models and generate forecasts on test sets
8. Evaluate forecasts: execute run_evaluation.py to evalute the forecasting performance on both training and test sets
