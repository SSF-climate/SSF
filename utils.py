import numpy as np
import pandas as pd
import cfg_target
import pickle
from random import randint
from random import seed
import torch
import model
from joblib import Parallel, delayed

def load_results(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f,encoding='bytes')
    return data


def save_results(filename,results):
    with open(filename, 'wb') as fh:
        pickle.dump(results, fh)
