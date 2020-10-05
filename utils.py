import numpy as np
import pandas as pd
import pickle


def load_results(filename):
    """load a pickle file

    Args:
    filename: the path + file name for the file to be loaded

    Returns: a pickle file
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save_results(filename, results):
    """save a pickle file

    Args:
    filename: the path + file name for the file to be saved
    results: the data array to be saved
    """
    with open(filename, 'wb') as fh:
        pickle.dump(results, fh)
