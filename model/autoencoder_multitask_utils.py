from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


class MapDataset_quad(Dataset):
    """
    X: of size [num_samples, num_features], is the covariates matrix
    y: of size [num_samples,num_locations], is the target variable, which has the same length as each element in X
    C: of size [num_samples, num_locations, num_locations], is the covariance/correlation matrix among all locations.
    """
    def __init__(self, X, y, C):
        self.data = X
        self.labels = y
        self.cov = C
        self.num_var = len(X)

    def __len__(self):
        return len(self.labels)  # of how many examples you have

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.cov[index]


class MapDataset(Dataset):
    """
    X: of size [num_samples, num_features], is the covariates matrix
    y: of size [num_samples,num_locations], is the target variable, which has the same length as each element in X
    """
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.labels)  # of how many examples(images?) you have

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class MapDataset_ar(Dataset):
    """
        x1: of size [num_samples, num_features], is the covariates matrix
        x2: a list of historical target variable for autoregression
        y: of size [num_samples,num_locations], is the target variable, which has the same length as each element in X
    """
    def __init__(self, x1, x2, y):
        self.data = x1
        self.target = x2
        self.labels = y

    def __len__(self):
        return len(self.labels)  # of how many examples(images?) you have

    def __getitem__(self, index):
        return self.data[index], self.target[index], self.labels[index]


class MapDataset_CNN(Dataset):
    """
    X: a list containing covariates' map, len(X) = num_variables,
    y: of size [num_samples,num_locations], is the target variable, which has the same length as each element in X
    """
    def __init__(self, X, y):
        self.data = X
        self.labels = y
        self.num_var = len(X)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        var = []
        for i in range(self.num_var):
            var.append(self.data[i][index])
        return var, self.labels[index]


def init_weight(mdl):
    """Initalize the weights for each deep learning model
    Args:
    mdl: a deep learning model
    """
    for name, param in mdl.named_parameters():
        if 'weight' in name:
            param = nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))

# define quadratic loss: y^T*C*y


def quad_loss(prediction: torch.Tensor, target: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    # implementation of Quadratic loss: (y-f(x))^T*C^-1*(y-f(x))
    # check dimension of prediction and target, make sure they are vectors of size dx1
    # target: torch.Tensor in the shape of batch_size x num_locations (100x197)
    # prediction: made by neural network, also in the shape of batch_size x num_locations (100x197)
    # C: Covariance/correlation matrix contains the spatial information among locations,
    # in the shape of batch_size x num_locations x num_locations (100x197x197)
    target = target.unsqueeze(1)  # convert into the size of batch_size x 1 x num_locations to do batch version of matrix multiplication

    prediction = prediction.unsqueeze(1)
    res = target - prediction

    loss = torch.bmm(torch.bmm(res, C), torch.transpose(res, 1, 2)) / res.shape[-1]

    return loss.mean()


def epsilon_loss(target, prediction, epsilon):
    # epsilon_loss if residual (target-prediction) is less than epsilon, it is computed as 0 in the loss function
    res = target - prediction

    loc0 = torch.where(res.abs() <= epsilon)
    loc_plus = torch.where(res.clone() > epsilon)
    loc_minus = torch.where(res.clone() < -epsilon)

    res[loc0] = 0  # if |res|<= epsilon => 0
    res[loc_plus] = res.clone()[loc_plus].clone() - epsilon  # if res > epsilon, then  res = res-epsilon
    res[loc_minus] = res.clone()[loc_minus] + epsilon  # if res<-epsilon, then res = res+epsilon

    return torch.mean(res**2)
