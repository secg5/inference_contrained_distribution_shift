import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from itertools import accumulate
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer




class WeightedLogisticRegression(nn.Module):
    """Logistic regression.

    Basic classification model to use as a basis for an optimization problem.
    """

    def __init__(self, X, target ,weights_array):
        super(WeightedLogisticRegression, self).__init__()
        num_features =  X.shape[1]
        dataset_size = X.shape[0]
        beta = cp.Variable((num_features, 1)) 
        b = cp.Variable((1, 1))
        
        alpha = cp.Parameter((weights_array.shape[1], 1), nonneg=True)
        weights = weights_array @ alpha
        aux = cp.multiply(target, X @ beta + b) - cp.logistic(X @ beta + b)
        log_likelihood = (1. / dataset_size) * (aux.T @ weights)
        prob = cp.Problem(cp.Maximize(log_likelihood)) 
        self.fit_logreg = CvxpyLayer(prob, parameters=[weights], variables=[beta,b])
        

    def forward(self, alpha_weights):
        """Standard computations for produce a logistic regression."""
        betas = self.fit_logreg(alpha_weights)
        return betas