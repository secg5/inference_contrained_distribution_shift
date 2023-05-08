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



class WeightedLogisticRegression(nn.Module):
    """Logistic regression.

    Basic classification model to use as a basis for an optimization problem.
    """

    def __init__(self, num_features, num_labels):
        super(WeightedLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=num_features, 
                                      out_features=num_labels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        """Standard computations for produce a logistic regression."""
        features = self.linear(features)
        return self.sigmoid(features)