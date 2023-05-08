import scipy
import datetime

import torch
import numpy as np
import cvxpy as cp

import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from folktables import ACSDataSource, ACSEmployment

def compute_ate_conditional_mean(A, y):
    """Computes the ate using inverse propensity weighting.

    Args:
        A (_type_): vector of observed outcomes
        propensity_scores (_type_): Vector of propensity scores
        y (_type_): response variable
    
    Returns
        ate: Average treatment effect.
    """
    conditional_mean = (y*A).sum()
    return conditional_mean/A.sum()


def compute_ate_ipw(A, propensity_scores, y):
    """Computes the ate using inverse propensity weighting.

    Args:
        A (_type_): vector of observed outcomes
        propensity_scores (_type_): Vector of propensity scores
        y (_type_): response variable
    
    Returns
        ate: Average treatment effect.
    """
    ipw_1 = A*(1/propensity_scores)
    ipw_0 = (1 - A)*(1/(1-propensity_scores))
    ate = (y*ipw_1 - y*ipw_0).mean()
    return ate


def empirical_propensity_score(data, levels):
    """Computes the empirical propensity scores.
     
    Computes the empirical propensity scores for each combination of 
    feature values.
    """
    unique_vals = levels[1:]
    A = levels[0][1]
    counts = np.zeros([len(lst) for lst in unique_vals])
    tot_counts = np.zeros([len(lst) for lst in unique_vals])

    def retrieve_strata(row, strata):
        example = [0]*len(strata)
        for i, feature in enumerate(strata):
            for j, val in enumerate(feature):  
                if row[val] == 1:
                    example[i] = j

        return tuple(example)

    for index in range(data.shape[0]):

        row = data.iloc[index]
        example = retrieve_strata(row, unique_vals)

        if row[A] == 1:
            counts[example] += 1
        tot_counts[example] += 1
    
    tot_counts[tot_counts == 0] = 0.00001
    probs = counts / tot_counts

    probs[probs == 0] = 0.00001
    probs[probs == 1] = 1 - 0.00001
  
    empirical_probs = np.zeros([data.shape[0]])
    for index in range(data.shape[0]):
        row =  data.iloc[index]
        example = retrieve_strata(row, unique_vals)
        empirical_probs[index] = probs[example]
    return empirical_probs, tot_counts

def build_counts(data : pd.DataFrame, levels: List[List], target:str):
    """
    Given a dummie variable dataset this function returns a matrix counts
    from all posible feature strata.
    """
    shape = [0 for i in range(len(levels) + 1)]  

    # 2 comes from the response variable Y (in this case binary)
    shape[0] = 2
    
    for index, level in enumerate(levels):
        number_levels = len(level)
        shape[index + 1] = number_levels
        
    count = torch.zeros(shape)

    for _, row in data.iterrows():
        position = [0 for i in range(len(levels) + 1)]
        position[0] = int(row[target])
        
        for index, level in enumerate(levels):
            for index_j, feature in enumerate(level):
                if row[feature] == 1:
                   position[index + 1] = index_j
        count[tuple(position)] += 1

    count[count == 0] = 0.00001
    return count  

def build_dataset(X: np.ndarray, group: np.ndarray):
    data = pd.DataFrame()
    levels = []
    for column in range(X.shape[1]):
        features = pd.get_dummies(X[:,column])
        strata_number = features.shape[1]
        names = [str(column) + "_" + str(j) for j in range(int(strata_number))]
        data[names] = 0 
        data[names] = features
        levels.append(names)
    data[["white", "non-white"]] = pd.get_dummies(group)
    return data, levels

def build_strata_counts_matrix(weight_features: torch.Tensor, 
                               counts: torch.Tensor, level: List[str]):
    """Builds linear restrictions for a convex optimization problem.
    
    This method build a Matrix full of counts by combination of strata.
    It will return two matrixes each one associated to the idividuals
    with y=0 or y=1:
    
    A_1 (a_{ij}), a_{ij} is the number of observations in the dataset
    such that y=1 x_i = 1 and x_j =1.
    
    A_0 (a_{ij}), a_{ij} is the number of observations in the dataset
    such that y=1 x_i = 1 and x_j =1.
    
    Note that unlike i, j is fixed, in the code outside this method
    varies trough female and male.
    
    returns A_0, A_1
    """
    _, features = weight_features.shape
    level_size = len(level)
    
    y_0_ground_truth = torch.zeros(level_size, features)
    y_1_ground_truth = torch.zeros(level_size, features)
    
    data_count_0 = counts[0]
    data_count_1 = counts[1]

    for level in range(level_size):
        t = data_count_0[level].flatten().unsqueeze(1)
        features = weight_features[level*t.shape[0]:(level + 1)*t.shape[0]]        
        y_0_ground_truth[level] = (features*t).sum(dim=0)

    
    for level in range(level_size):
        t_ = data_count_1[level].flatten().unsqueeze(1)    
        features_ = weight_features[level*t.shape[0]:(level + 1)*t.shape[0]]        
        y_1_ground_truth[level] = (features_*t_).sum(dim=0)
    
    return y_0_ground_truth, y_1_ground_truth

def get_folks_tables_data():
    """Creates the features used for the folk's tables experiments. The features
    are:
    ACSEmployment = folktables.BasicProblem(
        features=[
            'AGEP',
            'SCHL',
            'MAR',
            'RELP',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            'SEX',
            'RAC1P',
        ],
    Returns:
        _type_: _description_
    """
    data_source = ACSDataSource(survey_year='2018', 
                                horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    X, label, group = ACSEmployment.df_to_numpy(acs_data)
    group = 1*(group == 1)

    X = X.astype(int)
    label = label.astype(int)
    # last feature is the group
    X_norace = X[:,-8: -1]
    # Now sex is the last one, change inplace modification source of errors
    sex = X_norace[:, -1]
    return X_norace, label, sex, group