"""The following file executes a training pipe-line 
    for a vectorized robust debiaser."""
import scipy
import torch
import itertools
import datetime

import pandas as pd
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

from typing import List

DATASET_SIZE = 100000 

def build_counts(data : pd.DataFrame, levels: List[List], target:str):
    """
    Given a dummie variable dataset it return a matrix counts
    from all posible combinations of features
    """
    
    shape = [0 for i in range(len(levels) + 1)]        
    # 2 comes from the response variable Y in this case binary
    shape[0] = 2
    
    for index, level in enumerate(levels):
        number_levels = len(level)
        shape[index + 1] = number_levels
        
    count = torch.zeros(shape)

    for _, row in data.iterrows():
        position = [0 for i in range(len(levels) + 1)]
        # import pdb; pdb.set_trace()
        position[0] = int(row[target])
        
        for index, level in enumerate(levels):
            for index_j, feature in enumerate(level):
                if row[feature] == 1:
                   position[index + 1] = index_j
        count[tuple(position)] += 1
    # Machetimbis para que después no me estallen los gradientes
    count[count == 0] = 0.00001
    return count       

def build_strata_counts_matrix(weight_features: torch.Tensor, 
                               counts: torch.Tensor, level: List[str]):
    """Builds linear restrictions for a convex opt problem.
    
    This method build a Matrix with counts by combination of strata,
    It will return two matrixes each one associated to the idividuals
    with y=0 or y=1:
    
    A_1 (a_{ij}), a_{ij} is the number of observations in the dataset
    such that y=1 x_i = 1 and x_j =1.
    
    A_0 (a_{ij}), a_{ij} is the number of observations in the dataset
    such that y=1 x_i = 1 and x_j =1.
    
    Note that unlike i, j is fixed, in the code outside this method
    varies trough female and male
    
    returns A_0, A_1
    """
    _, features = weight_features.shape
    level_size = len(level)
    # This should be associated with the y?
    
    y_0_ground_truth = torch.zeros(level_size, features)
    y_1_ground_truth = torch.zeros(level_size, features)
    
    data_count_0 = counts[0]
    data_count_1 = counts[1]
    
    for level in range(level_size):
        # Flaw here only works with the first level.
        t = data_count_0[level].flatten().unsqueeze(1)
        features = weight_features[level*t.shape[0]:(level + 1)*t.shape[0]]        
        y_0_ground_truth[level] = (features*t).sum(dim=0)

        
    for level in range(level_size):
        # Flaw here only works with the first level.
        t_ = data_count_1[level].flatten().unsqueeze(1)    
        features_ = weight_features[weight_features.shape[0]//2 + level*t.shape[0]:weight_features.shape[0]//2 + (level + 1)*t.shape[0]]        
        y_1_ground_truth[level] = (features_*t_).sum(dim=0)
        

    return y_0_ground_truth, y_1_ground_truth

def simulate_multiple_outcomes(dataset_size: int, _feature_number:int = 4):
    """_summary_

    Args:
        dataset_size (int): _description_

    Returns:
        _type_: _description_
    """
    
    X = np.random.choice(a=[0, 1, 2], size=dataset_size, p=[0.5, 0.3, 0.2])
    X_2 =np.random.binomial(size=dataset_size, n=1, p=0.4)

    # Changes ORIGINAL ONE X - X_2
    pi_A = scipy.special.expit(X_2 - X)
    A = 1*(pi_A > np.random.uniform(size=dataset_size))

    mu = scipy.special.expit(2*A - X + X_2)
    y = 1*(mu > np.random.uniform(size=dataset_size))

    # Changes ORIGINAL ONE X + X_2
    obs = scipy.special.expit(X - X_2) > np.random.uniform(size=dataset_size)
    X_total = np.stack((X, X_2), axis=-1)

    return X_total, A, y, obs
    
def create_dataframe(X, A):
    skewed_data = pd.DataFrame()
    skewed_data[["little", "moderate", "quite rich"]] = pd.get_dummies(X[:,0])
    skewed_data[["White", "Non White"]] = pd.get_dummies(X[:,1])
    skewed_data[["female", "male"]] = pd.get_dummies(A)
    return skewed_data


# def compute_ATE(counts_1, counts_0):
#     """Computes the ATE for a given dataset."""
#     # int_x (p(y|x, A = 0, R = 1) - p(y|x, A = 1, R = 1))p(x)
#     sex = 1
#     sex_base = 0
    
#     probs = counts_1/(counts_1 + counts_0)

#     prob_x = counts_0[sex] + counts_0[sex_base]+ counts_1[sex] + counts_1[sex_base]
#     ATE = ((probs[sex_base])*prob_x/prob_x.sum()).sum()
#     return ATE

def compute_conditional_ATE(counts_1, counts_0):
    """Computes the conditional ATE for a given dataset."""
    # int_x (p(y|x, A=0) - p(y|x, A=1))p(x|A=1)
    sex = 1
    sex_base = 0
    
    probs = counts_1/(counts_1 + counts_0)

    total_weight_count = counts_1[sex] + counts_0[sex]
    c_ATE = ((probs[sex] - probs[sex_base])*total_weight_count/total_weight_count.sum()).sum()
    return c_ATE

def compute_debias_ate_ipw(A, propensity_scores_A, propensity_scores_R, y, obs):
    """Computes the ate using inverse propensity weighting.

    Args:
        A (_type_): vector of observed outcomes
        propensity_scores (_type_): Vector of propensity scores
        y (_type_): response variable
    
    Returns
        ate: Averga treatment effect.
    """
    propensity_score = propensity_scores_A*propensity_scores_R
    propensity_score_0 = (1-propensity_scores_A)*( 1 - propensity_scores_R)
    ipw_1 = A*obs*(1/propensity_score)
    ipw_0 = (1 - A)*(1/propensity_score_0)
    ate = (y*ipw_1).mean()
    return ate

def compute_ate_ipw(A, propensity_scores, y):
    """Computes the ate using inverse propensity weighting.

    Args:
        A (_type_): vector of observed outcomes
        propensity_scores (_type_): Vector of propensity scores
        y (_type_): response variable
    
    Returns
        ate: Averga treatment effect.
    """
    ipw_1 = A*(1/propensity_scores)
    ipw_0 = (1 - A)*(1/(1-propensity_scores))
    ate = (y*ipw_1).mean()
    return ate


def propensity_score_matching(weights_0, weights_1, data):
    """_summary_

    Args:
        weights_1 (_type_): _description_
        weights_2 (_type_): _description_
    """
    #  TODO (debug)
    weights = np.zeros(data.shape[0])
    for index, row  in data.iterrows():
        if row["Creditability"] == 0 & row["female"] == 1:
            weights[index] = weights_0[0]
        elif row["Creditability"] == 0 & row["female"] == 0:
            weights[index] = weights_0[1]
        elif row["Creditability"] == 1 & row["male"] == 0:
            weights[index] = weights_1[0]
        else:
            weights[index] = weights_1[1]
    return weights
    
    

def prpensity_scores_by_strata(levels):
  
    # scipy.special.expit(X_2 - X)
    # calculate the frequency of occurrence of each combination of feature values
    shape = [0 for i in range(len(levels))]        
    
    for index, level in enumerate(levels):
        number_levels = len(level)
        shape[index] = number_levels
        
    propensity_scores = torch.zeros(shape)

    for i in range(3):
        for j in range(2):
            propensity_scores[i,j] = scipy.special.expit(j - i)

    return propensity_scores
   
def run_search(A_0, A_1,data_count_1, data_count_0, weights_features, upper_bound, gt_ate, propensity_scores):
    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True)
    W = np.unique(weights_features.numpy(), axis=0)
    optim = torch.optim.Adam([alpha], 0.01)
    scheduler = StepLR(optim, step_size=500, gamma=0.1)
    loss_values = []
    for iteration in range(2000):
        w = cp.Variable(alpha.shape[0])
        alpha_fixed = alpha.squeeze().detach().numpy()
        A_0 = A0.numpy()
        A_1 = A1.numpy()
        # Frank Wolfe methods (projection-free methods)
        # try different random starts
        # fix the data and randomly initialize the alpha and see what happens.
        # Add Momentum

        objective = cp.sum_squares(w - alpha_fixed)
        restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1]
        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()
        
        
        alpha.data = torch.tensor(w.value).float()

        weights_y0 = (weights_features[:weights_features.shape[0]//2]@alpha).reshape(*data_count_0.shape)
        weights_y1 = (weights_features[weights_features.shape[0]//2:]@alpha).reshape(*data_count_1.shape)
        
        weighted_counts_1 = weights_y1*data_count_1
        weighted_counts_0 = weights_y0*data_count_0
        
        sex=1
    
        # ATE = (propensity_scores * (1/weighted_counts_1[sex])).mean()
        ATE = compute_conditional_ATE(weighted_counts_1, weighted_counts_0)
        
        loss = -ATE if upper_bound else ATE
        loss_values.append(ATE.detach().numpy())
        if iteration % 500 == 0:
            print(f"ATE: {ATE.item()}, ground_truth {gt_ate}")
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

    if upper_bound:
        ret = max(loss_values)
    else:
        ret = min(loss_values)
    return ret, loss_values, alpha

def empirical_propensity_score(X):
    # it s wrong
    # Code generated by Chat GPT
    # find unique values of each feature
    unique_vals = [np.unique(X[:, i]) for i in range(X.shape[1])]
    
    # calculate the frequency of occurrence of each combination of feature values
    counts = np.zeros([len(unique_vals[0]), len(unique_vals[1])])
    for i in range(X.shape[0]):
        counts[X[i,0], X[i,1]] += 1

    # normalize the counts to obtain empirical probabilities
    probs = counts / np.sum(counts)

    # assign each probability to its corresponding row in the original array
    empirical_probs = np.zeros([X.shape[0]])
    for i in range(X.shape[0]):
        empirical_probs[i] = probs[X[i,0], X[i,1]]
    return empirical_probs


if __name__ == '__main__':

    X_raw, A_raw, y_raw, obs = simulate_multiple_outcomes(DATASET_SIZE)
    X, A, y, = X_raw[obs], A_raw[obs], y_raw[obs]

    skewed_data = create_dataframe(X, A)
    data = create_dataframe(X_raw, A_raw)

    levels = [["female", "male"],  ["little", "moderate", "quite rich"], ["White", "Non White"]]
    
    skewed_data["Creditability"] = y
    data["Creditability"] = y_raw
    raw_counts = build_counts(data, levels, "Creditability")
    counts = build_counts(skewed_data, levels, "Creditability")
    obs_prob = counts/raw_counts

    gt_ate = compute_conditional_ATE(raw_counts[1], raw_counts[0])
    empirical_ate = compute_conditional_ATE(counts[1], counts[0])
    
    weights_features = torch.zeros(counts.numel(), 12)
    idx = 0
    
    for target in [0, 1]:
        index_aux = 0
        for m, se in enumerate(["female", "male"]):
            for j, income in enumerate(["little", "moderate", "quite rich"]):
                for i, race in enumerate(["White", "Non White"]):
                    features = [0]*(2*3*2)
                    features[index_aux] = 1
                    weights_features[idx] = torch.tensor(features).float()
                    # weights_features[idx] = torch.tensor(features + features_race).float()
                    idx += 1 
                    index_aux += 1
    
    # Should I playa round with different restrictions?
    y00_female = sum((data["Creditability"] == 0) & (data["female"] == 1))
    y01_female = sum((data["Creditability"] == 1) & (data["female"] == 1))

    y00_male = sum((data["Creditability"] == 0) & (data["male"] == 1))
    y01_male = sum((data["Creditability"] == 1) & (data["male"] == 1))

    bias_y00_female = sum((skewed_data["Creditability"] == 0) & (skewed_data["female"] == 1))
    bias_y01_female = sum((skewed_data["Creditability"] == 1) & (skewed_data["female"] == 1))

    bias_y00_male = sum((skewed_data["Creditability"] == 0) & (skewed_data["male"] == 1))
    bias_y01_male = sum((skewed_data["Creditability"] == 1) & (skewed_data["male"] == 1))

    
    b0 = np.array([y00_female, y00_male])
    b1 = np.array([y01_female, y01_male])

    obs_b0 = np.array([bias_y00_female, bias_y00_male])
    obs_b1 = np.array([bias_y01_female, bias_y01_male])
    

    weights_0 = obs_b0/b0
    weights_1 = obs_b1/b1


    data_count_0 = counts[0]
    data_count_1 = counts[1]

    

    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["female", "male"])
    # Changes ORIGINAL ONE X - X_2
    # pi_A = scipy.special.expit(X_2 - X)
    # A = 1*(pi_A > np.random.uniform(size=dataset_size))

    # mu = scipy.special.expit(2*A - X + X_2)
    # y = 1*(mu > np.random.uniform(size=dataset_size))

    # # Changes ORIGINAL ONE X + X_2
    # obs = scipy.special.expit(X - X_2) > np.random.uniform(size=dataset_size)

    gt_propensity_weighting_A = scipy.special.expit(X_raw[:,1] - X_raw[:,0]) 
    gt_propensity_weighting_R =  scipy.special.expit(X_raw[:, 0] - X_raw[:, 1])
    population_propensity_weighting_A = scipy.special.expit(X[:,1] - X[:,0])
    observed_weights = propensity_score_matching(weights_0, weights_1, skewed_data)
    empirical_probs = empirical_propensity_score(X)
    # I dont have the true Data size (n)
    ipw = compute_debias_ate_ipw(A_raw, gt_propensity_weighting_A, gt_propensity_weighting_R, y_raw, obs)
    biased_ipw = compute_ate_ipw(A, population_propensity_weighting_A, y)
    real_outcome = (scipy.special.expit(2 - X_raw[:,0] + X_raw[:,1])> np.random.uniform(size=DATASET_SIZE)).mean()
    # empirical_biased_ipw = compute_ate_ipw(A, empirical_probs, y)
    # NAIVE_IPW = compute_ate_ipw(A, observed_weights, y)
    prop_score_tensor = prpensity_scores_by_strata([["little", "moderate", "quite rich"],["White", "Non White"]])
    for i in range(1):
        upper_bound = True
        max_bound, max_loss_values, alpha_max = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, prop_score_tensor)

        upper_bound = False
        min_bound, min_loss_values, alpha_min = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, prop_score_tensor)

        c_time = datetime.datetime.now()
        timestamp = str(c_time.timestamp())
        timestamp = "_".join(timestamp.split("."))


        print(f"min:{float(min_bound)} , gt:{gt_ate},  max:{float(max_bound)}")
        plt.plot(min_loss_values)
        plt.plot(max_loss_values)
        plt.legend(["min", "max"])
        plt.axhline(y=gt_ate, color='r', linestyle='-')
        plt.axhline(y=empirical_ate, color='black', linestyle='-')
        # plt.axhline(y=ipw, color='g', linestyle='dashed')
        # plt.axhline(y=biased_ipw, color='cyan', linestyle='dashed')
        # plt.axhline(y=real_outcome, color='olive', linestyle='dashed')
    plt.savefig(f"losses_{timestamp}")