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

DATASET_SIZE = 47777

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
    return empirical_probs, torch.tensor(probs)

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
    data[["female", "male"]] = pd.get_dummies(group)
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

   
def run_search(A_0, A_1,data_count_1, data_count_0,
                     weights_features, upper_bound, gt_ate, propensity_scores):
    """Runs the search for the optimal weights."""

    prop_scores_0 = 1/(1-propensity_scores)
    prop_scores_1 = 1/propensity_scores
    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True)
    optim = torch.optim.Adam([alpha], 0.01)
    loss_values = []
    n_sample = data_count_1.sum() + data_count_0.sum()
    for iteration in range(2000):
        w = cp.Variable(weights_features.shape[1])
        alpha_fixed = alpha.squeeze().detach().numpy()
        A_0 = A0.numpy()
        A_1 = A1.numpy()

        objective = cp.sum_squares(w - alpha_fixed)
        restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= (n_sample/DATASET_SIZE)]
        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()
        
        alpha.data = torch.tensor(w.value).float()
        weights_y1 = (weights_features@alpha).reshape(*data_count_1.shape)
        weighted_counts_1 = weights_y1*data_count_1
        
        w_counts_1 = weighted_counts_1[1] 
        w_counts_0 =  weighted_counts_1[0]
       
        ht_A1 = (prop_scores_1*w_counts_1).sum()
        ht_A0 = (prop_scores_0*w_counts_0).sum()
        # N is known as it can be looked up from where the b were gathered.
        ate = (ht_A1 - ht_A0)/n_sample
        
        loss = -ate if upper_bound else ate
        loss_values.append(ate.detach().numpy())
        if iteration % 500 == 0:
            print(f"ATE: {ate.item()}, ground_truth {gt_ate}")
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    if upper_bound:
        ret = max(loss_values)
    else:
        ret = min(loss_values)
    return ret, loss_values, alpha

if __name__ == '__main__':

    data_source = ACSDataSource(survey_year='2018', 
                                horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    X, label, group = ACSEmployment.df_to_numpy(acs_data)
    group = 1*(group == 1)

    X = X.astype(int)
    label = label.astype(int)
    # last feature is the group
    print(X.shape)
    # X = X[:,12:-1]
    X = X[:,-8: -1]
    print(X.shape)
    sex = X[:, -2]
   
    dataset_size = X.shape[0]
    obs = scipy.special.expit(X[:,0] - X[:,1] + X[:,2]) > np.random.uniform(size=dataset_size)
    
    # Generates the data.
    X_sample, group_sample, y = X[obs], group[obs], label[obs]
    
    data, levels = build_dataset(X, group) 
    skewed_data, levels = build_dataset(X_sample, group_sample)
    print(levels)
    data["Creditability"] = label
    skewed_data["Creditability"] = y
    
    levels = [["female", "male"]] + levels
    print(levels)
    counts = build_counts(skewed_data, levels, "Creditability")

    number_strata = 1
    for level in levels:
        number_strata *= len(level)

    weights_features = torch.eye(number_strata)

    # # Creates groundtruth values to generate linear restrictions.
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

    data_count_0 = counts[0]
    data_count_1 = counts[1]
    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["female", "male"])
    

    # TODO (Santiago): Encapsulate benchmark generation process.
    # gt_propensity_weighting_A = scipy.special.expit(X_raw[:,1] - X_raw[:,0]) 
    # gt_propensity_weighting_R =  scipy.special.expit(X_raw[:, 0] - X_raw[:, 1])
    # population_propensity_weighting_A = scipy.special.expit(X[:,1] - X[:,0])
    # Which one make sense to use in this scenario?
    propensity_scores, prop_score_tensor = empirical_propensity_score(skewed_data, levels)
    real_propensity_scores, real_prop_score_tensor = empirical_propensity_score(data, levels)
    # gt_ate = compute_debias_ate_ipw()
    biased_ipw = compute_ate_ipw(group_sample, propensity_scores, y)
    print("biased_ipw", biased_ipw)
    ipw = compute_ate_ipw(group, real_propensity_scores, label)
    print("ipw", ipw)
    gt_ate = ipw

    # for i in range(1):
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
    plt.axhline(y=gt_ate, color='g', linestyle='dashed')
    plt.axhline(y=biased_ipw, color='cyan', linestyle='dashed')
    # plt.axhline(y=empirical_biased_ipw, color='olive', linestyle='dashed')
    plt.legend(["min", "max",  "IPW", "Empirical IPW"])
    plt.title("Average treatment effect.")# plt.title("Learning Curves for 10 trials.")
    plt.savefig(f"losses_{timestamp}")