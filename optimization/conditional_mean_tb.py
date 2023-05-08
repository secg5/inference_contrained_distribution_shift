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
from framework import compute_ate_conditional_mean, build_counts, build_dataset, \
                      get_folks_tables_data


def build_strata_counts_matrix(weight_features: torch.Tensor, 
                               counts: torch.Tensor, level: int, level_size:int):
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
    
    y_0_ground_truth = torch.zeros(level_size, features)
    y_1_ground_truth = torch.zeros(level_size, features)
    
    data_count_0 =  torch.swapaxes(counts[0], 0, level)
    data_count_1 =  torch.swapaxes(counts[1], 0, level)
    # import pdb; pdb.set_trace()
    # assert np.equal(data_count_0, counts[0]).sum() == np.prod(data_count_0.shape)
    # assert np.equal(data_count_1, counts[1]).sum() == np.prod(data_count_1.shape)
   
    for level in range(level_size):
        t = data_count_0[level].flatten().unsqueeze(1)
        features = weight_features[level*t.shape[0]:(level + 1)*t.shape[0]]        
        y_0_ground_truth[level] = (features*t).sum(dim=0)

        
    for level in range(level_size):
        t_ = data_count_1[level].flatten().unsqueeze(1)    
        features_ = weight_features[level*t.shape[0]:(level + 1)*t.shape[0]]        
        y_1_ground_truth[level] = (features_*t_).sum(dim=0)
    
    return y_0_ground_truth, y_1_ground_truth

def run_search(A_0, A_1, A_2, A_3, data_count_1, data_count_0,
                     weights_features, upper_bound, gt_ate, dataset_size):
    """Runs the search for the optimal weights."""
    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True)
    optim = torch.optim.Adam([alpha], 0.01)
    loss_values = []
    n_sample = data_count_1.sum() + data_count_0.sum()
    print(n_sample/dataset_size)
    A_0 = A0.numpy()
    A_1 = A1.numpy()
    A_2 = A2.numpy()
    A_3 = A3.numpy()
    A_4 = A4.numpy()
    A_5 = A5.numpy()
    A_6 = A6.numpy()
    A_7 = A7.numpy()
    for iteration in range(2000):
        w = cp.Variable(weights_features.shape[1])
        alpha_fixed = alpha.squeeze().detach().numpy()
       

        objective = cp.sum_squares(w - alpha_fixed)
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1]
        # P(r|A=1) is bounded below by n/N
        restrictions = [#A_0@ w == b0, A_1@ w == b1, 
                        A_7@ w == b7,
                        weights_features@w >= n_sample/dataset_size]
        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve(solver=cp.ECOS)
        # import pdb;pdb.set_trace()
        alpha.data = torch.tensor(w.value).float()
        weights_y1 = (weights_features@alpha).reshape(*data_count_1.shape)
        # weighted_counts_1 = weights_y1*data_count_1*pr_r_t
        weighted_counts_1 = weights_y1*data_count_1
        weights_y0 = (weights_features@alpha).reshape(*data_count_0.shape)
        # weighted_counts_0 = weights_y0*data_count_0*pr_r_t
        weighted_counts_0 = weights_y0*data_count_0

        w_counts_1 = weighted_counts_1.select(-1, 1)
        w_counts_0 = weighted_counts_0.select(-1, 1)
        # first axis is race
        # ht_A1 = w_counts_1.sum()
        # ht_A0 = (prop_scores_0*w_counts_0).sum()
        # N is known as it can be looked up from where the b were gathered.
        # ate = ht_A1/DATASET_SIZE
        size = w_counts_1.sum() + w_counts_0.sum()
        ate = w_counts_1.sum()/size
        
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
    # import pdb; pdb.set_trace()
    np.save("alpha", alpha.detach().numpy())
    return ret, loss_values, alpha

# ACSEmployment = folktables.BasicProblem(
#     features=[
#         'AGEP',
#         'SCHL',
#         'MAR',
#         'RELP',
#         'DIS',
#         'ESP',
#         'CIT',
#         'MIG',
#         'MIL',
#         'ANC',
#         'NATIVITY',
#         'DEAR',
#         'DEYE',
#         'DREM',
#         'SEX',
#         'RAC1P',
#     ],

# def build_restrictions():
#     """_summary_
#     """
    # y00_female = sum((data["Creditability"] == 0) & (data["white"] == 1))
    # y01_female = sum((data["Creditability"] == 1) & (data["white"] == 1))
    # y00_male = sum((data["Creditability"] == 0) & (data["non-white"] == 1))
    # y01_male = sum((data["Creditability"] == 1) & (data["non-white"] == 1))

    # bias_y00_female = sum((skewed_data["Creditability"] == 0) & (skewed_data["white"] == 1))
    # bias_y01_female = sum((skewed_data["Creditability"] == 1) & (skewed_data["non-white"] == 1))
    # bias_y00_male = sum((skewed_data["Creditability"] == 0) & (skewed_data["white"] == 1))
    # bias_y01_male = sum((skewed_data["Creditability"] == 1) & (skewed_data["non-white"] == 1))

if __name__ == '__main__':

    X, label, sex, group = get_folks_tables_data()
   
   
    dataset_size = X.shape[0]
    obs = scipy.special.expit(X[:,0] - X[:,1] + X[:,2]) > np.random.uniform(size=dataset_size)
    
    # Generates the data.
    X_sample, group_sample, y = X[obs], group[obs], label[obs]
    sex_group = sex[obs]
    data, levels = build_dataset(X, group) 
    skewed_data, levels = build_dataset(X_sample, group_sample)
    print(levels)
    data["Creditability"] = label
    skewed_data["Creditability"] = y
    
    levels = [["white", "non-white"]] + levels
    print(levels)
    counts = build_counts(skewed_data, levels, "Creditability")

    number_strata = 1
    for level in levels:
        number_strata *= len(level)
    print(number_strata)
    weights_features = torch.eye(number_strata)
    # # Creates groundtruth values to generate linear restrictions.

    y00_female = sum((data["Creditability"] == 0) & (data["white"] == 1))
    y01_female = sum((data["Creditability"] == 1) & (data["white"] == 1))
    y00_male = sum((data["Creditability"] == 0) & (data["non-white"] == 1))
    y01_male = sum((data["Creditability"] == 1) & (data["non-white"] == 1))

    bias_y00_female = sum((skewed_data["Creditability"] == 0) & (skewed_data["white"] == 1))
    bias_y01_female = sum((skewed_data["Creditability"] == 1) & (skewed_data["non-white"] == 1))
    bias_y00_male = sum((skewed_data["Creditability"] == 0) & (skewed_data["white"] == 1))
    bias_y01_male = sum((skewed_data["Creditability"] == 1) & (skewed_data["non-white"] == 1))
    # ['2_0', '2_1']
    y00_2_0 = sum((data["Creditability"] == 0) & (data["2_0"] == 1))
    y01_2_0 = sum((data["Creditability"] == 1) & (data["2_0"] == 1))
    y00_2_1 = sum((data["Creditability"] == 0) & (data["2_1"] == 1))
    y01_2_1 = sum((data["Creditability"] == 1) & (data["2_1"] == 1))

    bias_y00_2_0 = sum((skewed_data["Creditability"] == 0) & (skewed_data["2_0"] == 1))
    bias_y01_2_0 = sum((skewed_data["Creditability"] == 1) & (skewed_data["2_0"] == 1))
    bias_y00_2_1 = sum((skewed_data["Creditability"] == 0) & (skewed_data["2_1"] == 1))
    bias_y01_2_1 = sum((skewed_data["Creditability"] == 1) & (skewed_data["2_1"] == 1))

    # ['1_0', '1_1', '1_2', '1_3']
    y00_1_0 = sum((data["Creditability"] == 0) & (data["1_0"] == 1))
    y01_1_0 = sum((data["Creditability"] == 1) & (data["1_0"] == 1))
    y00_1_1 = sum((data["Creditability"] == 0) & (data["1_1"] == 1))
    y01_1_1 = sum((data["Creditability"] == 1) & (data["1_1"] == 1))
    y00_1_2 = sum((data["Creditability"] == 0) & (data["1_2"] == 1))
    y01_1_2 = sum((data["Creditability"] == 1) & (data["1_2"] == 1))
    y00_1_3 = sum((data["Creditability"] == 0) & (data["1_3"] == 1))
    y01_1_3 = sum((data["Creditability"] == 1) & (data["1_3"] == 1))

    bias_y00_1_0 = sum((skewed_data["Creditability"] == 0) & (skewed_data["1_0"] == 1))
    bias_y01_1_0 = sum((skewed_data["Creditability"] == 1) & (skewed_data["1_0"] == 1))
    bias_y00_1_1 = sum((skewed_data["Creditability"] == 0) & (skewed_data["1_1"] == 1))
    bias_y01_1_1 = sum((skewed_data["Creditability"] == 1) & (skewed_data["1_1"] == 1))
    bias_y00_1_2 = sum((skewed_data["Creditability"] == 0) & (skewed_data["1_2"] == 1))
    bias_y01_1_2 = sum((skewed_data["Creditability"] == 1) & (skewed_data["1_2"] == 1))
    bias_y00_1_3 = sum((skewed_data["Creditability"] == 0) & (skewed_data["1_3"] == 1))
    bias_y01_1_3 = sum((skewed_data["Creditability"] == 1) & (skewed_data["1_3"] == 1))

    y00_0_0 = sum((data["Creditability"] == 0) & (data["0_0"] == 1))
    y01_0_0 = sum((data["Creditability"] == 1) & (data["0_0"] == 1))
    y00_0_1 = sum((data["Creditability"] == 0) & (data["0_1"] == 1))
    y01_0_1 = sum((data["Creditability"] == 1) & (data["0_1"] == 1))
    y00_0_2 = sum((data["Creditability"] == 0) & (data["0_2"] == 1))
    y01_0_2 = sum((data["Creditability"] == 1) & (data["0_2"] == 1))
    y00_0_3 = sum((data["Creditability"] == 0) & (data["0_3"] == 1))
    y01_0_3 = sum((data["Creditability"] == 1) & (data["0_3"] == 1))
    y00_0_4 = sum((data["Creditability"] == 0) & (data["0_4"] == 1))
    y01_0_4 = sum((data["Creditability"] == 1) & (data["0_4"] == 1))

    bias_y00_0_0 = sum((skewed_data["Creditability"] == 0) & (skewed_data["0_0"] == 1))
    bias_y01_0_0 = sum((skewed_data["Creditability"] == 1) & (skewed_data["0_0"] == 1))
    bias_y00_0_1 = sum((skewed_data["Creditability"] == 0) & (skewed_data["0_1"] == 1))
    bias_y01_0_1 = sum((skewed_data["Creditability"] == 1) & (skewed_data["0_1"] == 1))
    bias_y00_0_2 = sum((skewed_data["Creditability"] == 0) & (skewed_data["0_2"] == 1))
    bias_y01_0_2 = sum((skewed_data["Creditability"] == 1) & (skewed_data["0_2"] == 1))
    bias_y00_0_3 = sum((skewed_data["Creditability"] == 0) & (skewed_data["0_3"] == 1))
    bias_y01_0_3 = sum((skewed_data["Creditability"] == 1) & (skewed_data["0_3"] == 1))
    bias_y00_0_4 = sum((skewed_data["Creditability"] == 0) & (skewed_data["0_4"] == 1))
    bias_y01_0_4 = sum((skewed_data["Creditability"] == 1) & (skewed_data["0_4"] == 1))
    
    b0 = np.array([y00_female, y00_male])
    b1 = np.array([y01_female, y01_male])
    b2 = np.array([y00_2_0, y00_2_1])
    b3 = np.array([y01_2_0, y01_2_1])
    b4 = np.array([y00_1_0, y00_1_1, y00_1_2, y00_1_3])
    b5 = np.array([y01_1_0, y01_1_1, y01_1_2, y01_1_3])
    b6 = np.array([y00_0_0, y00_0_1, y00_0_2, y00_0_3, y00_0_4])
    b7 = np.array([y01_0_0, y01_0_1, y01_0_2, y01_0_3, y01_0_4])

    obs_b0 = np.array([bias_y00_female, bias_y00_male])
    obs_b1 = np.array([bias_y01_female, bias_y01_male])
    obs_b2 = np.array([bias_y00_2_0, bias_y00_2_1])
    obs_b3 = np.array([bias_y01_2_0, bias_y01_2_1])
    obs_b4 = np.array([bias_y00_1_0, bias_y00_1_1, bias_y00_1_2, bias_y00_1_3])
    obs_b5 = np.array([bias_y01_1_0, bias_y01_1_1, bias_y01_1_2, bias_y01_1_3])
    obs_b6 = np.array([bias_y00_0_0, bias_y00_0_1, bias_y00_0_2, bias_y00_0_3, bias_y00_0_4])
    obs_b7 = np.array([bias_y01_0_0, bias_y01_0_1, bias_y01_0_2, bias_y01_0_3, bias_y01_0_4])


    data_count_0 = counts[0]
    data_count_1 = counts[1]
    A0, A1 = build_strata_counts_matrix(weights_features, counts, 0, 2)
    A2, A3 = build_strata_counts_matrix(weights_features, counts, 3, 2)
    A4, A5 = build_strata_counts_matrix(weights_features, counts, 2, 4)
    A6, A7 = build_strata_counts_matrix(weights_features, counts, 1, 5)
    

    # TODO (Santiago): Encapsulate benchmark generation process.
    # gt_propensity_weighting_A = scipy.special.expit(X_raw[:,1] - X_raw[:,0]) 
    # gt_propensity_weighting_R =  scipy.special.expit(X_raw[:, 0] - X_raw[:, 1])
    # population_propensity_weighting_A = scipy.special.expit(X[:,1] - X[:,0])
    # Which one make sense to use in this scenario?
    # Higest weight alpha

    mean_race = compute_ate_conditional_mean(1 -group_sample, y)
    mean_race_t = data_count_1[1].sum()/(data_count_1[1].sum() + data_count_0[1].sum()) 
    print("test", mean_race, mean_race_t)
    
    # Observed
    biased_empirical_mean = compute_ate_conditional_mean(1 - sex_group, y)
    mean_race_t = data_count_1.select(-1,1).sum()/(data_count_1.select(-1,1).sum() + data_count_0.select(-1,1).sum()) 
    print("test_2", biased_empirical_mean, mean_race_t)
    # Real
    empirical_mean = compute_ate_conditional_mean(1 - sex, label)
    print(empirical_mean, biased_empirical_mean)

    np.save("baselines", np.array([biased_empirical_mean, empirical_mean]))
    gt_ate = empirical_mean

    for index in range(1):
        upper_bound = True
        max_bound, max_loss_values, alpha_max = run_search(A0, A1, A2, A3, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, dataset_size)

        upper_bound = False
        min_bound, min_loss_values, alpha_min = run_search(A0, A1, A2, A3, data_count_1, data_count_0, weights_features, upper_bound, gt_ate,  dataset_size)
        
        c_time = datetime.datetime.now()
        timestamp = str(c_time.timestamp())
        timestamp = "_".join(timestamp.split("."))
        np.save(f"min_loss_{index}", min_loss_values)
        np.save(f"max_loss_{index}", max_loss_values)

        print(f"min:{float(min_bound)} , gt:{gt_ate},  max:{float(max_bound)}")
        plt.plot(min_loss_values)
        plt.plot(max_loss_values)
        # plt.axhline(y=gt_ate, color='g', linestyle='dashed')
        plt.axhline(y=empirical_mean, color='cyan', linestyle='dashed')
        plt.axhline(y=biased_empirical_mean, color='olive', linestyle='dashed')
        plt.legend(["min", "max",  "True condintional mean", "Empirical conditional mean"])
        plt.title("Conditional mean.")# plt.title("Learning Curves for 10 trials.")
        plt.savefig(f"losses_{timestamp}")

