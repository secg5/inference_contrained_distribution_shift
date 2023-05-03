import scipy
import datetime

import torch
import numpy as np
import cvxpy as cp
import itertools

import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from framework import compute_ate_conditional_mean, build_counts, build_dataset, \
                      build_strata_counts_matrix, get_folks_tables_data



   
def run_search(A_0, A_1,data_count_1, data_count_0,
                     weights_features, upper_bound, gt_ate, dataset_size):
    """Runs the search for the optimal weights."""

    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True)
    optim = torch.optim.Adam([alpha], 0.01)
    loss_values = []
    n_sample = data_count_1.sum() + data_count_0.sum()
    print(n_sample/dataset_size)
    size_t = data_count_1.select(-1, 1).sum() + data_count_0.select(-1, 1).sum()
    mean =  data_count_1.select(-1, 1).sum()/size_t
    print("mean", mean)
    for iteration in range(5000):
        w = cp.Variable(weights_features.shape[1])
        alpha_fixed = alpha.squeeze().detach().numpy()
        A_0 = A0.numpy()
        A_1 = A1.numpy()

        objective = cp.sum_squares(w - alpha_fixed)
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1]
        # P(r|A=1) is bounded below by n/N
        restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= n_sample/dataset_size]
        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()
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

    return ret, loss_values, alpha
 
def traverse_combinations(list_of_lists):
    for combination in itertools.product(*list_of_lists):
        yield combination

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
    print(skewed_data.shape)
    number_strata = 1
    
    for level in levels:
        number_strata *= len(level)
    
    number_strata = 1
    for level in levels:
        number_strata *= len(level)
    print(number_strata)

    idx = 0
    idj = 0
    weights_features = torch.zeros(number_strata, 2*5*4*2)
    starting_tuple = ("white", '0_0', '1_0', '2_0')
    previous_tuple = starting_tuple
    flag = True

    for combination in traverse_combinations(levels):
        current_tuple = (combination[0], combination[1], combination[2], combination[3])
        if previous_tuple != current_tuple:
            if current_tuple == starting_tuple:
                idj = 0
            else:
                idj += 1
        weight = [0]*(2*5*4*2)
        weight[idj] = 1
        weights_features[idx] = torch.tensor(weight).float()
        idx += 1
       
        # if idx % 24 == 1:
        #     print(current_tuple, previous_tuple)
        #     print(idx, number_strata)
        #     print(weights_features.sum(axis=0))
        previous_tuple = current_tuple
    print(weights_features.sum(axis=0))

    print(number_strata)
    # weights_features = torch.eye(number_strata)
    # Creates groundtruth values to generate linear restrictions.
    y00_female = sum((data["Creditability"] == 0) & (data["white"] == 1))
    y01_female = sum((data["Creditability"] == 1) & (data["white"] == 1))

    y00_male = sum((data["Creditability"] == 0) & (data["non-white"] == 1))
    y01_male = sum((data["Creditability"] == 1) & (data["non-white"] == 1))

    bias_y00_female = sum((skewed_data["Creditability"] == 0) & (skewed_data["white"] == 1))
    bias_y01_female = sum((skewed_data["Creditability"] == 1) & (skewed_data["non-white"] == 1))

    bias_y00_male = sum((skewed_data["Creditability"] == 0) & (skewed_data["white"] == 1))
    bias_y01_male = sum((skewed_data["Creditability"] == 1) & (skewed_data["non-white"] == 1))

    b0 = np.array([y00_female, y00_male])
    b1 = np.array([y01_female, y01_male])
    obs_b0 = np.array([bias_y00_female, bias_y00_male])
    obs_b1 = np.array([bias_y01_female, bias_y01_male])

    data_count_0 = counts[0]
    data_count_1 = counts[1]
    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["white", "non-white"])
    
    # gt_ate = compute_debias_ate_ipw()
    # np.save("baselines", np.array([biased_empirical_mean, empirical_mean, ipw]))
    # print("baselines:", [biased_empirical_mean, empirical_mean, ipw])
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

    gt_ate = empirical_mean

    for index in range(10):
        upper_bound = True
        max_bound, max_loss_values, alpha_max = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, dataset_size)

        upper_bound = False
        min_bound, min_loss_values, alpha_min = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, dataset_size)

        c_time = datetime.datetime.now()
        timestamp = str(c_time.timestamp())
        timestamp = "_".join(timestamp.split("."))
        np.save(f"numerical_results/min_loss_4_{index}", min_loss_values)
        np.save(f"numerical_results/max_loss_4_{index}", max_loss_values)

        print(f"min:{float(min_bound)} , gt:{gt_ate},  max:{float(max_bound)}")
        plt.plot(min_loss_values)
        plt.plot(max_loss_values)
        # plt.axhline(y=gt_ate, color='g', linestyle='dashed')
        plt.axhline(y=empirical_mean, color='cyan', linestyle='dashed')
        plt.axhline(y=biased_empirical_mean, color='olive', linestyle='dashed')
        plt.legend(["min", "max",  "True condintional mean", "Empirical conditional mean"])
        plt.title("Conditional mean.")# plt.title("Learning Curves for 10 trials.")
        plt.savefig(f"losses_{timestamp}")

