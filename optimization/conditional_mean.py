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
    for iteration in range(3000):
        w = cp.Variable(weights_features.shape[1])
        alpha_fixed = alpha.squeeze().detach().numpy()
        A_0 = A0.numpy()
        A_1 = A1.numpy()

        objective = cp.sum_squares(w - alpha_fixed)
        restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= n_sample/dataset_size]
        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()

        alpha.data = torch.tensor(w.value).float()
        weights_y1 = (weights_features@alpha).reshape(*data_count_1.shape)

        weighted_counts_1 = weights_y1*data_count_1
        weights_y0 = (weights_features@alpha).reshape(*data_count_0.shape)

        weighted_counts_0 = weights_y0*data_count_0

        w_counts_1 = weighted_counts_1.select(-1, 1)
        w_counts_0 = weighted_counts_0.select(-1, 1)

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

    b0 = np.array([y00_female, y00_male])
    b1 = np.array([y01_female, y01_male])
    obs_b0 = np.array([bias_y00_female, bias_y00_male])
    obs_b1 = np.array([bias_y01_female, bias_y01_male])

    data_count_0 = counts[0]
    data_count_1 = counts[1]
    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["white", "non-white"])
    
    biased_empirical_mean = compute_ate_conditional_mean(1 - sex_group, y)
    empirical_mean = compute_ate_conditional_mean(1 - sex, label)

    np.save("baselines", np.array([biased_empirical_mean, empirical_mean]))
    print("baselines:", [biased_empirical_mean, empirical_mean])

    for index in range(10):
        upper_bound = True
        max_bound, max_loss_values, alpha_max = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, empirical_mean, dataset_size)

        upper_bound = False
        min_bound, min_loss_values, alpha_min = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, empirical_mean, dataset_size)

        c_time = datetime.datetime.now()
        timestamp = str(c_time.timestamp())
        timestamp = "_".join(timestamp.split("."))
        np.save(f"min_loss_all_{index}", min_loss_values)
        np.save(f"max_loss_all_{index}", max_loss_values)

        print(f"min:{float(min_bound)} , gt:{empirical_mean},  max:{float(max_bound)}")
        plt.plot(min_loss_values)
        plt.plot(max_loss_values)
        plt.axhline(y=empirical_mean, color='cyan', linestyle='dashed')
        plt.axhline(y=biased_empirical_mean, color='olive', linestyle='dashed')
        plt.legend(["min", "max",  "True condintional mean", "Empirical conditional mean"])
        plt.title("Conditional mean.")# plt.title("Learning Curves for 10 trials.")
        plt.savefig(f"losses_{timestamp}")

