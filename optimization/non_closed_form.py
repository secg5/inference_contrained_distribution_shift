import scipy
import datetime

import torch
import numpy as np
import cvxpy as cp
import itertools
import time

import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model import WeightedLogisticRegression
from torch.linalg import inv, lstsq
from framework import compute_ate_conditional_mean, build_counts, build_dataset, \
                      build_strata_counts_matrix, get_folks_tables_data

def build_dataset(X: np.ndarray, group: np.ndarray):
    data = pd.DataFrame()
    levels = []
    for column in range(X.shape[1]):
        features = pd.get_dummies(X[:,column])
        strata_number = features.shape[1]
        names = [str(column) + "_" + str(j) for j in range(int(strata_number))]
        data[names[:-1]] = 0 
        data[names[:-1]] = features[features.columns[:-1]]
        levels.append(names[:-1])
    data["white"] = pd.get_dummies(group)[0]
    return data, levels

def build_dataset_full(X: np.ndarray, group: np.ndarray):
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

def create_features_tensor(data, label):
    features = data.iloc[:,:-1]
    features = features.to_numpy().astype(np.double)
    features = np.concatenate([features, np.ones((features.shape[0],1))], axis =-1)
    features_tensor = torch.tensor(features).float()
    target = torch.tensor(label, dtype = torch.long).float()
    target = torch.unsqueeze(target, -1)
    return features_tensor, target

def assign_weights(data, hash_map, weights_features):
    """Code that from the weihgts matrix assigns the corresponding weight to 
    each feature according to the correct combination of feature strata.

    Args:
        features (_type_): _description_
        weights (_type_): _description_
    """
    skewed_data.iloc[0,:].index
    weigths = []
    data_features = data[data.columns[:-1]]
    for i  in range(data_features.shape[0]):
        indexes = data_features.iloc[i,:] == 1
        columns_names = data_features.iloc[i,:][indexes].index
        # tuple_features = (columns_names[-1], columns_names[0], columns_names[1], columns_names[2])
        tuple_features = (columns_names[0], columns_names[1])
        # print(tuple_features)
        # import pdb; pdb.set_trace()
        # tuple_features = (columns_names[-1], columns_names[0], columns_names[1], columns_names[2], columns_names[3], columns_names[4], columns_names[5], columns_names[6])
        weight_index = hash_map[tuple_features]
        weight = weights_features[weight_index]
        weigths.append(weight)
    # import pdb; pdb.set_trace()
    return torch.stack(weigths)

def run_search(A_0, A_1,data_count_1, data_count_0,
                     weights_features, upper_bound, gt_ate, dataset_size, data, labels, weights_array, g_cpu):
    """Runs the search for the optimal weights."""
   

    
    torch.autograd.set_detect_anomaly(True)
    g_cpu.manual_seed(11)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True, generator=g_cpu)
    print(alpha)
    
    features_tensor, target = create_features_tensor(skewed_data, labels)
    loss_values = []
    # import pdb; pdb.set_trace()
    n_sample = data_count_1.sum() + data_count_0.sum()
    
    optim = torch.optim.Adam([alpha], 0.9)
    print(features_tensor.shape)
    for iteration in range(150):
        w = cp.Variable(weights_features.shape[1])
        alpha_fixed = alpha.squeeze().detach().numpy()
        A_0 = A0.numpy()
        A_1 = A1.numpy()

        objective = cp.sum_squares(w - alpha_fixed)
    
        restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= n_sample/dataset_size]
        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()
        
        alpha.data = torch.tensor(w.value).float()
        
        weights = weights_array @ alpha
        # weights.data = torch.clamp(weights, min=0.001)
        # import pdb; pdb.set_trace()

        sq_weights = torch.sqrt(weights)
        W = torch.eye(weights.shape[0])*sq_weights
        coeff = lstsq(W@features_tensor, W@target,  driver = 'gelsd')
        
        loss_1 =  coeff[0][0]
        if iteration == 0:
            print(alpha)
            print(loss_1.detach().numpy())
        beta = -loss_1 if upper_bound else loss_1
        loss_values.append( loss_1.detach().numpy())
        if iteration % 10 == 0:
            print(f"ATE: {loss_1.detach().numpy()}, ground_truth {beta.detach().numpy()}")
        
        optim.zero_grad()
        beta.backward()
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
    size = 25000
    X, label, sex, group =  X[:size], label[:size], sex[:size], group[:size]
    
    print(label.shape)
    print(label.shape)


    dataset_size = X.shape[0]
    print("dataset_size", dataset_size)
    obs = scipy.special.expit(X[:,0] - X[:,1]) > np.random.uniform(size=dataset_size)
    prb_obs = (sum(obs)/dataset_size)/ scipy.special.expit(X[:,0] - X[:,1])[obs]
    # Generates the data.
    X_sample, group_sample, y = X[obs], group[obs], label[obs]
    sex_group = sex[obs]
    
   
    data, levels,  = build_dataset(X, group) 
    skewed_data, levels = build_dataset(X_sample, group_sample)

    data_full, levels_full  = build_dataset_full(X, group) 
    skewed_data_full, levels_full  = build_dataset_full(X_sample, group_sample) 

    data["Creditability"] = label
    data_full["Creditability"] = label

    skewed_data["Creditability"] = y
    skewed_data_full["Creditability"] = y
    
    levels = [["white"]] + levels
    levels_full = [["white", "non-white"]] + levels_full
    print(levels)
    print(levels_full)

    counts = build_counts(skewed_data_full, levels_full, "Creditability")
    print(skewed_data.shape)
    number_strata = 1
    start_time = time.time()
    
    ground_truth_model = LinearRegression().fit(data.iloc[:,:-1], label)
    baseline_model = LinearRegression().fit(skewed_data.iloc[:,:-1], y)
    print("ground_truth_model_sklearn", ground_truth_model.coef_[0])
    print("baseline_sklearn", baseline_model.coef_[0])
    baseline = baseline_model.coef_[0]
    ground_truth = ground_truth_model.coef_[0]

    print("=======Sanity check=======")
    features_tensor_gt, target_gt = create_features_tensor(data, label)
    ground_truth_model_torch = lstsq(features_tensor_gt, target_gt,  driver = 'gelsd')
    print("gt_torch", ground_truth_model_torch[0][0])
    print("=======Real weights=======")
    features_tensor_bsline, target_bsline = create_features_tensor(skewed_data, y)
    baseline_model_torch = lstsq(features_tensor_bsline, target_bsline,  driver = 'gelsd')
    print("baseline_torch", baseline_model_torch[0][0])

    sq_weights = torch.sqrt(torch.Tensor(prb_obs))
    W = torch.eye(features_tensor_bsline.shape[0])*sq_weights
    baseline_adjusted_torch  = lstsq(W@features_tensor_bsline, W@target_bsline,  driver = 'gelsd')
    print("baseline_torch_adjusted", baseline_adjusted_torch[0][0])

    
    for level in levels_full:
        number_strata *= len(level)
    
    weights_features = torch.eye(number_strata)
    idx = 0
    hash_map = {}

    for combination in traverse_combinations(levels_full):
        hash_map[combination] = idx
        idx += 1
    print("weights_features shpae", weights_features.shape)
    
    idx = 0
    idj = 0
    weights_features = torch.zeros(number_strata, 5*4)
    print(weights_features.shape)
    # starting_tuple = ("white", '0_0', '1_0', '2_0')
    starting_tuple = ('0_0', '1_0')
    previous_tuple = starting_tuple
    flag = True
    hash_map = {}

    for combination in traverse_combinations(levels_full):
        current_tuple = (combination[1], combination[2])
        if previous_tuple != current_tuple:
            if current_tuple == starting_tuple:
                idj = 0
            else:
                idj += 1
        weight = [0]*(5*4)
        weight[idj] = 1
        weights_features[idx] = torch.tensor(weight).float()

        hash_map[current_tuple] = idx
        idx += 1
        previous_tuple = current_tuple
    print(weights_features.sum(axis=0))

    
    weights_array = assign_weights(skewed_data_full, hash_map, weights_features)
    print(f"Shape number of weights {weights_array.shape}")

    
    # Creates groundtruth values to generate linear restrictions.
    y00_female = sum((data_full["Creditability"] == 0) & (data_full["white"] == 1))
    y01_female = sum((data_full["Creditability"] == 1) & (data_full["white"] == 1))

    y00_male = sum((data_full["Creditability"] == 0) & (data_full["non-white"] == 1))
    y01_male = sum((data_full["Creditability"] == 1) & (data_full["non-white"] == 1))

    b0 = np.array([y00_female, y00_male])
    b1 = np.array([y01_female, y01_male])


    data_count_0 = counts[0]
    data_count_1 = counts[1]

    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["white", "non-white"])

    
    NUM_FEATURES = len(skewed_data.columns[:-1])
    print(f"NUM_FEATURES : {NUM_FEATURES}")
    NUM_LABELS = 1

    for index in range(5):
        g_cpu = torch.Generator()
        upper_bound = True
        max_bound, max_loss_values, alpha_max = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, ground_truth, dataset_size, skewed_data, y, weights_array, g_cpu)
        print("=======////=======")
        upper_bound = False
        min_bound, min_loss_values, alpha_min = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, ground_truth, dataset_size, skewed_data, y, weights_array, g_cpu)

        c_time = datetime.datetime.now()
        timestamp = str(c_time.timestamp())
        timestamp = "_".join(timestamp.split("."))
        np.save(f"non_close_results/min_loss_tight_{index}", min_loss_values)
        np.save(f"non_close_results/max_loss_tight_{index}", max_loss_values)

        print(f"min:{float(min_bound)} , gt:{ground_truth},  max:{float(max_bound)}")
        plt.plot(min_loss_values)
        plt.plot(max_loss_values)
        plt.axhline(y=ground_truth, color='cyan', linestyle='dashed')
        plt.axhline(y=baseline, color='olive', linestyle='dashed')
        # plt.legend(["min", "max", "Ground truth", "baseline"])
        plt.title("Model output.")# plt.title("Learning Curves for 10 trials.")
        plt.savefig(f"losses_{timestamp}")

