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

    count[count == 0] = 1e-12
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
    X = np.random.multivariate_normal(mean=np.zeros(4), cov=np.identity(4), size = dataset_size)
   
    X = pd.DataFrame([pd.cut(X[:,0], [-np.inf, 0, np.inf]).codes,
                    pd.cut(X[:,1], [-np.inf, 1, np.inf]).codes,
                    pd.cut(X[:,2], [-np.inf, -1, np.inf]).codes,
                    pd.cut(X[:,3], [-np.inf, -1, 1, np.inf]).codes]).T
    
    pi_x = scipy.special.expit((2*X[0] - 4*X[1] + 2*X[2] - X[3])/4)
    A = 1*(pi_x > np.random.uniform(size=dataset_size))

    
    
    # mu_0 = scipy.special.expit(X[:,1] - X[:,2] + X[:,3] - 2*A)
    mu_0 = scipy.special.expit(X[1] - X[3] - 2*A)
    y_0 = 1*(mu_0 > np.random.uniform(size=dataset_size))
    # Being observed depends on sex and income
    obs = scipy.special.expit(X[3] - 3*A) > np.random.uniform(size=dataset_size)
    # Source of error? Should we being computing the ATE?
    yc_00 = np.mean(scipy.special.expit(X[1] - X[3]))
    yc_01 = np.mean(scipy.special.expit(X[1] - X[3] - 2))
    
    gt_ate = yc_00 - yc_01
    print(f"Groundtruth:{gt_ate}") 
    return X, A, y_0, obs, gt_ate
    
def create_dataframe(X, A):
    skewed_data = pd.DataFrame()
    skewed_data[["white", "non-white"]] = pd.get_dummies(X[0])
    skewed_data[["Age Bucket 1", "Age Bucket 2"]] = pd.get_dummies(X[1])
    skewed_data[["own", "rent"]] = pd.get_dummies(X[2])
    skewed_data[["little", "moderate", "quite rich"]] = pd.get_dummies(X[3])
    skewed_data[["female", "male"]] = pd.get_dummies(A)
    return skewed_data

def run_search(A_0, A_1,data_count_1, data_count_0, weights_features, upper_bound, gt_ate, obs_prob):
    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True)
    W = np.unique(weights_features.numpy(), axis=0)
    optim = torch.optim.Adam([alpha], 1e-4)
    # scheduler = StepLR(optim, step_size=300, gamma=0.9)
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    # scheduler = CosineAnnealingLR(optim,
    #                             T_max = 4000, # Maximum number of iterations.
    #                             eta_min = 1e-4)
    loss_values = []
    for iteration in range(10000):
        # # Issue with proyection!!!!!!!!
        # import pdb; pdb.set_trace()
        # print(alpha)
        w = cp.Variable(alpha.shape[0])
        alpha_fixed = alpha.squeeze().detach().numpy()
        A_0 = A0.numpy()
        A_1 = A1.numpy()
        objective = cp.sum_squares(w - alpha_fixed)
        # Fix a couple of strata restrictions and try to make it work.
        # import pdb; pdb.set_trace()
        # print(alpha)
        inv_prop = 1/(obs_prob.numpy().flatten())
        # parabola
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, inv_prop[8:24] == w[8:24], inv_prop[32:48] == w[32:48]]
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, inv_prop[:8] == w[:8], inv_prop[24:32] == w[24:32]]
        # [["female", "male"], ["white", "non-white"], ["Age Bucket 1", "Age Bucket 2"], ["little", "moderate", "quite rich"]]
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, inv_prop[8:11] == w[8:11], inv_prop[32:35] == w[32:35]]
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, inv_prop[:3] == w[:3]]
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, inv_prop[4:] == w[4:]]
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, w[:24] == w[24:]]#inv_prop[16:24] == w[16:24], inv_prop[40:48] == w[40:48]]
        restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, inv_prop[8:16] == w[8:16], inv_prop[32:40] == w[32:40]]
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1, inv_prop[1:24] == w[1:24], inv_prop[25:] == w[25:]]

        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()

        alpha.data = torch.tensor(w.value).float()
        # Can I see how to change the weights to keep increasing/ decreasing the function
        # Sources of issue: Optimizer 
        # Problem with the formulation
        # What happens with full freedom
        weights_y0 = (weights_features[:weights_features.shape[0]//2]@alpha).reshape(*data_count_0.shape)
        weights_y1 = (weights_features[weights_features.shape[0]//2:]@alpha).reshape(*data_count_1.shape)
        
        weighted_counts_1 = weights_y1*data_count_1
        weighted_counts_0 = weights_y0*data_count_0

        sex = 1
        sex_base = 0
        
        probs = weighted_counts_1/(weighted_counts_1 + weighted_counts_0)

        total_weight_count = weighted_counts_1[sex] + weighted_counts_0[sex]
        ATE = ((probs[sex_base] - probs[sex])*total_weight_count/total_weight_count.sum()).sum()
        
        if upper_bound:
            loss = -ATE
        else:
            loss = ATE
        loss_values.append(float(ATE.detach().numpy()))
        if iteration % 500 == 0:
            print(f"ATE: {ATE.item()}, ground_truth {gt_ate}")
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        # scheduler.step()

    if upper_bound:
        ret = max(loss_values)
    else:
        ret = min(loss_values)
    return ret, loss_values

if __name__ == '__main__':

    DATASET_SIZE = 100000 
    X_raw, A_raw, y_0_raw, obs, gt_ate = simulate_multiple_outcomes(DATASET_SIZE)
    X, A, y_0 = X_raw[obs], A_raw[obs], y_0_raw[obs]

    skewed_data = create_dataframe(X, A)
    data = create_dataframe(X_raw, A_raw)

    levels = [["female", "male"], ["white", "non-white"], ["Age Bucket 1", "Age Bucket 2"], ["little", "moderate", "quite rich"]]
    
    skewed_data["Creditability"] = y_0
    data["Creditability"] = y_0_raw

    raw_counts = build_counts(data, levels, "Creditability")
    counts = build_counts(skewed_data, levels, "Creditability")
    weights_features = torch.zeros(counts.numel(), 2*2*2*2*3)

    obs_prob = counts/raw_counts
    # More expresive paramterization
    #w >=1 because of inverse propensity weighting
    idx = 0
    # for target in [0, 1]:
    #     for m, se in enumerate(["female", "male"]):
    #         for i, race in enumerate(["white", "non-white"]):
    #                 for j, income in enumerate(["little", "moderate", "quite rich"]):
    #                     for k, housing in enumerate(["own", "rent"]):
    #                         for l, age in enumerate(['Age Bucket 1', 'Age Bucket 2']):
    #                             credit_se_features = [0]*4
    #                             credit_se_features[target*2 + m] = 1
    #                             income_features = [0]*3
    #                             income_features[j] = 1
    #                             weights_features[idx] = torch.tensor(credit_se_features + income_features ).float()
    #                             idx += 1

    for target in [0, 1]:
        for m, se in enumerate(["female", "male"]):
            for i, race in enumerate(["white", "non-white"]):
                for l, age in enumerate(['Age Bucket 1', 'Age Bucket 2']):
                            for j, income in enumerate(["little", "moderate", "quite rich"]):
                                # credit_features = [0]*2
                                # credit_features[target] = 1
                                # sex_features = [0]*2
                                # sex_features[target] = 1
                                # race_features = [0]*2
                                # race_features[i] = 1
                                # income_features = [0]*3
                                # income_features[j] = 1
                                # housing_features = [0]*2
                                # housing_features[k] = 1
                                # age_features = [0]*2
                                # age_features[l] = 1
                                features = [0]*(2*2*2*2*3)
                                features[idx] = 1
                                weights_features[idx] = torch.tensor(features).float()
                                idx += 1
    print("weights_features", weights_features)

    y00_female = sum((data["Creditability"] == 0) & (data["female"] == 1))
    y01_female = sum((data["Creditability"] == 1) & (data["female"] == 1))

    y00_male = sum((data["Creditability"] == 0) & (data["male"] == 1))
    y01_male = sum((data["Creditability"] == 1) & (data["male"] == 1))
   
    b0 = np.array([y00_female, y00_male])
    b1 = np.array([y01_female, y01_male])

    data_count_0 = counts[0]
    data_count_1 = counts[1]

    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["female", "male"])
    
    # upper_bound = False
    # min_bound, min_loss_values = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, obs_prob)

    upper_bound = True
    max_bound, max_loss_values = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, obs_prob)

    c_time = datetime.datetime.now()
    timestamp = str(c_time.timestamp())
    timestamp = "_".join(timestamp.split("."))

    # print(f"min:{float(min_bound)} , gt:{gt_ate},  max:{float(max_bound)}")
    # plt.plot(min_loss_values)

    plt.plot(max_loss_values)
    # plt.axhline(y=gt_ate, color='r', linestyle='-')
    # plt.legend(["min", "max"])
    plt.savefig(f"losses_{timestamp}")