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

DATASET_SIZE = 100000 

def empirical_propensity_score(X, A):
    """Computes the empirical propensity scores.
     
    Computes the empirical propensity scores for each combination of 
    feature values.
    """
    unique_vals = [np.unique(X[:, i]) for i in range(X.shape[1])]
    
    counts = np.zeros([len(unique_vals[0]), len(unique_vals[1])])
    tot_counts = np.zeros([len(unique_vals[0]), len(unique_vals[1])])
    for i in range(X.shape[0]):
        if A[i] == 1:
            counts[X[i,0], X[i,1]] += 1
        tot_counts[X[i,0], X[i,1]] += 1

    probs = counts / tot_counts

    empirical_probs = np.zeros([X.shape[0]])
    for i in range(X.shape[0]):
        empirical_probs[i] = probs[X[i,0], X[i,1]]

    return empirical_probs

def prpensity_scores_by_strata(levels: List[List]):
    """Computes the propensity scores by strata.
    """
    shape = [0 for i in range(len(levels))]        
    
    for index, level in enumerate(levels):
        number_levels = len(level)
        shape[index] = number_levels
        
    propensity_scores = torch.zeros(shape)

    for i in range(3):
        for j in range(2):
            propensity_scores[i,j] = scipy.special.expit(j - i)

    return propensity_scores

def compute_debias_ate_ipw():
    """Computes the ate using the potential outcomes distribution.

    Args:
        A (_type_): vector of observed outcomes
        propensity_scores (_type_): Vector of propensity scores
        y (_type_): response variable
    
    Returns
        ate: Average treatment effect.
    """
    #TODO (Santiago) : Deprecate this function, move functionality to simulate_multiple_outcomes.
    #TODO (Santiago) : Create unit test for this.
    y0_gt = scipy.special.expit(0)*(0.5*0.6) + scipy.special.expit(-1)*(0.6*0.3) + scipy.special.expit(-2)*(0.2*0.6) + scipy.special.expit(1)*(0.5*0.4) + scipy.special.expit(0)*(0.3*0.4) + scipy.special.expit(-1)*(0.2*0.4)
    y1_gt = scipy.special.expit(2)*(0.5*0.6) + scipy.special.expit(1)*(0.6*0.3) + scipy.special.expit(0)*(0.2*0.6) + scipy.special.expit(3)*(0.5*0.4) + scipy.special.expit(2)*(0.3*0.4) + scipy.special.expit(1)*(0.2*0.4)

    # px_1 = [0.5, 0.3, 0.2]
    # px_2 = [0.6, 0,4]
    # for i in range(3):
    #     for j in range(2):
    #         y0_gt += scipy.special.expit( - i + j)*(px_1[i]*px_2[j])
    #         y1_gt += scipy.special.expit(2 - i + j )*(px_1[i]*px_2[j])

    return y1_gt - y0_gt

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

def simulate_multiple_outcomes(dataset_size: int):
    """Simulates a dataset with multiple outcomes.
    
    The following function simulates a data generating process with multiple
    outcomes. Moreover, it also generates a sample of observations from the
    original dataset.

    Args:
        dataset_size (int): The dataset size.

    Returns:
        Tuple: the sample from the generated dataset.
    """
    
    X = np.random.choice(a=[0, 1, 2], size=dataset_size, p=[0.5, 0.3, 0.2])
    X_2 =np.random.binomial(size=dataset_size, n=1, p=0.4)

    pi_A = scipy.special.expit(X_2 - X)
    A = 1*(pi_A > np.random.uniform(size=dataset_size))
    mu = scipy.special.expit(2*A - X + X_2)
    y = 1*(mu > np.random.uniform(size=dataset_size))
    
    mu2 = scipy.special.expit((X + X_2)/2 - A)
    y2 = 1*(mu2 > np.random.uniform(size=dataset_size))

    obs = scipy.special.expit(X - X_2) > np.random.uniform(size=dataset_size)
    X_total = np.stack((X, X_2), axis=-1)

    return X_total, A, y, obs, y2

def create_dataframe(X: np.array, A: np.array):
    """Creates a dataframe with the data.

    The data creates dummie variables from categorical variables passed as
    parameters as well as the treatment variable.
    """
    skewed_data = pd.DataFrame()
    skewed_data[["little", "moderate", "quite rich"]] = pd.get_dummies(X[:,0])
    skewed_data[["White", "Non White"]] = pd.get_dummies(X[:,1])
    skewed_data[["female", "male"]] = pd.get_dummies(A)
    return skewed_data




def run_search(num_constr, A_0, A_1,data_count_1, data_count_0,
                     weights_features, upper_bound, gt_ate, pr_r, dataset_size=DATASET_SIZE):
    """Runs the search for the optimal weights."""

    pr_r_t = torch.tensor(pr_r)
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
        other_A_0 = other_A0.numpy()
        other_A_1 = other_A1.numpy()
        
        objective = cp.sum_squares(w - alpha_fixed)
        # restrictions = [A_0@ w == b0, A_1@ w == b1, weights_features@w >= 1]
        restrictions = [weights_features@w >= n_sample/dataset_size,
                        A_0@ w == b0, 
                        A_1@ w == b1,
                        # other_A_0@ w == other_b0] 
                        other_A_1@ w == other_b1]
        prob = cp.Problem(cp.Minimize(objective), restrictions[:num_constr + 1])
        prob.solve()
        
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



if __name__ == '__main__':
    

    # Generates the data.
    X_raw, A_raw, y_raw, obs, y2_raw = simulate_multiple_outcomes(DATASET_SIZE)
    X, A, y, y2 = X_raw[obs], A_raw[obs], y_raw[obs], y2_raw[obs]

    skewed_data = create_dataframe(X, A)
    data = create_dataframe(X_raw, A_raw)

    levels = [["female", "male"],  ["little", "moderate", "quite rich"], \
                                   ["White", "Non White"]]

    skewed_data["Creditability"] = y
    data["Creditability"] = y_raw
    counts = build_counts(skewed_data, levels, "Creditability")

    skewed_data["other_outcome"] = y2
    data["other_outcome"] = y2_raw
    counts2 = build_counts(skewed_data, levels, "other_outcome")


    #Creates a parametrization to compute the weights for each feature strata.
    weights_features = torch.zeros(12, 12)
    idx = 0
    for m, se in enumerate(["female", "male"]):
        for j, income in enumerate(["little", "moderate", "quite rich"]):
            for i, race in enumerate(["White", "Non White"]):
                features = [0]*(2*3*2)
                features[idx] = 1
                weights_features[idx] = torch.tensor(features).float()
                idx += 1 


    # Creates groundtruth values to generate linear restrictions.
    y00_female = sum((data["Creditability"] == 0) & (data["female"] == 1))
    y01_female = sum((data["Creditability"] == 1) & (data["female"] == 1))

    y00_male = sum((data["Creditability"] == 0) & (data["male"] == 1))
    y01_male = sum((data["Creditability"] == 1) & (data["male"] == 1))

    b0 = np.array([y00_female, y00_male])
    b1 = np.array([y01_female, y01_male])

    data_count_0 = counts[0]
    data_count_1 = counts[1]
    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["female", "male"])

    # Creates groundtruth values to generate linear restrictions *for other outcome*.
    other_y00_female = sum((data["other_outcome"] == 0) & (data["female"] == 1))
    other_y01_female = sum((data["other_outcome"] == 1) & (data["female"] == 1))

    other_y00_male = sum((data["other_outcome"] == 0) & (data["male"] == 1))
    other_y01_male = sum((data["other_outcome"] == 1) & (data["male"] == 1))

    other_b0 = np.array([other_y00_female, other_y00_male])
    other_b1 = np.array([other_y01_female, other_y01_male])

    other_A0, other_A1 = build_strata_counts_matrix(weights_features, counts2, ["female", "male"])

    
    gt_propensity_weighting_A = scipy.special.expit(X_raw[:,1] - X_raw[:,0]) 
    gt_propensity_weighting_R =  scipy.special.expit(X_raw[:, 0] - X_raw[:, 1])
    population_propensity_weighting_A = scipy.special.expit(X[:,1] - X[:,0])

    empirical_probs = empirical_propensity_score(X, A)
    gt_ate = compute_debias_ate_ipw()
    biased_ipw = compute_ate_ipw(A, population_propensity_weighting_A, y)
    empirical_biased_ipw = compute_ate_ipw(A, empirical_probs, y)
    prop_score_tensor = prpensity_scores_by_strata([["little", "moderate", "quite rich"],["White", "Non White"]])

    emprical_conditonal_mean_1 = compute_ate_conditional_mean(X[:, -1], y)
    # emprical_conditonal_mean_0 = compute_ate_conditional_mean(y, 1 - A)
    # emp_ate = emprical_conditonal_mean_1 - emprical_conditonal_mean_0
    true_conditonal_mean = compute_ate_conditional_mean(X_raw[:, -1], y_raw)

    # import pdb; pdb.set_trace()
    w_counts_1 = data_count_1.select(-1, 1)
    w_counts_0 = data_count_0.select(-1, 1)

    size = w_counts_1.sum() + w_counts_0.sum()
    ate = w_counts_1.sum()/size
    print(f"Conditional mean: {ate}, {emprical_conditonal_mean_1}")

    for num_constr in range(1,4):
        for index in range(1):

            upper_bound = True
            max_bound, max_loss_values, alpha_max = run_search(num_constr, A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, prop_score_tensor)

            upper_bound = False
            min_bound, min_loss_values, alpha_min = run_search(num_constr, A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, prop_score_tensor)

            c_time = datetime.datetime.now()
            timestamp = str(c_time.timestamp())
            timestamp = "_".join(timestamp.split("."))
            # np.save(f"res/min_loss_c{num_constr}_{index}", min_loss_values)
            # np.save(f"res/max_loss_c{num_constr}_{index}", max_loss_values)

            print(f"min:{float(min_bound)} , gt:{gt_ate},  max:{float(max_bound)}")
            plt.plot(min_loss_values)
            plt.plot(max_loss_values)
            # plt.axhline(y=gt_ate, color='g', linestyle='dashed')
            plt.axhline(y=true_conditonal_mean, color='cyan', linestyle='dashed')
            plt.axhline(y=emprical_conditonal_mean_1, color='olive', linestyle='dashed')
            plt.legend(["min", "max", "True Conditional mean", "Empirical conditional mean"])
            plt.title("Conditional mean.")# plt.title("Learning Curves for 10 trials.")

        # print(f"min:{float(min_bound)} , gt:{gt_ate},  max:{float(max_bound)}")
        # plt.plot(min_loss_values)
        # plt.plot(max_loss_values)
        # plt.axhline(y=gt_ate, color='g', linestyle='dashed')
        # plt.axhline(y=empirical_biased_ipw, color='cyan', linestyle='dashed')
        # plt.axhline(y=emp_ate, color='olive', linestyle='dashed')
        # plt.legend(["min", "max", "Ground truth ate", "IPW", "Empirical a
    plt.savefig(f"losses_{timestamp}")
