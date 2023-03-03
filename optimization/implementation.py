# TODO (Santiago): Brief description of the Script
import scipy
import torch
import itertools

import pandas as pd
import cvxpy as cp
import numpy as np

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
        position[0] = row[target]
        
        for index, level in enumerate(levels):
            for index_j, feature in enumerate(level):
                if row[feature] == 1:
                   position[index + 1] = index_j
        count[tuple(position)] += 1
    # Machetimbis para que después no me estallen los gradientes
    count[count == 0] = 0.00001
    return count       

def build_strata_counts_matrix(weight_features: torch.Tensor, counts: torch.Tensor, level: List[str]):
    """Builds linear restrictions for a convex opt problem.
    
    This method build a Matrix with counts by combination of strata,
    It will return two matrixes each one associated to the idividuals
    with y=1 or y=1:
    
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

if __name__ == '__main__':
    """The following file executes a training pipe-line 
    for a vectorized robust debiaser
    """

    # TODO (Santiago) Refactor Dataset creation for a simulated dataset
    data = pd.read_csv("german_credit_data.csv", index_col=0)
    data_labels = pd.read_csv("german_credit.csv")
    data_dummies = pd.get_dummies(data["Sex"])

    data_dummies[["free", "own", "rent"]] = pd.get_dummies(data["Housing"])
    data_dummies[["little", "moderate", "quite rich", "rich"]] = pd.get_dummies(data["Saving accounts"])
    data_dummies[["Age Bucket 1", "Age Bucket 2"]] = pd.get_dummies(pd.cut(data.Age, [18, 45, 75]))
    data_dummies["Creditability"] = data_labels["Creditability"]

    income_skewed_data = []
    confounders = ["little", "moderate", "quite rich", "rich"]
    # sample_prop = np.random.uniform(size=4)
    sample_prop = [0.8, 0.8, 0.4, 0.6]

    for ii in range(4):
        income_skewed_data.append(data_dummies[data_dummies[confounders[ii]] == 1].sample(frac=sample_prop[ii]))

    income_skewed_data = pd.concat(income_skewed_data)

    # Here is alist of all levels of the dataset: each one as a list containing its corresponding estrata
    levels = [['female', 'male'], ['free', 'own', 'rent'], ['little', 'moderate',
       'quite rich', 'rich'], ['Age Bucket 1', 'Age Bucket 2']]
   

    counts = build_counts(income_skewed_data, levels, "Creditability")
    weights_features = torch.zeros(counts.numel(), 9)
    idx = 0
    for target in [0, 1]:
        for i, sex in enumerate(["female", "male"]):
            for housing in ['free', 'own', 'rent']:
                for j, income in enumerate(['little', 'moderate', 'quite rich', 'rich']):
                    for age in ['Age Bucket 1', 'Age Bucket 2']:
                        credit_sex_features = [0]*4
                        credit_sex_features[target*2 + i] = 1
                        income_features = [0]*4
                        income_features[j] = 1
                        weights_features[idx] = torch.tensor(credit_sex_features + income_features + [1]).float()
                        idx += 1

    b0 = np.array([91, 209])
    b1 = np.array([219, 481])
    data_count_0 = counts[0]
    data_count_1 = counts[1]
    aux = data_dummies.groupby('female')['Creditability'].mean()
    gt_ate= aux[1] - aux[0]
    gt_ate

    A0, A1 = build_strata_counts_matrix(weights_features, counts, ["female", "male"])
    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True)
    W = np.unique(weights_features.numpy(), axis=0)
    tol = 0.00000001

    optim = torch.optim.Adam([alpha], lr=5e-4)

    for iteration in range(10000):
        w = cp.Variable(alpha.shape[0])
        alpha_fixed = alpha.squeeze().detach().numpy()
        A_0 = A0.numpy()
        A_1 = A1.numpy()

        objective = cp.sum_squares(w - alpha_fixed)
        # With the two rstrictions (as it should) the problem is infeasible W >=1 ????
        restrictions = [A_0@ w == b0, A_1@ w == b1]
        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()

        alpha.data = torch.tensor(w.value).float()
        weights_y1 = (weights_features[weights_features.shape[0]//2:]@alpha).reshape(*data_count_0.shape)
        weights_y0 = (weights_features[:weights_features.shape[0]//2]@alpha).reshape(*data_count_1.shape)
        
        
        weighted_counts_1 = weights_y1*data_count_1
        weighted_counts_0 = weights_y0*data_count_0

        sex = 1
        sex_base = 0
        
        probs = weighted_counts_1/(weighted_counts_1 + weighted_counts_0)

        
        total_weight_count = weighted_counts_1[sex] + weighted_counts_0[sex]
        ATE = ((probs[sex] - probs[sex_base])*total_weight_count/total_weight_count.sum()).sum()
        
        loss = ATE
        if iteration % 500 == 0:
            print(f"ATE: {ATE.item()}, ground_truth {gt_ate}")
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        