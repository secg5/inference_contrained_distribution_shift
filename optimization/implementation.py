from typing import List
import pandas as pd
import cvxpy as cp
import numpy as np
import torch
import itertools

def create_weight_matrix(levels: List[List]):
    """
    Creates all posible combinations of dummie variables from
    a list of levels.
    
    Example:
    From [[apple, orange], [juice, jam, salad]]
    
    returns: 
    [[1,0,1,0, 0],[1,0, 0, 1, 0],[1,0, 0,0, 1]
    [0,1,1,0, 0],[0,1,0,1, 0],[0,1,0,0, 1]]
    """
    weight_features = []
    index = 0
    vectors = []
    for feature in levels:
        number_levels = len(feature)
        diagonal = torch.ones([number_levels])
        one_hot_vectors = torch.diag(diagonal)
        vectors.append(one_hot_vectors)
    for pair in itertools.product(*vectors):
        weight_features.append(torch.cat(pair))
    weight_features = torch.stack(weight_features, axis=0)
    return weight_features
    
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
        t_ = data_count_1[level].flatten().unsqueeze(1)

        features = weight_features[level*t.shape[0]:(level + 1)*t.shape[0]]
        features_ = weight_features[level*t_.shape[0]:(level + 1)*t_.shape[0]]
        
        y_0_ground_truth[level] = (features*t).sum(dim=0)
        y_1_ground_truth[level] = (features_*t_).sum(dim=0)

    return y_0_ground_truth, y_1_ground_truth


if __name__ == '__main__':
    """The following file executes a training pipe-line 
    for a vectorized robust debiaser
    """


    data = pd.read_csv("german_credit_data.csv", index_col=0)
    data_labels = pd.read_csv("german_credit.csv")
    data_dummies = pd.get_dummies(data["Sex"])

    data_dummies[["free", "own", "rent"]] = pd.get_dummies(data["Housing"])
    data_dummies[["little", "moderate", "quite rich", "rich"]] = pd.get_dummies(data["Saving accounts"])
    data_dummies[["Age Bucket A", "Age Bucket B", "Age Bucket c", "Age Bucket D"]] = pd.get_dummies(pd.cut(data.Age, [18, 25, 45, 60, 75]))
    data_dummies["Creditability"] = data_labels["Creditability"]

    females = data_dummies[data_dummies.female == 1]
    females_good = females[females.Creditability == 1]
    females_bad = females[females.Creditability == 0]

    males = data_dummies[data_dummies.female == 0]
    males_good = males[males.Creditability == 1]
    males_bad = males[males.Creditability == 0]

    ma_good_sample = males_good.sample(frac = 0.25)
    fem_bad_sample = females_bad.sample(frac = 0.25)\

    sex_skewed_data = pd.concat([ma_good_sample, males_bad, females_good, fem_bad_sample])

    # Here is alist of all levels of the dataset: each one as a list containing its corresponding estrata
    levels = [['female', 'male'],['free', 'own', 'rent'], ['little', 'moderate',
        'quite rich', 'rich'], ['Age Bucket A', 'Age Bucket B', 'Age Bucket c',
        'Age Bucket D']]

    counts = build_counts(sex_skewed_data, levels, "Creditability")
    weight_features = create_weight_matrix(levels)
    data_count_0 = counts[0]
    data_count_1 = counts[1]

    b0 = np.array([91, 209])
    b1 = np.array([219, 481])

    A0, A1 = build_strata_counts_matrix(weight_features, counts, ["female", "male"])
    torch.autograd.set_detect_anomaly(True)
alpha = torch.rand(weight_features.shape[1], requires_grad=True)
W = np.unique(weight_features.numpy(), axis=0)
tol = 0.00000001

optim = torch.optim.Adam([alpha], lr=5e-2)

for iteration in range(10000):
    x = cp.Variable(alpha.shape[0])
    alpha_fixed = alpha.squeeze().detach().numpy()
    A_0 = A0.numpy()
    A_1 = A1.numpy()

    objective = cp.sum_squares(x - alpha_fixed)
    # With the two rstrictions (as it should) the problem is infeasible
    restrictions = [W@x >=1, A_0@ x == b0]
    prob = cp.Problem(cp.Minimize(objective), restrictions)
    prob.solve()

    #
    alpha.data = torch.tensor(x.value).float()
    weights_y1 = (weight_features@alpha).reshape(*data_count_0.shape)
    weights_y0 = (weight_features@alpha).reshape(*data_count_1.shape)
    
    weighted_counts_1 = weights_y1*data_count_1
    weighted_counts_0 = weights_y0*data_count_0

    sex = 1
    sex_base = 0
    
    probs = weighted_counts_1/(weighted_counts_1 + weighted_counts_0)

    total_weight_count = weighted_counts_1[sex] + weighted_counts_0[sex]
    # Here is the g-formula, The assertation now is sex
    diff = ((probs[sex] - probs[sex_base])*total_weight_count/total_weight_count.sum()).sum()
    total_weighted_count_base =  weighted_counts_1[sex_base] + weighted_counts_0[sex_base]
    base = (probs[sex_base]*total_weighted_count_base/total_weighted_count_base.sum()).sum()
    
    loss = diff
    if iteration % 500 == 0:
        print(diff.item(), diff.item()/base.item(), base.item())
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    