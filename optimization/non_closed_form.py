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
from model import WeightedLogisticRegression
from framework import compute_ate_conditional_mean, build_counts, build_dataset, \
                      build_strata_counts_matrix, get_folks_tables_data

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
        tuple_features = (columns_names[0], columns_names[1], columns_names[2])
        weight_index = hash_map[tuple_features]
        weight = weights_features[weight_index]
        weigths.append(weight)
    # import pdb; pdb.set_trace()
    return torch.stack(weigths)


def train_model(model, data, labels):
    """training procedure to produce uncertanty quantification.

    In order to produce uncertainty quantification on the coefficients,
    It is necessary to alternate the differiantiation between the coe
    fficients and the debiasing weights.

    Args:
        model (_type_): _description_
    """
    # BCE loss for fitting the logistic regression
   

    for i in range(20):
        
        
        for epoch in range(10):
            for instance, label in zip(data, labels):
                # Step 1. Remember that PyTorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2.
                features = torch.tensor(instance.astype(np.double)).float()
                # Step 3. Run our forward pass.
                if i == 0:
                    weight = 1.
                else:
                    weight = assign_weights(features, label, model.weights)
                probs = model(features) * weight

                with torch.no_grad():            
                        probs.clamp_(0, 1)
               
                target = torch.tensor(label, dtype = torch.long).expand((1)).float()

                # Step 4. Compute the loss, gradients, and update the parameters by
                loss = loss_function(probs, target)

                if loss.requires_grad:
                    loss.backward(create_graph=True)
                    optimizer.step()

        # Put an assert ensuring the betas have 0 gradients.
        coeff = next(model.linear.parameters())
        # import pdb; pdb.set_trace()
        loss_w = second_loss_function(-coeff[0][7], torch.tensor(0.))
        print(loss_w)
        loss_w.backward()
        optimizer_2.step()

        with torch.no_grad():            
            # are theese good assumptions?
            model.weights.clamp_(coeff[0][7], 0.25 - coeff[0][7])

    return coeff[0][0]
   
def run_search(A_0, A_1,data_count_1, data_count_0,
                     weights_features, upper_bound, gt_ate, dataset_size, data, labels, weights_array):
    """Runs the search for the optimal weights."""
   
    loss_function = torch.nn.BCELoss()
    # Two optimizers, they have the relevant parameters to optimize.
    model = WeightedLogisticRegression(NUM_FEATURES, NUM_LABELS)
    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(weights_features.shape[1], requires_grad=True)
    optim = torch.optim.Adam([alpha], 0.01)
    optim_2 = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    loss_values = []
    n_sample = data_count_1.sum() + data_count_0.sum()
    print(n_sample/dataset_size)
    size_t = data_count_1.select(-1, 1).sum() + data_count_0.select(-1, 1).sum()
    mean =  data_count_1.select(-1, 1).sum()/size_t
    print("mean", mean)
    for iteration in range(2000):
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
        weitght_loss = weights_array @ alpha
        for epoch in range(100):
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2.
            # import pdb; pdb.set_trace()
            features = data.to_numpy().astype(np.double)
            features_tensor = torch.tensor(features).float()
            # Step 3. Run our forward pass.
            # import pdb; pdb.set_trace()
            probs = model(features_tensor) * torch.unsqueeze(weitght_loss, -1)

            with torch.no_grad():            
                    probs.clamp_(0, 1)
            
            target = torch.tensor(labels, dtype = torch.long).float()
            target = torch.unsqueeze(target, -1)

            # Step 4. Compute the loss, gradients, and update the parameters by
            loss = loss_function(probs, target)

            if loss.requires_grad:
                loss.backward(create_graph=True)
                optim_2.step()
        # TODO: Check if the gradeint on alpha is backpropagating trough the inner loop.
        # import pdb; pdb.set_trace()
        
        loss =  model.linear.weight[0][0]
        beta = -loss if upper_bound else loss
        loss_values.append(beta.detach().numpy())
        if iteration % 500 == 0:
            print(loss)
            print(f"ATE: {beta.item()}, ground_truth {gt_ate}")
        
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

    baseline_model = LogisticRegression(random_state=0).fit(data, label)
    ground_truth_model = LogisticRegression(random_state=0).fit(skewed_data, y)
    
    # import pdb; pdb.set_trace()
    print(X_sample.shape[0])
    print(baseline_model.coef_[0][0])
    print(X.shape[0])
    print(ground_truth_model.coef_[0][0])

    baseline = baseline_model.coef_[0][0]
    ground_truth = ground_truth_model.coef_[0][0]

    for level in levels:
        number_strata *= len(level)
    
    number_strata = 1
    for level in levels:
        number_strata *= len(level)
    print(number_strata)

    idx = 0
    idj = 0
    weights_features = torch.zeros(number_strata, 5*4*2)
    # starting_tuple = ("white", '0_0', '1_0', '2_0')
    starting_tuple = ('0_0', '1_0', '2_0')
    previous_tuple = starting_tuple
    flag = True
    hash_map = {}

    for combination in traverse_combinations(levels):
        current_tuple = (combination[1], combination[2], combination[3])
        if previous_tuple != current_tuple:
            if current_tuple == starting_tuple:
                idj = 0
            else:
                idj += 1
        weight = [0]*(5*4*2)
        weight[idj] = 1
        weights_features[idx] = torch.tensor(weight).float()

        hash_map[current_tuple] = idx
        idx += 1
        # if idx % 24 == 1:
        #     print(current_tuple, previous_tuple)
        #     print(idx, number_strata)
        #     print(weights_features.sum(axis=0))
        previous_tuple = current_tuple
    print(weights_features.sum(axis=0))

    # import pdb;pdb.set_trace()
    weights_array = assign_weights(skewed_data, hash_map, weights_features)
    print(f"Shape number of weights {weights_array.shape}")

    # # weights_features = torch.eye(number_strata)
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
    
    # # gt_ate = compute_debias_ate_ipw()
    # # np.save("baselines", np.array([biased_empirical_mean, empirical_mean, ipw]))
    # # print("baselines:", [biased_empirical_mean, empirical_mean, ipw])
    mean_race = compute_ate_conditional_mean(1 - group_sample, y)
    mean_race_t = data_count_1[1].sum()/(data_count_1[1].sum() + data_count_0[1].sum()) 
    print("test", mean_race, mean_race_t)
    
    # Observed
    # REVISAR !!!!!!!
    biased_empirical_mean = compute_ate_conditional_mean(1 - sex_group, y)
    mean_race_t = data_count_1.select(-1,1).sum()/(data_count_1.select(-1,1).sum() + data_count_0.select(-1,1).sum()) 
    print("test_2", biased_empirical_mean, mean_race_t)
    # Real
    empirical_mean = compute_ate_conditional_mean(1 - sex, label)
    print(empirical_mean, biased_empirical_mean)

    gt_ate = empirical_mean
    NUM_FEATURES = len(skewed_data.columns[:-1])
    print(f"NUM_FEATURES : {NUM_FEATURES}")
    NUM_LABELS = 1
    skewed_data_features = skewed_data[skewed_data.columns[:-1]] 
    for index in range(1):
        upper_bound = True
        max_bound, max_loss_values, alpha_max = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, dataset_size, skewed_data_features, y, weights_array)

        upper_bound = False
        min_bound, min_loss_values, alpha_min = run_search(A0, A1, data_count_1, data_count_0, weights_features, upper_bound, gt_ate, dataset_size, skewed_data_features, y, weights_array)

        c_time = datetime.datetime.now()
        timestamp = str(c_time.timestamp())
        timestamp = "_".join(timestamp.split("."))
    #     np.save(f"numerical_results/min_loss_4_{index}", min_loss_values)
    #     np.save(f"numerical_results/max_loss_4_{index}", max_loss_values)

        print(f"min:{float(min_bound)} , gt:{gt_ate},  max:{float(max_bound)}")
        plt.plot(min_loss_values)
        plt.plot(max_loss_values)
        plt.axhline(y=ground_truth, color='cyan', linestyle='dashed')
        plt.axhline(y=baseline, color='olive', linestyle='dashed')
        plt.legend(["min", "max", "Ground truth", "baseline"])
        plt.title("Model output.")# plt.title("Learning Curves for 10 trials.")
        plt.savefig(f"losses_{timestamp}")

