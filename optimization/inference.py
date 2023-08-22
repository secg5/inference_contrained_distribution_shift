import datetime
import itertools
import os
from typing import Dict, List, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import pickle
from tqdm import tqdm

import argparse
import json
from config import parse_config_dict
from datasets import FolktablesLoader, SimulationLoader

def traverse_combinations(list_of_lists):
    for combination in itertools.product(*list_of_lists):
        yield combination

def read_json(config_filename):
    # Load configuration from JSON file
    with open(config_filename, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict

def get_feature_weights(
    matrix_type: str, dataset_type: str, levels: List[List[str]]
) -> torch.Tensor:
    """Creates a parametrization to compute the weights for each feature strata.

    Each one of if statement is equivalent to the following for loops respectively:

    weights_features = torch.zeros(12, 12)
    idx = 0
    for m, se in enumerate(["female", "male"]):
        for j, income in enumerate(["little", "moderate", "quite rich"]):
            for i, race in enumerate(["White", "Non White"]):
                features = [0]*(2*3*2)
                features[idx] = 1
                weights_features[idx] = torch.tensor(features).float()
                idx += 1 
    --------------------------------------------
    weights_features = torch.zeros(12, 6)
    idx = 0
    for m, se in enumerate(["female", "male"]):
        idm = 0
        for j, income in enumerate(["little", "moderate", "quite rich"]):
            for i, race in enumerate(["White", "Non White"]):
                features = [0]*(3*2)
                features[idm] = 1
                weights_features[idx] = torch.tensor(features).float()
                idx += 1 
                idm += 1
    --------------------------------------------
    weights_features = torch.zeros(12, 8)
    idx = 0
    idj = 0
    for m, se in enumerate(["female", "male"]):
        idm = 0
        for j, income in enumerate(["little", "moderate", "quite rich"]):
            for i, race in enumerate(["White", "Non White"]):
                features = [0]*(3*2)
                sex_features = [0]*2
                features[idm] = 1
                sex_features[idj] = 1
                weights_features[idx] = torch.tensor(features + sex_features).float()
                idx += 1 
                idm += 1
        idj += 1
    """
    if dataset_type == "simulation":
        if matrix_type == "Nx12":
            return torch.eye(12, 12)
        elif matrix_type == "Nx8":
            block_1 = torch.eye(6, 6)
            block_1[:, -2] = 1
            block_1[:, -1] = 0

            block_2 = torch.eye(6, 6)
            block_2[:, -2] = 0
            block_2[:, -1] = 1
            return torch.cat([block_1, block_2])
        elif matrix_type == "Nx6":
            return torch.cat([torch.eye(6, 6), torch.eye(6, 6)])
        else:
            raise ValueError(
                f"Invalid feature matrix type {matrix_type} for simulated dataset."
            )
    elif dataset_type == "folktables":
        number_strata = 1
        for level in levels:
            number_strata *= len(level)

        if matrix_type == "Nx12":
            feature_weights = torch.eye(number_strata)
            return feature_weights
        elif matrix_type == "Nx8":
            idx = 0
            idj = 0
            idm = 0
            feature_weights = torch.zeros(number_strata, 5*4*2 + 2)
            starting_tuple = ('MIL_0', 'ANC_0', 'NATIVITY_0')
            previous_tuple = starting_tuple
            starting_feature = 'white'
            previous_feature = starting_feature
            for combination in traverse_combinations(levels):
                current_tuple = (combination[1], combination[2], combination[3])
                current_feature = combination[0]
                if previous_tuple != current_tuple:
                    if current_tuple == starting_tuple:
                        idj = 0
                    else:
                        idj += 1
                if previous_feature != current_feature:
                    if current_feature == starting_feature:
                        idm = 0
                    else:
                        idm += 1

                weight = [0]*(5*4*2)
                weight[idj] = 1
                weight_race = [0]*2
                weight_race[idm] = 1 
                feature_weights[idx] = torch.tensor(weight + weight_race).float()
                idx += 1
                previous_tuple = current_tuple
                previous_feature = current_feature
            return feature_weights

        elif matrix_type == "Nx6":
            idx = 0
            idj = 0
            feature_weights = torch.zeros(number_strata, 5*4*2)
            starting_tuple = ('MIL_0', 'ANC_0', 'NATIVITY_0')
            previous_tuple = starting_tuple
            for combination in traverse_combinations(levels):
                current_tuple = (combination[1], combination[2], combination[3])
                if previous_tuple != current_tuple:
                    if current_tuple == starting_tuple:
                        idj = 0
                    else:
                        idj += 1
                weight = [0]*(5*4*2)
                weight[idj] = 1
                feature_weights[idx] = torch.tensor(weight).float()
                idx += 1
                previous_tuple = current_tuple
            return feature_weights
        else:
            raise ValueError(
                f"Invalid feature matrix type {matrix_type} for folktables dataset."
            )
    else:
        raise ValueError(f"Invalid dataset type {dataset_type}.")


def traverse_level_combinations(levels: List[List[str]]) -> List[Tuple]:
    for combination in itertools.product(*levels):
        yield combination


def get_feature_strata(
    df: pd.DataFrame, levels: List[List[str]], target: str
) -> Dict[Tuple, pd.DataFrame]:
    levels_combinations = list(itertools.product(*levels))
    strata_dfs = dict()
    for combination in levels_combinations:
        query_string = " & ".join([f"`{level}` == 1" for level in combination])
        strata_dfs[(0,) + combination] = df[df[target] == 0].query(query_string)

    for combination in levels_combinations:
        query_string = " & ".join([f"`{level}` == 1" for level in combination])
        strata_dfs[(1,) + combination] = df[df[target] == 1].query(query_string)
    return strata_dfs


def get_strata_counts(
    strata_dfs: Dict[Tuple, pd.DataFrame], levels: List[List[str]]
) -> torch.Tensor:
    strata_counts = list()
    # Ensure same order?
    for strata_df in strata_dfs.values():
        if strata_df.shape[0] == 0:
            strata_counts.append(0.00001)
        else:
            strata_counts.append(strata_df.shape[0])

    output_shape = [2] + [len(level) for level in levels]
    return torch.tensor(strata_counts).reshape(output_shape)


def get_count_restrictions(
    data: pd.DataFrame, target: str, treatment_level: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    y00_val_1 = sum((data[target] == 0) & (data[treatment_level[0]] == 1))
    y01_val_1 = sum((data[target] == 1) & (data[treatment_level[0]] == 1))

    y00_val_2 = sum((data[target] == 0) & (data[treatment_level[1]] == 1))
    y01_val_2 = sum((data[target] == 1) & (data[treatment_level[1]] == 1))

    restriction_00 = np.array([y00_val_1, y00_val_2])
    restriction_01 = np.array([y01_val_1, y01_val_2])

    return restriction_00, restriction_01

def compute_f_divergence(p, q, type="chi2"):
    if type == "chi2":
        # return 0.5 * torch.sum((p-q)**2/(p+q+1e-8))
        return 0.5 * torch.sum((p-q)**2/(q+1e-8))
    else:
        raise ValueError(f"Invalid divergence type {type}.")


    # # Creates groundtruth values to generate linear restrictions *for other outcome*.
    # other_y00_female = sum((data["other_outcome"] == 0) & (data["female"] == 1))
    # other_y01_female = sum((data["other_outcome"] == 1) & (data["female"] == 1))

    # other_y00_male = sum((data["other_outcome"] == 0) & (data["male"] == 1))
    # other_y01_male = sum((data["other_outcome"] == 1) & (data["male"] == 1))

    # other_b0 = np.array([other_y00_female, other_y00_male])
    # other_b1 = np.array([other_y01_female, other_y01_male])

    # other_A0, other_A1 = build_strata_counts_matrix(weights_features, counts2, ["female", "male"])


# def get_cov_restrictions(
#     data: pd.DataFrame, target: str, treatment_level: List[str], cov_vars: List[str]
# ) -> Tuple[np.ndarray, np.ndarray]:
#     y_00_val_1_df = data[(data[target] == 0) & (data[treatment_level[0]] == 1)]
#     y_01_val_1_df = data[(data[target] == 1) & (data[treatment_level[0]] == 1)]
#     y_00_val_2_df = data[(data[target] == 0) & (data[treatment_level[1]] == 1)]
#     y_01_val_2_df = data[(data[target] == 1) & (data[treatment_level[1]] == 1)]

#     y_00_val_1 = y_00_val_1_df.cov().loc[*cov_vars]
#     y_01_val_1 = y_01_val_1_df.cov().loc[*cov_vars]
#     y_00_val_2 = y_00_val_2_df.cov().loc[*cov_vars]
#     y_01_val_2 = y_01_val_2_df.cov().loc[*cov_vars]

#     restriction_00 = np.array([y_00_val_1, y_00_val_2])
#     restriction_01 = np.array([y_01_val_1, y_01_val_2])
#     return restriction_00, restriction_01


def build_strata_counts_matrix(
    feature_weights: torch.Tensor, counts: torch.Tensor, treatment_level: List[str]
):
    """Builds linear restrictions for a convex optimization problem, according to
    a specific restrictions parameterization. This is the phi parametrization described
    in the paper.

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
    _, features = feature_weights.shape
    level_size = len(treatment_level)

    y_0_ground_truth = torch.zeros(level_size, features)
    y_1_ground_truth = torch.zeros(level_size, features)

    data_count_0 = counts[0]
    data_count_1 = counts[1]

    for level in range(level_size):
        t = data_count_0[level].flatten().unsqueeze(1)
        features = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
        y_0_ground_truth[level] = (features * t).sum(dim=0)

    for level in range(level_size):
        t_ = data_count_1[level].flatten().unsqueeze(1)
        features_ = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
        y_1_ground_truth[level] = (features_ * t_).sum(dim=0)

    return y_0_ground_truth.numpy(), y_1_ground_truth.numpy()


# def build_strata_covs_matrix(
#     feature_weights: torch.Tensor, covs: torch.Tensor, treatment_level: List[str]
# ):
#     _, features = feature_weights.shape
#     level_size = len(treatment_level)

#     y_0_ground_truth = torch.zeros(level_size, features)
#     y_1_ground_truth = torch.zeros(level_size, features)

#     data_covs_0 = covs[0]
#     data_covs_1 = covs[1]

#     for level in range(level_size):
#         t = data_covs_0[level].flatten().unsqueeze(1)
#         features = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
#         y_0_ground_truth[level] = (features * t).mean(dim=0)

#     for level in range(level_size):
#         t_ = data_covs_1[level].flatten().unsqueeze(1)
#         features_ = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
#         y_1_ground_truth[level] = (features_ * t_).mean(dim=0)

#     return y_0_ground_truth.numpy(), y_1_ground_truth.numpy()


def get_restrictions(
    restriction_type: str, A_dict, w, restriction_values, n_sample, 
    dataset_size, rho, ratio, q
):
    restrictions = [feature_weights @ w >= n_sample / dataset_size]
    if restriction_type == "count":
        restrictions += [
            A_dict["count"][0] @ w == restriction_values["count"][0],
            A_dict["count"][1] @ w == restriction_values["count"][1],
        ]
    elif restriction_type == "count_minus":
        restrictions += [
            A_dict["count_minus"][0] @ w == restriction_values["count_minus"][0]
        ]
    elif restriction_type == "count_plus":
        restrictions += [
            A_dict["count_plus"][0] @ w == restriction_values["count_plus"][0],
            A_dict["count_plus"][1] @ w == restriction_values["count_plus"][1],
            A_dict["count_plus"][3] @ w == restriction_values["count_plus"][3],
        ]
    elif restriction_type == "DRO":
       chi_square_divergence = cp.multiply((ratio - 1)**2, q)
       restrictions += [ .5 * cp.sum(chi_square_divergence) <= rho]
    else:
        raise ValueError(f"Restriction type {restriction_type} not supported.")
    return restrictions


def run_search(
    A_dict,
    strata_estimands,
    feature_weights,
    restriction_values,
    upper_bound,
    n_iters,
    restriction_type,
    dataset_size,
    rho
):
    """Runs the search for the optimal weights."""

    data_count_0 = strata_estimands["count"][0]
    data_count_1 = strata_estimands["count"][1]

    torch.autograd.set_detect_anomaly(True)
    alpha = torch.rand(feature_weights.shape[1], requires_grad=True)
    optim = torch.optim.Adam([alpha], 0.01)
    loss_values = []
    n_sample = data_count_1.sum() + data_count_0.sum()

    for _ in tqdm(
        range(n_iters), desc=f"Optimizing Upper Bound: {upper_bound}", leave=False
    ):
        w = cp.Variable(feature_weights.shape[1])
        alpha_fixed = alpha.squeeze().detach().numpy()

        ratio = feature_weights @ w
        q = (data_count_1 + data_count_0) / n_sample
        q = q.reshape(-1, 1)
        q = torch.flatten(q)

        objective = cp.sum_squares(w - alpha_fixed)
        restrictions = get_restrictions(
            restriction_type=restriction_type,
            A_dict=A_dict,
            w=w,
            restriction_values=restriction_values,
            n_sample=n_sample,
            dataset_size=dataset_size,
            rho=rho,
            ratio=ratio,
            q=q
        )

        prob = cp.Problem(cp.Minimize(objective), restrictions)
        prob.solve()

        alpha.data = torch.tensor(w.value).float()
        weights_y1 = (feature_weights @ alpha).reshape(*data_count_1.shape)
        weights_y0 = (feature_weights @ alpha).reshape(*data_count_0.shape)
        weighted_counts_1 = weights_y1 * data_count_1
        weighted_counts_0 = weights_y0 * data_count_0

        w_counts_1 = weighted_counts_1.select(-1, 1)
        w_counts_0 = weighted_counts_0.select(-1, 1)

        size = w_counts_1.sum() + w_counts_0.sum()
        conditional_mean = w_counts_1.sum() / size

        loss = -conditional_mean if upper_bound else conditional_mean
        loss_values.append(conditional_mean.detach().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()

    if upper_bound:
        ret = max(loss_values)
    else:
        ret = min(loss_values)

    return ret, loss_values, alpha


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    parser = argparse.ArgumentParser(description="Run experiments with JSON configuration.")
    parser.add_argument("config", help="Path to the JSON configuration file")
    args = parser.parse_args()
    config_dict = read_json(args.config)
    config = parse_config_dict(config_dict)

    rng = np.random.default_rng(config.random_seed)
    if config.dataset_type == "simulation":
        data_loader = SimulationLoader(
            dataset_size=config.dataset_size,
            correlation_coeff=config.correlation_coeff,
            rng=rng,
        )
        dataset = data_loader.load()
        treatment_level = dataset.levels[0]
        dataset_size = dataset.population_df.shape[0]
        sample_size = dataset.sample_df.shape[0]
    elif config.dataset_type == "folktables":
        data_loader = FolktablesLoader(
            rng=rng, states=config.states, feature_names=config.feature_names
        )
        dataset = data_loader.load()
        treatment_level = dataset.levels[0]
        dataset_size = dataset.population_df.shape[0]
        sample_size = dataset.sample_df.shape[0]

    restriction_values = {
        "count": get_count_restrictions(
            data=dataset.population_df,
            target=dataset.target,
            treatment_level=treatment_level,
        ), 
        "count_minus": get_count_restrictions(
            data=dataset.population_df,
            target=dataset.target,
            treatment_level=treatment_level,
        ),
        "count_plus": get_count_restrictions(
            data=dataset.population_df,
            target=dataset.target,
            treatment_level=treatment_level,
        ) + get_count_restrictions(
            data=dataset.population_df,
            target=dataset.alternate_outcome,
            treatment_level=treatment_level,
        )
    }

    strata_dfs = get_feature_strata(
        df=dataset.sample_df, levels=dataset.levels, target=dataset.target
    )
    strata_dfs_alternate_outcome = get_feature_strata(
        df=dataset.sample_df, 
        levels=dataset.levels, 
        target=dataset.alternate_outcome
    )
    strata_dfs_population = get_feature_strata(
        df=dataset.population_df, 
        levels=dataset.levels, 
        target=dataset.target
    )

    strata_estimands = {
        "count": get_strata_counts(
            strata_dfs=strata_dfs, levels=dataset.levels
        ),
        "count_plus": get_strata_counts(
            strata_dfs=strata_dfs_alternate_outcome, levels=dataset.levels
        )
    }

    strata_estimands_population = {
        "DRO": get_strata_counts(
            strata_dfs=strata_dfs_population, levels=dataset.levels
        )
    }

    rho = compute_f_divergence(
                            strata_estimands_population["DRO"]/dataset_size,
                            strata_estimands["count"]/sample_size,
                            type="chi2"
                            )
    print("rho: ", rho)
    c_time = datetime.datetime.now()
    timestamp = str(c_time.strftime("%b%d-%H%M"))
    plotting_dfs = []

    for matrix_type in tqdm(config.matrix_types, desc="Matrix Type"):
        feature_weights = get_feature_weights(
            matrix_type=matrix_type,
            dataset_type=config.dataset_type,
            levels=dataset.levels,
        )
        A_dict = {
            "count": build_strata_counts_matrix(
                feature_weights, strata_estimands["count"], treatment_level
            ),
            "count_minus": build_strata_counts_matrix(
                feature_weights, strata_estimands["count"], treatment_level
            ),
            "count_plus": build_strata_counts_matrix(
                feature_weights, strata_estimands["count"], treatment_level
            ) + build_strata_counts_matrix(
                feature_weights, strata_estimands["count_plus"], treatment_level
            )
        }

        for restriction_idx, restriction_type in tqdm(
            enumerate(config.restriction_trials),
            desc="Restrictions",
            total=len(config.restriction_trials),
            leave=False,
        ):
            if restriction_type == "DRO" and matrix_type != "Nx12":
                continue
            for trial_idx in tqdm(range(config.n_trials), desc="Trials", leave=False):
                max_bound, max_loss_values, alpha_max = run_search(
                    A_dict=A_dict,
                    strata_estimands=strata_estimands,
                    feature_weights=feature_weights,
                    restriction_values=restriction_values,
                    upper_bound=True,
                    n_iters=config.n_optim_iters,
                    restriction_type=restriction_type,
                    dataset_size=dataset_size,
                    rho=rho
                )

                min_bound, min_loss_values, alpha_min = run_search(
                    A_dict=A_dict,
                    strata_estimands=strata_estimands,
                    feature_weights=feature_weights,
                    restriction_values=restriction_values,
                    upper_bound=False,
                    n_iters=config.n_optim_iters,
                    restriction_type=restriction_type,
                    dataset_size=dataset_size,
                    rho=rho
                )
                plotting_dfs.append(
                    pd.DataFrame(
                        {
                            "max_bound": max_bound,
                            "min_bound": min_bound,
                            "max_loss": max_loss_values,
                            "min_loss": min_loss_values,
                            "restriction_type": restriction_type,
                            "trial_idx": trial_idx,
                            "matrix_type": matrix_type,
                            "step": np.arange(len(max_loss_values)),
                        }
                    )
                )

    plotting_df = pd.concat(plotting_dfs).astype(
        {
            "max_loss": np.float64,
            "min_loss": np.float64,
            "step": np.int64,
            "trial_idx": np.int64,
            "max_bound": np.float64,
            "min_bound": np.float64,
            "restriction_type": str,
            "matrix_type": str,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.axhline(
        y=dataset.true_conditional_mean,
        color="cyan",
        linestyle="dashed",
        label="True Empirical Conditional Mean",
    )
    ax.axhline(
        y=dataset.empirical_conditional_mean,
        color="olive",
        linestyle="dashed",
        label="Empirical Conditional Mean",
    )
    sns.lineplot(
        data=plotting_df,
        x="step",
        y="max_loss",
        hue="restriction_type",
        style="matrix_type",
        palette="tab10",
        ax=ax,
    )
    sns.lineplot(
        data=plotting_df,
        x="step",
        y="min_loss",
        hue="restriction_type",
        style="matrix_type",
        palette="tab10",
        ax=ax,
        legend=False,
    )
    ax.set_title("Conditional Mean")
    fig.savefig(f"losses_{timestamp}")
    plotting_df.to_csv("plotting_df.csv")

    with open('dataset_metadata.pkl', 'wb') as outp:
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
