from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment
from scipy.special import expit


@dataclass
class Dataset:
    population_df: pd.DataFrame
    sample_df: pd.DataFrame
    levels: List[List[str]]
    target: str
    alternate_outcome: str
    empirical_conditional_mean: float
    true_conditional_mean: float
    population_df_colinear: pd.DataFrame = None
    sample_df_colinear: pd.DataFrame = None
    levels_colinear: List[List[str]] = None


class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Dataset:
        pass


class SimulationLoader(DatasetLoader):
    def __init__(
        self, dataset_size: int, correlation_coeff: float, rng: np.random.Generator
    ) -> None:
        self.dataset_size = dataset_size
        self.correlation_coeff = correlation_coeff
        self.rng = rng

    def _simulate_dataset(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulates a dataset with multiple outcomes.

        The following function simulates a data generating process with multiple
        outcomes. Moreover, it also generates a sample of observations from the
        original dataset.

        Args:
            dataset_size (int): The dataset size.
            rng (np.random.Generator): The random number generator.
            correlation_coeff (float, optional): The correlation coefficient
            to determine influence of X_1 and X_2 on indicator used to sample
            the biased dataset.

        Returns:
            Tuple: the sample from the generated dataset.
        """

        # X_1 = self.rng.choice(a=[0, 1, 2], size=self.dataset_size, p=[0.5, 0.3, 0.2])
        # X_2 = self.rng.binomial(size=self.dataset_size, n=1, p=0.4)

        # pi_A = expit(X_2 - X_1)
        # A = 1 * (pi_A > self.rng.uniform(size=self.dataset_size))
        # mu = expit(2 * A - X_1 + X_2)
        # y = 1 * (mu > self.rng.uniform(size=self.dataset_size))

        # mu2 = expit((X_1 + X_2)/2 - A)
        # y2 = 1*(mu2 > np.random.uniform(size=self.dataset_size))

        # obs = expit(X_1 - X_2) > self.rng.uniform(
        #     size=self.dataset_size
        # )
        # X_total = np.stack((X_1, X_2), axis=-1)

        X = np.random.choice(a=[0, 1, 2], size=self.dataset_size, p=[0.5, 0.3, 0.2])
        X_2 =np.random.binomial(size=self.dataset_size, n=1, p=0.4)

        pi_A = expit(X_2 - X)
        A = 1*(pi_A > np.random.uniform(size=self.dataset_size))
        mu = expit(2*A - X + X_2)
        y = 1*(mu > np.random.uniform(size=self.dataset_size))
        
        mu2 = expit((X + X_2)/2 - A)
        y2 = 1*(mu2 > np.random.uniform(size=self.dataset_size))

        obs = expit(X - X_2) > np.random.uniform(size=self.dataset_size)
        X_total = np.stack((X, X_2), axis=-1)

        return X_total, A, y, obs, y2

    def _create_dataframe(self, X: np.ndarray, A: np.ndarray) -> pd.DataFrame:
        """Creates a dataframe with the data.

        The data creates dummie variables from categorical variables passed as
        parameters as well as the treatment variable.
        """
        df = pd.DataFrame()
        df[["little", "moderate", "quite rich"]] = pd.get_dummies(X[:, 0])
        df[["White", "Non White"]] = pd.get_dummies(X[:, 1])
        df[["female", "male"]] = pd.get_dummies(A)
        return df

    def _get_ate_conditional_mean(self, A, y):
        """Computes the ate using inverse propensity weighting.

        Args:
            A (_type_): vector of observed outcomes
            propensity_scores (_type_): Vector of propensity scores
            y (_type_): response variable

        Returns
            ate: Average treatment effect.
        """
        conditional_mean = (y * A).sum()
        return conditional_mean / A.sum()

    def load(self) -> Dataset:
        levels = [
            ["female", "male"],
            ["little", "moderate", "quite rich"],
            ["White", "Non White"],
        ]

        X_raw, A_raw, y_raw, obs, y_2_raw = self._simulate_dataset()

        X = X_raw[obs]
        A = A_raw[obs]
        y = y_raw[obs]
        y_2 = y_2_raw[obs]

        empirical_conditional_mean = self._get_ate_conditional_mean(X[:, -1], y)
        true_conditional_mean = self._get_ate_conditional_mean(X_raw[:, -1], y_raw)

        sample_df = self._create_dataframe(X, A)
        sample_df["Creditability"] = y
        sample_df["other_outcome"] = y_2

        population_df = self._create_dataframe(X_raw, A_raw)
        population_df["Creditability"] = y_raw
        population_df["other_outcome"] = y_2_raw

        dataset = Dataset(
            population_df=population_df,
            sample_df=sample_df,
            levels=levels,
            target="Creditability",
            alternate_outcome="other_outcome",
            empirical_conditional_mean=empirical_conditional_mean,
            true_conditional_mean=true_conditional_mean,
        )
        return dataset


class FolktablesLoader(DatasetLoader):
    def __init__(
        self,
        feature_names: List[str],
        states: List[str],
        rng: np.random.Generator,
        survey_year: str = "2018",
        horizon: str = "1-Year",
        survey: str = "person",
        size = None
    ) -> None:
        self.feature_names = feature_names
        self.states = states
        self.rng = rng
        self.survey_year = survey_year
        self.horizon = horizon
        self.survey = survey
        self.size = size

    def _download_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_source = ACSDataSource(
            survey_year=self.survey_year, horizon=self.horizon, survey=self.survey
        )
        acs_data = data_source.get_data(states=self.states, download=True)
        X, y, group = ACSEmployment.df_to_numpy(acs_data)
        A = 1 * (group == 1)

        X = X.astype(int)
        y = y.astype(int)

        # last feature is the group
        selected_idxs = []
        for idx, feature in enumerate(ACSEmployment.features):
            if feature in self.feature_names:
                selected_idxs.append(idx)

        X_selected = X[:, selected_idxs]

        obs = expit(
            X_selected[:, 0] - X_selected[:, 1]) > self.rng.uniform(size=X.shape[0])
        size = X.shape[0] if not self.size else self.size
        return X_selected[:size], A[:size], y[:size], obs[:size]

    def _get_ate_conditional_mean(self, A: np.ndarray, y: np.ndarray) -> float:
        """Computes the ate using inverse propensity weighting.

        Args:
            A (np.ndarray): vector of observed outcomes
            propensity_scores (_type_): Vector of propensity scores
            y (np.ndarray): response variable

        Returns
            ate: Average treatment effect.
        """
        conditional_mean = (y * A).sum()
        return conditional_mean / A.sum()

    def _create_dataframe(
        self, X: np.ndarray, A: np.ndarray, feature_names: List[str], full=True
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        data = pd.DataFrame()
        levels = []
        for column_idx in range(X.shape[1]):
            features = pd.get_dummies(X[:, column_idx])
            strata_number = features.shape[1]
            names = [
                str(feature_names[column_idx]) + "_" + str(j)
                for j in range(int(strata_number))
            ]
            if full:
                data[names] = 0
                data[names] = features
                levels.append(names)
            else:
                data[names[:-1]] = 0 
                data[names[:-1]] = features[features.columns[:-1]]
                levels.append(names[:-1])
        if full:
            data[["white", "non-white"]] = pd.get_dummies(A)
        else:
            data["white"] = pd.get_dummies(A)[0]
        return data, levels

    def load(self) -> Dataset:
        X, A, y, obs = self._download_data()

        X_sample = X[obs]
        A_sample = A[obs]
        y_sample = y[obs]

        # Compute conditional mean using sex feature
        empirical_conditional_mean = self._get_ate_conditional_mean(
            1 - X_sample[:, -1], y_sample
        )
        true_conditional_mean = self._get_ate_conditional_mean(1 - X[:, -1], y)

        population_df, levels = self._create_dataframe(X, A, self.feature_names, full=False)
        levels = [["white"]] + levels
        population_df["Creditability"] = y

        sample_df, _ = self._create_dataframe(X_sample, A_sample, self.feature_names, full=False)
        sample_df["Creditability"] = y_sample

        population_df_colinear, levels_colinear = self._create_dataframe(X, A, self.feature_names, full=True)
        levels_colinear = [["white", "non-white"]] + levels_colinear
        population_df_colinear["Creditability"] = y

        sample_df_colinear, _ = self._create_dataframe(X_sample, A_sample, self.feature_names, full=True)
        sample_df_colinear["Creditability"] = y_sample

        dataset = Dataset(
            population_df=population_df,
            sample_df=sample_df,
            population_df_colinear=population_df_colinear,
            sample_df_colinear=sample_df_colinear,
            levels=levels,
            levels_colinear=levels_colinear,
            target="Creditability",
            alternate_outcome="DEAR_1",
            empirical_conditional_mean=empirical_conditional_mean,
            true_conditional_mean=true_conditional_mean,
        )
        return dataset