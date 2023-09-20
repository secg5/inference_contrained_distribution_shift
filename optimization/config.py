from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    random_seeds: List[int]
    dataset_type: str
    correlation_coeff: int
    dataset_size: int
    feature_names: List
    states: List
    n_trials: int
    n_optim_iters: int
    restriction_trials: List
    matrix_types: List
    statistic: str
    n_cov_pairs: Optional[int] = None


def parse_config_dict(config_dict):
    config = Config(**config_dict)
    return config
