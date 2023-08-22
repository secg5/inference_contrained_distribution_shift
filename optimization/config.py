from typing import List, Tuple
from dataclasses import dataclass,  asdict

@dataclass
class Config:
    random_seed: str
    dataset_type: str
    correlation_coeff: int
    dataset_size: int
    feature_names: List
    states: List
    n_trials: int
    n_optim_iters: int
    restriction_trials: List
    matrix_types: List

def parse_config_dict(config_dict):
    config = Config(**config_dict)
    return config
      