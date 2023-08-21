# Dataset config
# random_seed = 1234
# dataset_type = "simulation"
# correlation_coeff = 1
# dataset_size = 100_000
# feature_names = [
#     "MIL",
#     "ANC",
#     "NATIVITY",
#     "DEAR",
#     "DEYE",
#     "DREM",
#     "SEX",
# ]
# states = ["AL"]

# # Inference config
# n_trials = 1
# n_optim_iters = 3_000
# restriction_trials = ["count"]
# matrix_types = ["Nx12", "Nx8", "Nx6"]

random_seed = 1234
dataset_type = "folktables"
correlation_coeff = 1
dataset_size = 100_000
feature_names = [
    "MIL",
    "ANC",
    "NATIVITY",
    "DEAR",
    "DEYE",
    "DREM",
    "SEX",
]
states = ["AL"]

# Inference config
n_trials = 1
n_optim_iters = 3_000
restriction_trials = ["count"] #"count_plus", "count_minus", "DRO"
matrix_types = ["Nx6"] # "Nx12"
