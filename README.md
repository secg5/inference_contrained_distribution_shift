[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)
# Statistical Inference Under Constrained Selection Bias


## Replication Instructions

Run the right script with the right config file to replicate each experiment. All the required elements are in the `optimization` directory.

For example, to replicate the Omni DRO experiment on the synthetic dataset, run:

```bash
python inference.py experiments_metadata/sim_dro_omni.json
```

Each script will generate a folder with the results of the experiment. The folder will be named with the date and time of the experiment. You should use the timestamp as input for the functions in the `plotting/paper_plots.ipynb` notebook.

The following tables specify the script and config file for each experiment.

### Synthetic Dataset


| Experiment       | Script       | Config File          |
| ---------------- | ------------ | -------------------- |
| Omni DRO         | inference.py | sim_dro_omni         |
| Constraints Ours | inference.py | sim_constraints_ours |
| Constraints DRO  | inference.py | sim_constraints_dro  |
| Theta Ours       | inference.py | sim_theta_ours       |
| Theta DRO        | inference.py | sim_theta_dro        |


### Semi-Synthetic Dataset

| Experiment       | Script       | Config File                 |
| ---------------- | ------------ | --------------------------- |
| Omni DRO         | inference.py | semi-synth_dro_omni         |
| Constraints Ours | inference.py | semi-synth_constraints_ours |
| Constraints DRO  | inference.py | semi-synth_constraints_dro  |
| Theta Ours       | inference.py | semi-synth_theta_ours       |
| Theta DRO        | inference.py | semi-synth_theta_dro        |

### Regression

| Experiment       | October                      | Config File                 |
| ---------------- | ---------------------------- | --------------------------- |
| Omni DRO         | inference_non_closed_form.py | regression_dro_omni         |
| Constraints Ours | inference_non_closed_form.py | regression_constraints_ours |
| Constraints DRO  | inference_non_closed_form.py | regression_constraints_dro  |
| Theta Ours       | inference_non_closed_form.py | regression_theta_ours       |
| Theta DRO        | inference_non_closed_form.py | regression_theta_dro        |

### Logistic Regression

| Experiment | October                      | Config File |
| ---------- | ---------------------------- | ----------- |
| Theta Ours | inference_non_closed_form.py | logistic    |

### Covariance

| Experiment | October      | Config File |
| ---------- | ------------ | ----------- |
| Theta Ours | inference.py | covariance  |
