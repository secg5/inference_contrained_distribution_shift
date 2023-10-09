import os

import pandas as pd

STEP = 3499
STEP_TWO = 4999


def generate_theta_plots_1_2_3(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))

    max_vals_1 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["matrix_type"] == "Nx12")
        & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_2 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["matrix_type"] == "Nx8")
        & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_3 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["matrix_type"] == "Nx6")
        & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_4 = plotting_df[
        (plotting_df["step"] == STEP) & (plotting_df["restriction_type"] == "DRO")
    ]["max_bound"]
    max_vals_5 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["restriction_type"] == "DRO_worst_case")
    ]["max_bound"]

    min_vals_1 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["matrix_type"] == "Nx12")
        & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_2 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["matrix_type"] == "Nx8")
        & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_3 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["matrix_type"] == "Nx8")
        & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_4 = plotting_df[
        (plotting_df["step"] == STEP) & (plotting_df["restriction_type"] == "DRO")
    ]["min_bound"]
    min_vals_5 = plotting_df[
        (plotting_df["step"] == STEP)
        & (plotting_df["restriction_type"] == "DRO_worst_case")
    ]["min_bound"]

    # Experiment 1
    boxplot = ax.boxplot(
        [max_vals_1, min_vals_1],
        positions=[1, 1],
        widths=0.2,
        showfliers=False,
        showmeans=False,
        showbox=False,
        showcaps=False,
    )
    ax.vlines(
        1,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Experiment 2
    boxplot = ax.boxplot(
        [max_vals_2, min_vals_2],
        positions=[2, 2],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showcaps=False,
        showmeans=False,
    )
    ax.vlines(
        2,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Experiment 3
    boxplot = ax.boxplot(
        [max_vals_3, min_vals_3],
        positions=[3, 3],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    ax.vlines(
        3,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [2.9, 3.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [2.9, 3.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Experiment 4
    boxplot = ax.boxplot(
        [max_vals_4, min_vals_4],
        positions=[4, 4],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showcaps=False,
    )
    ax.vlines(
        4,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [3.9, 4.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [3.9, 4.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Experiment 5
    boxplot = ax.boxplot(
        [max_vals_5, min_vals_5],
        positions=[5, 5],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showcaps=False,
    )
    ax.vlines(
        5,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [4.9, 5.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [4.9, 5.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    true_conditional_mean = plotting_df["true_conditional_mean"].mean()
    min_empirical_conditional_mean = plotting_df["empirical_conditional_mean"].min()
    max_empirical_conditional_mean = plotting_df["empirical_conditional_mean"].max()

    ax.axhline(
        y=true_conditional_mean,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=2,
    )

    # Get x axis limits
    min_x, max_x = ax.get_xlim()
    ax.fill_between(
        [min_x, 2, max_x],
        [min_empirical_conditional_mean] * 3,
        [max_empirical_conditional_mean] * 3,
        color="olive",
        alpha=0.2,
        label="Naive estimator",
        hatch="//",
    )

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(
        ["Experiment 1", "Experiment 2", "Experiment 3", "DRO", "DRO (Worst Case)"]
    )

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=20)
    ax.set_xlabel("Parametric form of $\\theta(X)$", fontsize=20)


def generate_theta_plots_4_5_6(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))

    max_vals_1 = plotting_df[
        (plotting_df["step"] == STEP_TWO)
        & (plotting_df["restriction_type"] == "count_minus")
    ]["max_bound"]
    max_vals_2 = plotting_df[
        (plotting_df["step"] == STEP_TWO) & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_3 = plotting_df[
        (plotting_df["step"] == STEP_TWO)
        & (plotting_df["restriction_type"] == "count_plus")
    ]["max_bound"]

    min_vals_1 = plotting_df[
        (plotting_df["step"] == STEP_TWO)
        & (plotting_df["restriction_type"] == "count_minus")
    ]["min_bound"]
    min_vals_2 = plotting_df[
        (plotting_df["step"] == STEP_TWO) & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_3 = plotting_df[
        (plotting_df["step"] == STEP_TWO)
        & (plotting_df["restriction_type"] == "count_plus")
    ]["min_bound"]
    # Experiment 4
    boxplot = ax.boxplot(
        [max_vals_1, min_vals_1],
        positions=[1, 1],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    # ax.vlines(1, ymin=min(min_vals_1), ymax=max(max_vals_1))
    ax.vlines(
        1,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Experiment 5
    boxplot = ax.boxplot(
        [max_vals_2, min_vals_2],
        positions=[2, 2],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    # ax.vlines(2, ymin=min(min_vals_2), ymax=max(max_vals_2))
    ax.vlines(
        2,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Experiment 6
    boxplot = ax.boxplot(
        [max_vals_3, min_vals_3],
        positions=[3, 3],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    # ax.vlines(3, ymin=min(min_vals_3), ymax=max(max_vals_3))
    ax.vlines(
        3,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [2.9, 3.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [2.9, 3.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    true_conditional_mean = plotting_df["true_conditional_mean"].mean()
    min_empirical_conditional_mean = plotting_df["empirical_conditional_mean"].min()
    max_empirical_conditional_mean = plotting_df["empirical_conditional_mean"].max()
    ax.axhline(
        y=true_conditional_mean,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=2,
    )
    # Get x axis limits
    min_x, max_x = ax.get_xlim()
    ax.fill_between(
        [min_x, max_x],
        [min_empirical_conditional_mean] * 2,
        [max_empirical_conditional_mean] * 2,
        color="olive",
        alpha=0.2,
        label="Naive estimator",
        hatch="//",
    )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Experiment 4", "Experiment 5", "Experiment 6"])

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=20)


def generate_cov_plots(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))

    max_vals_cov = plotting_df[
        (plotting_df["step"] == STEP_TWO)
        & (plotting_df["restriction_type"] == "cov_positive")
    ]["max_bound"]
    max_vals_count = plotting_df[
        (plotting_df["step"] == STEP_TWO) & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]

    min_vals_cov = plotting_df[
        (plotting_df["step"] == STEP_TWO)
        & (plotting_df["restriction_type"] == "cov_positive")
    ]["min_bound"]
    min_vals_count = plotting_df[
        (plotting_df["step"] == STEP_TWO) & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]

    # Cov
    boxplot = ax.boxplot(
        [max_vals_cov, min_vals_cov],
        positions=[1, 1],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    ax.vlines(
        1,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Count
    boxplot = ax.boxplot(
        [max_vals_count, min_vals_count],
        positions=[2, 2],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    ax.vlines(
        2,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    true_conditional_mean = plotting_df["true_conditional_mean"].mean()
    min_empirical_conditional_mean = plotting_df["empirical_conditional_mean"].min()
    max_empirical_conditional_mean = plotting_df["empirical_conditional_mean"].max()
    ax.axhline(
        y=true_conditional_mean,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=2,
    )
    # Get x axis limits
    min_x, max_x = ax.get_xlim()
    ax.fill_between(
        [min_x, max_x],
        [min_empirical_conditional_mean] * 2,
        [max_empirical_conditional_mean] * 2,
        color="olive",
        alpha=0.2,
        label="Naive estimator",
        hatch="//",
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Covariance", "Count"])

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=20)
    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=20)

    # # Add a legend
    ax.legend(loc="lower right", fontsize=18)


def generate_regression_plots(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))

    max_vals_count_plus = plotting_df[
        (plotting_df["step"] == 999) & (plotting_df["restriction_type"] == "count_plus")
    ]["max_bound"]
    max_vals_count = plotting_df[
        (plotting_df["step"] == 999) & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]

    min_vals_count_plus = plotting_df[
        (plotting_df["step"] == 999) & (plotting_df["restriction_type"] == "count_plus")
    ]["min_bound"]
    min_vals_count = plotting_df[
        (plotting_df["step"] == 999) & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]

    # Count
    boxplot = ax.boxplot(
        [max_vals_count, min_vals_count],
        positions=[1, 1],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    ax.vlines(
        1,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [0.9, 1.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    # Count Plus
    boxplot = ax.boxplot(
        [max_vals_count_plus, min_vals_count_plus],
        positions=[2, 2],
        widths=0.2,
        showfliers=False,
        showbox=False,
        showmeans=False,
        showcaps=False,
    )
    ax.vlines(
        2,
        ymin=boxplot["whiskers"][0].get_ydata()[1],
        ymax=boxplot["whiskers"][3].get_ydata()[1],
    )

    # Fill between boxplot whiskers
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][0].get_ydata()[1]] * 2,
        [boxplot["whiskers"][1].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )
    ax.fill_between(
        [1.9, 2.1],
        [boxplot["whiskers"][2].get_ydata()[1]] * 2,
        [boxplot["whiskers"][3].get_ydata()[1]] * 2,
        color="C0",
        alpha=0.5,
        hatch="//",
    )

    true_coef_mean = plotting_df["true_coef"].mean()
    min_empirical_coef = plotting_df["empirical_empirical_coef"].min()
    max_empirical_coef = plotting_df["empirical_empirical_coef"].max()
    ax.axhline(
        y=true_coef_mean,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=2,
    )
    # Get x axis limits
    min_x, max_x = ax.get_xlim()
    ax.fill_between(
        [min_x, max_x],
        [min_empirical_coef] * 2,
        [max_empirical_coef] * 2,
        color="olive",
        alpha=0.2,
        label="Naive estimator",
        hatch="//",
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Experiment 1", "Experimnent 2"])

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    ax.set_ylabel("Estimated Coefficient", fontsize=20)
    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=20)

    # # # Add a legend
    # ax.legend(loc="lower right", fontsize=18)
