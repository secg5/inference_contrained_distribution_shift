import os

import numpy as np
import pandas as pd


def generate_plot_dro(
    base_path,
    timestamp,
    ax,
):
    plotting_df = pd.read_csv(
        os.path.join(base_path, timestamp, "plotting_df.csv"), low_memory=False
    )
    max_steps = plotting_df.groupby(["matrix_type", "restriction_type"]).max()

    masks = {
        ("Nx12", "DRO_worst_case"): (
            plotting_df["step"] == max_steps.loc[("Nx12", "DRO_worst_case"), "step"]
        )
        & (plotting_df["restriction_type"] == "DRO_worst_case"),
        ("Nx12", "DRO"): (plotting_df["step"] == max_steps.loc[("Nx12", "DRO"), "step"])
        & (plotting_df["restriction_type"] == "DRO"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_conditional_mean",
        true_field="true_conditional_mean",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Worst Case", "Omniscient"])
    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=20)
    ax.set_xlabel("DRO Benchmarks", fontsize=20)


def generate_theta_plots_1_2_3(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(
        os.path.join(base_path, timestamp, "plotting_df.csv"), low_memory=False
    )
    max_steps = plotting_df.groupby(["matrix_type", "restriction_type"]).max()

    masks = {
        ("Nx12", "count"): (
            plotting_df["step"] == max_steps.loc[("Nx12", "count"), "step"]
        )
        & (plotting_df["matrix_type"] == "Nx12")
        & (plotting_df["restriction_type"] == "count"),
        ("Nx8", "count"): (
            plotting_df["step"] == max_steps.loc[("Nx8", "count"), "step"]
        )
        & (plotting_df["matrix_type"] == "Nx8")
        & (plotting_df["restriction_type"] == "count"),
        ("Nx6", "count"): (
            plotting_df["step"] == max_steps.loc[("Nx6", "count"), "step"]
        )
        & (plotting_df["matrix_type"] == "Nx6")
        & (plotting_df["restriction_type"] == "count"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")
    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_conditional_mean",
        true_field="true_conditional_mean",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Experiment 1", "Experiment 2", "Experiment 3"])
    ax.set_xlabel("Parametric form of $\\theta(X)$", fontsize=20)


def generate_theta_plots_4_5_6(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(
        os.path.join(base_path, timestamp, "plotting_df.csv"), low_memory=False
    )
    max_steps = plotting_df.groupby(["restriction_type"]).max()

    masks = {
        "count_minus": (plotting_df["step"] == max_steps.loc["count_minus", "step"])
        & (plotting_df["restriction_type"] == "count_minus"),
        "count": (plotting_df["step"] == max_steps.loc["count", "step"])
        & (plotting_df["restriction_type"] == "count"),
        "count_plus": (plotting_df["step"] == max_steps.loc["count_plus", "step"])
        & (plotting_df["restriction_type"] == "count_plus"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_conditional_mean",
        true_field="true_conditional_mean",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Experiment 4", "Experiment 5", "Experiment 6"])
    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=20)


def generate_cov_plots(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))
    max_steps = plotting_df.groupby(["restriction_type"]).max()

    masks = {
        "cov_positive": (plotting_df["step"] == max_steps.loc["cov_positive", "step"])
        & (plotting_df["restriction_type"] == "cov_positive"),
        "count": (plotting_df["step"] == max_steps.loc["count", "step"])
        & (plotting_df["restriction_type"] == "count"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_conditional_mean",
        true_field="true_conditional_mean",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Covariance", "Count"])
    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=20)
    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=20)

    ax.legend(loc="lower right", fontsize=18)


def generate_theta_plots_1_2_regression(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))
    max_steps = plotting_df.groupby("matrix_type").max()

    masks = {
        "Nx12": (plotting_df["step"] == max_steps.loc["Nx12", "step"])
        & (plotting_df["matrix_type"] == "Nx12")
        & (plotting_df["restriction_type"] == "count"),
        "Nx6": (plotting_df["step"] == max_steps.loc["Nx6", "step"])
        & (plotting_df["matrix_type"] == "Nx6")
        & (plotting_df["restriction_type"] == "count"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_empirical_coef",
        true_field="true_coef",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Experiment 1", "Experiment 2"])
    ax.set_ylabel("Estimated Coefficient", fontsize=20)
    ax.set_xlabel("Parametric form of $\\theta(X)$", fontsize=20)
    ax.legend(loc="lower right", fontsize=18)


def generate_theta_plots_3_4_regression(base_path: str, timestamp: str, ax):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))
    max_steps = plotting_df.groupby("restriction_type").max()

    masks = {
        "count": (plotting_df["step"] == max_steps.loc["count", "step"])
        & (plotting_df["restriction_type"] == "count"),
        "count_plus": (plotting_df["step"] == max_steps.loc["count_plus", "step"])
        & (plotting_df["restriction_type"] == "count_plus"),
    }

    plot_intervals(ax=ax, masks=masks, plotting_df=plotting_df, color="C0")

    plot_references(
        ax=ax,
        plotting_df=plotting_df,
        empirical_field="empirical_empirical_coef",
        true_field="true_coef",
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Experiment 3", "Experiment 4"])
    ax.set_xlabel("Number of constraints in $\\theta(X)$", fontsize=20)


def plot_intervals(ax, masks: dict, plotting_df: pd.DataFrame, color="C0"):
    experiment_bounds = {}
    for name, mask in masks.items():
        experiment_bounds[name] = {
            "max": plotting_df[mask]["max_bound"],
            "min": plotting_df[mask]["min_bound"],
        }
    for idx, (_, bounds) in enumerate(experiment_bounds.items(), start=1):
        ax.vlines(
            idx,
            ymin=np.median(bounds["min"]),
            ymax=np.median(bounds["max"]),
            color=color,
            linewidth=2,
        )

        for limit in ["max", "min"]:
            quantiles = np.quantile(bounds[limit], [0.05, 0.95])

            ax.fill_between(
                [idx - 0.1, idx + 0.1],
                [quantiles[0]] * 2,
                [quantiles[1]] * 2,
                color=color,
                alpha=0.1,
                hatch="//",
            )

            ax.plot(
                [idx - 0.05, idx + 0.05],
                [np.median(bounds[limit])] * 2,
                color=color,
                linewidth=2,
            )


def plot_references(
    ax, plotting_df: pd.DataFrame, true_field: str, empirical_field: str
):
    true_coef = plotting_df[true_field].mean()
    min_empirical_coef = plotting_df[empirical_field].min()
    max_empirical_coef = plotting_df[empirical_field].max()
    ax.axhline(
        y=true_coef,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=2,
    )

    min_x, max_x = ax.get_xlim()
    ax.fill_between(
        [min_x - 0.2, max_x + 0.2],
        [min_empirical_coef] * 2,
        [max_empirical_coef] * 2,
        color="olive",
        alpha=0.2,
        label="Naive estimator",
        hatch="//",
    )

    ax.set_xlim(min_x - 0.2, max_x + 0.2)
