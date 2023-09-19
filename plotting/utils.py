import os
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../optimization")
from datasets import *


def generate_theta_plots_1_2_3(base_path: str, timestamp: str):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))
    with open(os.path.join(base_path, timestamp, "dataset_metadata.pkl"), "rb") as fp:
        dataset = pickle.load(fp)

    max_vals_1 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["matrix_type"] == "Nx12")
        & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_2 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["matrix_type"] == "Nx8")
        & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_3 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["matrix_type"] == "Nx6")
        & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_4 = plotting_df[
        (plotting_df["step"] == 2999) & (plotting_df["restriction_type"] == "DRO")
    ]["max_bound"]

    min_vals_1 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["matrix_type"] == "Nx12")
        & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_2 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["matrix_type"] == "Nx8")
        & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_3 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["matrix_type"] == "Nx8")
        & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_4 = plotting_df[
        (plotting_df["step"] == 2999) & (plotting_df["restriction_type"] == "DRO")
    ]["min_bound"]

    fig, ax = plt.subplots()

    ax.axhline(
        y=dataset.true_conditional_mean,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=4,
        alpha=0.5,
    )
    ax.axhline(
        y=dataset.empirical_conditional_mean,
        color="olive",
        linestyle=":",
        label="Naive estimator",
        linewidth=4,
        alpha=0.5,
    )

    ax.boxplot([max_vals_1, min_vals_1], positions=[1, 1], widths=0.2, showfliers=False)
    plt.vlines(1, ymin=min(min_vals_1), ymax=max(max_vals_1))

    ax.boxplot([max_vals_2, min_vals_2], positions=[2, 2], widths=0.2, showfliers=False)
    plt.vlines(2, ymin=min(min_vals_2), ymax=max(max_vals_2))

    ax.boxplot([max_vals_3, min_vals_3], positions=[3, 3], widths=0.2, showfliers=False)
    plt.vlines(3, ymin=min(min_vals_3), ymax=max(max_vals_3))

    ax.boxplot([max_vals_4, min_vals_4], positions=[4, 4], widths=0.2, showfliers=False)
    plt.vlines(4, ymin=min(min_vals_4), ymax=max(max_vals_4))

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Experiment 1", "Experiment 2", "Experiment 3", "DRO"])

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=20)
    ax.set_xlabel("Parametric form of  $\\theta(X)$", fontsize=20)

    # # Add a legend
    ax.legend(loc="lower right", fontsize=18)

    if not os.path.exists(timestamp):
        os.mkdir(timestamp)
    plt.savefig(os.path.join(timestamp, "theta_plots_1_2_3.png"), bbox_inches="tight")


def generate_theta_plots_4_5_6(base_path: str, timestamp: str):
    plotting_df = pd.read_csv(os.path.join(base_path, timestamp, "plotting_df.csv"))
    with open(os.path.join(base_path, timestamp, "dataset_metadata.pkl"), "rb") as fp:
        dataset = pickle.load(fp)

    max_vals_1 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["restriction_type"] == "count_minus")
    ]["max_bound"]
    max_vals_2 = plotting_df[
        (plotting_df["step"] == 2999) & (plotting_df["restriction_type"] == "count")
    ]["max_bound"]
    max_vals_3 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["restriction_type"] == "count_plus")
    ]["max_bound"]

    min_vals_1 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["restriction_type"] == "count_minus")
    ]["min_bound"]
    min_vals_2 = plotting_df[
        (plotting_df["step"] == 2999) & (plotting_df["restriction_type"] == "count")
    ]["min_bound"]
    min_vals_3 = plotting_df[
        (plotting_df["step"] == 2999)
        & (plotting_df["restriction_type"] == "count_plus")
    ]["min_bound"]

    fig, ax = plt.subplots()

    ax.axhline(
        y=dataset.true_conditional_mean,
        color="blue",
        linestyle="dashed",
        label="True value",
        linewidth=4,
        alpha=0.5,
    )
    ax.axhline(
        y=dataset.empirical_conditional_mean,
        color="olive",
        linestyle=":",
        label="Naive estimator",
        linewidth=4,
        alpha=0.5,
    )

    ax.boxplot([max_vals_1, min_vals_1], positions=[1, 1], widths=0.2, showfliers=False)
    plt.vlines(1, ymin=min(min_vals_1), ymax=max(max_vals_1))

    ax.boxplot([max_vals_2, min_vals_2], positions=[2, 2], widths=0.2, showfliers=False)
    plt.vlines(2, ymin=min(min_vals_2), ymax=max(max_vals_2))

    ax.boxplot([max_vals_3, min_vals_3], positions=[3, 3], widths=0.2, showfliers=False)
    plt.vlines(3, ymin=min(min_vals_3), ymax=max(max_vals_3))

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Experiment 3", "Experiment 4", "Experiment 5"])

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    ax.set_ylabel("Conditional Mean $\widehat{\mathbb{E}}[Y|A=1]$", fontsize=20)
    ax.set_xlabel("Number of constraints in  $\\theta(X)$", fontsize=20)

    # # Add a legend
    ax.legend(loc="lower right", fontsize=18)

    if not os.path.exists(timestamp):
        os.mkdir(timestamp)
    plt.savefig(os.path.join(timestamp, "theta_plots_4_5_6.png"), bbox_inches="tight")
