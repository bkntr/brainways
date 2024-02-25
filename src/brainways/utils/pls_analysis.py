from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

import pyls
from pyls import PLSResults


def pls_analysis(
    results_df_pls: pd.DataFrame,
    condition_col: str,
    n_perm: int = 1000,
    n_boot: int = 1000,
) -> PLSResults:

    num_groups = (
        results_df_pls.index.get_level_values(condition_col)
        .value_counts(sort=False)
        .values
    )
    pls_results = pyls.meancentered_pls(
        X=results_df_pls.values, groups=num_groups, n_perm=n_perm, n_boot=n_boot
    )

    return pls_results


def get_results_df_for_pls(
    results_df: pd.DataFrame,
    values: str,
    condition_col: str,
    min_per_group: int,
    conditions: Optional[List[str]] = None,
) -> pd.DataFrame:
    if conditions:
        results_df = results_df[results_df[condition_col].isin(conditions)]
    results_df = results_df[
        results_df["is_gray_matter"] & ~results_df["is_parent_structure"]
    ]
    results_df_pls = results_df.pivot(
        index=["animal_id", condition_col], columns="acronym", values=values
    )
    results_df_pls = results_df_pls.sort_index(level=condition_col)
    results_df_pls = remove_columns_lacking_data(
        df=results_df_pls, condition=condition_col, min_per_group=min_per_group
    )
    results_df_pls = interpolate_by_cond(df=results_df_pls, condition=condition_col)
    return results_df_pls


def interpolate_by_cond(df, condition: str):
    cond_means = df.groupby(condition).mean()
    for cond, means in cond_means.iterrows():
        cond_slice = pd.IndexSlice[:, cond]
        df.loc[cond_slice, :] = df.loc[cond_slice, :].fillna(means)
    return df


def remove_columns_lacking_data(df, condition: str, min_per_group: int):
    cond_counts = df.groupby(condition).count()
    columns_with_enough_data = (cond_counts >= min_per_group).all(axis=0)
    return df.loc[:, columns_with_enough_data]


def get_estimated_lv_plot(
    pls_results: PLSResults, results_df_pls: pd.DataFrame, condition: str, lv: int = 0
) -> pd.DataFrame:
    lv = 0
    group_labels = results_df_pls.index.unique(condition).values

    estimate = pls_results.bootres.contrast[:, lv]
    ul = pls_results.bootres.contrast_ci[:, lv, 0]
    ll = pls_results.bootres.contrast_ci[:, lv, 1]
    stderr = np.abs(ll - ul) / 2

    return pd.DataFrame(
        {
            f"Estimate LV{lv+1}": estimate,
            "stderr": stderr,
            "Group": group_labels,
        }
    )


def save_estimated_lv_plot(path: Path, plot_df: pd.DataFrame):
    value_column = plot_df.columns[0]
    ax = sns.barplot(
        x="Group",
        y=value_column,
        data=plot_df,
        capsize=0.1,
        err_kws={"linewidth": 1.25},
        alpha=0.25,
        errorbar=None,
        color="k",
    )

    x = np.array([r.get_x() for r in ax.patches])
    w = np.array([r.get_width() for r in ax.patches])
    patch_x_center = x + w * 0.5
    sort_idxs = np.argsort(x)
    ax.errorbar(
        x=patch_x_center[sort_idxs],
        y=plot_df[value_column][sort_idxs],
        fmt="none",
        yerr=plot_df["stderr"][sort_idxs],
        ecolor="black",
    )

    plt.savefig(path)
    plt.close()


def get_lv_p_values_plot(pls_results: PLSResults) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LV": list(range(1, len(pls_results.permres.pvals) + 1)),
            "P Value": pls_results.permres.pvals,
        }
    )


def save_lv_p_values_plot(path: Path, plot_df: pd.DataFrame, alpha: float = 0.05):
    ax = sns.barplot(
        x="LV",
        y="P Value",
        data=plot_df,
        color="k",
        capsize=0.1,
        err_kws={"linewidth": 1.25},
        alpha=0.25,
        errorbar=None,
    )
    p_title = plot_df["P Value"].round(5).tolist()
    ax.axhline(alpha, color="k", alpha=0.5)
    ax.set_title(f"LV P Values {p_title}")

    plt.savefig(path)
    plt.close()


def get_salience_plot(
    pls_results: PLSResults, results_df_pls: pd.DataFrame, lv: int = 0
):
    return pd.DataFrame(
        {
            "Structure": results_df_pls.columns,
            "Salience": pls_results.bootres.x_weights_normed[:, lv],
            "stderr": pls_results.bootres.x_weights_stderr[:, lv],
        }
    )


def save_salience_plot(path: Path, plot_df: pd.DataFrame, alpha: float = 0.05):
    fig_w = len(plot_df) / 6
    fig_h = fig_w / 2
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.barplot(
        x="Structure",
        y="Salience",
        data=plot_df,
        capsize=0.1,
        err_kws={"linewidth": 1.25},
        errorbar=None,
    )
    ax.tick_params(axis="x", rotation=90)
    alpha_z = scipy.stats.norm.ppf(1 - alpha / 2)
    if (plot_df["Salience"] > alpha_z).any():
        ax.axhline(alpha_z, color="k", alpha=0.5)
    if (plot_df["Salience"] < -alpha_z).any():
        ax.axhline(-alpha_z, color="k", alpha=0.5)
    _ = ax.set_title("PLS Salience Plot")

    x = np.array([r.get_x() for r in ax.patches])
    w = np.array([r.get_width() for r in ax.patches])
    patch_x_center = x + w * 0.5
    sort_idxs = np.argsort(x)
    ax.errorbar(
        x=patch_x_center[sort_idxs],
        y=plot_df["Salience"][sort_idxs],
        fmt="none",
        yerr=plot_df["stderr"][sort_idxs],
        ecolor="black",
    )

    ymin, ymax = ax.get_ylim()
    ax.vlines(
        patch_x_center, ymin=ymin, ymax=0, color="k", linestyle="--", linewidth=0.5
    )
    ax.set_ylim(ymin, ymax)

    plt.savefig(path)
    plt.close()
