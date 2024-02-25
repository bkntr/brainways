import logging
from typing import List, Tuple

import pandas as pd
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import multipletests


def calculate_contrast(
    results_df: pd.DataFrame,
    condition_col: str,
    values_col: str,
    posthoc_comparisons: List[Tuple[str, str]],
    min_group_size: int,
    pvalue: float,
    multiple_comparisons_method: str = "fdr_bh",
):
    results_df = filter_df_areas(results_df)
    anova_df = calculate_anova(
        df=results_df,
        condition_col=condition_col,
        values_col=values_col,
        min_group_size=min_group_size,
        min_anova_pvalue=pvalue,
        multiple_comparisons_method=multiple_comparisons_method,
    )
    posthoc_df = calculate_posthoc(
        df=results_df,
        anova_df=anova_df,
        condition_col=condition_col,
        values_col=values_col,
        posthoc_comparisons=posthoc_comparisons,
        multiple_comparisons_method=multiple_comparisons_method,
    )
    return anova_df, posthoc_df


def calculate_anova(
    df: pd.DataFrame,
    condition_col: str,
    values_col: str,
    min_group_size: int,
    min_anova_pvalue: float,
    multiple_comparisons_method: str = "fdr_bh",
):
    df = df.rename(columns={condition_col: "__condition__", values_col: "__values__"})
    num_conditions = len(df["__condition__"].unique())

    anova_df = pd.DataFrame(columns=["F", "p"])
    for struct in df["acronym"].unique():
        struct_df = df[df["acronym"] == struct]
        condition_counts = struct_df.groupby("__condition__")["__values__"].count()
        # only do anova comparison for groups containing a minimum of `min_group_size` samples from each condition
        if len(condition_counts) < num_conditions:
            logging.warning(
                f"Skipping structure {struct} due to having too few conditions"
                f" {condition_counts.index} ({len(condition_counts)}<{num_conditions})"
            )
            continue
        if min(condition_counts) < min_group_size:
            logging.warning(
                f"Skipping structure {struct} due to having too few values in a"
                " condition"
                f" {condition_counts} ({min(condition_counts)}<{min_group_size})"
            )
            continue
        lm = ols(
            "__values__ ~ C(__condition__)", df, subset=df["acronym"] == struct
        ).fit()
        struct_anova = sm.stats.anova_lm(lm, typ=2)
        struct_anova = struct_anova.rename(columns={"PR(>F)": "p"})
        anova_df.loc[struct] = struct_anova.loc["C(__condition__)", ["F", "p"]]

    if multiple_comparisons_method:
        reject, pvals_corrected, _, _ = multipletests(
            anova_df["p"].values.reshape(-1),
            alpha=min_anova_pvalue,
            method=multiple_comparisons_method,
        )
    else:
        pvals_corrected = anova_df["p"].values
    anova_df["p_corrected"] = pvals_corrected
    anova_df["reject"] = reject

    return anova_df


def calculate_posthoc(
    df: pd.DataFrame,
    anova_df: pd.DataFrame,
    condition_col: str,
    values_col: str,
    posthoc_comparisons: List[Tuple[str, str]],
    multiple_comparisons_method: str = "fdr_bh",
):
    columns = ["-".join(comparison) for comparison in posthoc_comparisons]
    posthoc_p_df = pd.DataFrame(columns=columns)

    if not anova_df["reject"].any():
        logging.warning(
            "No regions rejected the null hypothesis, nothing to post-hoc on"
        )
        return posthoc_p_df

    for struct in anova_df[anova_df["reject"]].index:
        struct_df = df[df["acronym"] == struct]
        posthoc_pvalues = sp.posthoc_ttest(
            struct_df,
            val_col=values_col,
            group_col=condition_col,
            p_adjust=None,
        )
        posthoc_p_df.loc[struct] = {
            col: posthoc_pvalues.loc[comp[0], comp[1]]
            for col, comp in zip(columns, posthoc_comparisons)
        }

    if multiple_comparisons_method:
        reject, pvals_corrected, _, _ = multipletests(
            posthoc_p_df.values.reshape(-1), method=multiple_comparisons_method
        )
        pvals_corrected = pvals_corrected.reshape(posthoc_p_df.values.shape)
    else:
        pvals_corrected = posthoc_p_df.values

    posthoc_corrected_df = pd.DataFrame(
        data=pvals_corrected, index=posthoc_p_df.index, columns=columns
    )
    return posthoc_corrected_df


def filter_df_areas(df):
    return df[~df["is_parent_structure"] & df["is_gray_matter"]]
