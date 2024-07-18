import pandas as pd

from brainways.utils.pls_analysis import get_results_df_for_pls, pls_analysis


def test_pls_analysis():
    results_df = pd.DataFrame(
        [
            {
                "cond": condition,
                "animal_id": animal_id,
                "is_parent_structure": False,
                "is_gray_matter": True,
                "acronym": "a",
                "val": cells,
            }
            for condition, animal_id, cells in zip(
                ["a", "a", "a", "b", "b", "b"],
                ["a", "b", "c", "d", "e", "f"],
                [1, 2, 3, 4, 5, 6],
            )
        ]
    )
    results_df_pls = get_results_df_for_pls(
        results_df, values="val", condition_col="cond", min_per_group=3
    )
    pls_results = pls_analysis(
        results_df_pls=results_df_pls,
        condition_col="cond",
        n_perm=10,
        n_boot=10,
    )
    print(pls_results)
