import pandas as pd

from brainways.utils.contrast import calculate_contrast


def test_contrast():
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
    anova, posthoc = calculate_contrast(
        results_df=results_df,
        condition_col="cond",
        values_col="val",
        posthoc_comparisons=[("a", "b")],
        min_group_size=3,
        pvalue=1.0,
    )
    print(anova)
    print(posthoc)
