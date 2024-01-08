import stumpy
import pandas as pd
import numpy as np

# STEP 2 : FUNCTIONS FOR FALSE POSITIVES FILTERING


def false_positive_filter(
    refs: pd.DataFrame, anos: pd.DataFrame, cols: list, false_positive_filtering: bool
) -> list:
    """
    Identify and remove false positive features.

    Parameters
    ----------
    refs : pd.DataFrame
        The data in the reference intervals.
    anos : pd.DataFrame
        The data in the anomaly intervals.
    cols : list
        The columns corresponding to the remaining features of the desired
        type of anomaly.

    Returns
    -------
    new_cols : list
        The columns corresponding to the remaining features of the desired
        type of anomaly.
    """
    if false_positive_filtering:
        new_cols = []
        refs_df = refs[cols]
        anos_df = anos[cols]
        cols_to_visit = list(anos_df.columns[:-4])
        for ano in anos.index.unique():
            cols_for_this_ano = []
            nb_matches = []
            for col in cols_to_visit:
                pattern = anos_df.loc[ano, col]
                ts = refs_df.loc[:, col]
                matches = stumpy.match(pattern, ts, max_distance=28.0)
                nb_matches.append(len(matches))
                if len(list(matches)) <= 1:
                    if col not in new_cols:
                        cols_for_this_ano.append(col)
                # else:
                # print(f"Found {len(matches)} match(es) for {col} in ano {ano}")
            if not cols_for_this_ano:
                new_cols.append(cols_to_visit[np.array(nb_matches).argmin()])
            else:
                new_cols.append(cols_for_this_ano)

    else:
        new_cols = []
        anos_df = anos[cols]
        cols_to_visit = list(anos_df.columns[:-4])
        for ano in anos.index.unique():
            cols_for_this_ano = []
            for col in cols_to_visit:
                cols_for_this_ano.append(col)
            new_cols.append(cols_for_this_ano)
    return new_cols


def assign_cols_per_ano(anos, new_filtered_features):
    anos["filtered_columns"] = None
    for i, ano in enumerate(anos.index.unique()):
        anos.loc[ano, "filtered_columns"] = str(new_filtered_features[i])
    anos["filtered_columns"] = anos["filtered_columns"].apply(
        lambda x: x.strip("][").split(", ")
    )
    return anos
