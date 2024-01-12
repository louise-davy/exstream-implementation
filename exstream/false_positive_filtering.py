import stumpy
import pandas as pd
import numpy as np

# STEP 2 : FUNCTIONS FOR FALSE POSITIVES FILTERING


def false_positive_filter(
    ano_data: pd.DataFrame,
    refs: pd.DataFrame,
    false_positive_filtering: bool,
    max_distance: float,
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
    cols_for_this_ano = []
    nb_matches = []
    cols_to_visit = [col for col in ano_data.columns if col != "type_data"]
    if false_positive_filtering:
        for col in cols_to_visit:
            pattern = ano_data.loc[:, col]
            ts = refs.loc[:, col]
            matches = stumpy.match(pattern, ts, max_distance=max_distance)
            nb_matches.append(len(matches))
            if len(list(matches)) <= 5:
                if col not in cols_for_this_ano:
                    cols_for_this_ano.append(col)
        if not cols_for_this_ano:
            cols_for_this_ano.append(cols_to_visit[np.array(nb_matches).argmin()])

    else:
        cols_for_this_ano = cols_to_visit

    cols_for_this_ano.append("type_data")
    return cols_for_this_ano
