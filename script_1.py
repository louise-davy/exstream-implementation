import numpy as np
import pandas as pd
import logging

from exstream.correlation_filtering import correlated_features_filter
from exstream.false_positive_filtering import false_positive_filter
from exstream.entropy_based_single_reward_feature import (
    entropy_based_single_feature_reward,
    reward_leap_filter,
)
from utils.get_data import get_train_test_data, split_references_and_anomalies

logging.getLogger("numba").setLevel(logging.WARNING)


def compute_explanatory_features(distances: dict) -> dict:
    # col = [s.replace("'", "") for s in col]
    # selected_distances = {
    #     feat: dist for feat, dist in distances.items() if feat in anos.columns
    # }
    # print(selected_distances)
    if len(distances) > 1:
        filtered_features = reward_leap_filter(distances)
        selected_features = filtered_features
    else:
        selected_features = list(distances.keys())

    return selected_features


# NOTEBOOK STOPED HERE


def get_explanatory_features(
    refs: pd.DataFrame,
    anos: pd.DataFrame,
    cluster: bool,
    correlation_threshold: float,
    false_positive_filtering: bool,
    max_distance: float,
):
    # CORRELATION FILTERING
    all_data = pd.concat([refs, anos])

    logging.info("Filtering correlated features...")
    filtered_features = correlated_features_filter(
        all_data, correlation_threshold=correlation_threshold, cluster=cluster
    )
    logging.debug(f"Features after correlation filtering: {filtered_features}")
    logging.info(
        f"Dropped {len(all_data.columns[:-3]) - len(filtered_features)} features after"
        "correlation filtering"
    )
    refs = refs.loc[:, filtered_features]
    anos = anos.loc[:, filtered_features]

    explanatory_features = {}

    for ano in anos.index.unique():
        logging.info(f"Anomaly {ano}")
        ano_data = anos.loc[ano]
        ano_ref = refs.loc[ano]

        # FALSE POSITIVE FILTERING
        new_filtered_features = false_positive_filter(
            ano_ref, refs, false_positive_filtering, max_distance=max_distance
        )
        logging.debug(
            f"Features after false positive filtering: {new_filtered_features}"
        )
        logging.info(
            f"Dropped {len(filtered_features) - len(new_filtered_features)} features"
            "after false positive filtering"
        )

        ano_data = ano_data.loc[:, new_filtered_features]
        ano_ref = ano_ref.loc[:, new_filtered_features]
        ano_all = pd.concat([ano_ref, ano_data], axis=0)

        # ENTROPY BASED SINGLE FEATURE REWARD
        distance = entropy_based_single_feature_reward(ano_ref, ano_data, ano_all)

        final_features = compute_explanatory_features(distance)

        logging.debug(
            f"Features after entropy based single feature reward: {final_features}"
        )
        logging.info(
            f"Dropped {len(new_filtered_features) - len(final_features)} features after"
            "entropy based single feature reward"
        )

        explanatory_features[ano] = final_features

    return explanatory_features


def get_features_integer_indice(features: list, anomalies: pd.DataFrame):
    """
    Returns the indices of the specified features in the anomalies DataFrame.

    Parameters:
    - features (list): A list of feature names.
    - anomalies (pd.DataFrame): A DataFrame containing the anomalies.

    Returns:
    - indices (list): A list of integer indices corresponding to the features in the
    anomalies DataFrame.
    """
    indices = []
    for feature in features:
        indice = anomalies.columns.get_loc(feature)
        indices.append(indice)

    return indices


def construct_explanations(
    data_folder: str,
    label_filename: str,
    cluster: bool,
    correlation_threshold: float,
    false_positive_filtering: bool,
    max_distance: float,
    verbose: bool,
):
    """
    Constructs explanations for each label in the provided DataFrame.

    Args:
        labels (pd.DataFrame): DataFrame containing the labels with columns 'trace_id',
        'ano_id', and 'ano_type'.
        datafolder (str): Path to the data folder.
        label_filename (str): Filename of the label file.

    Returns:
        pd.DataFrame: DataFrame containing the constructed explanations with columns
        'trace_id', 'ano_id', 'ano_type', and 'explanation'.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.info("Importing data...")
    refs, anos = split_references_and_anomalies(data_folder, label_filename)
    _, labels = get_train_test_data(data_folder, label_filename)

    labels_df = labels[["trace_id", "ano_id"]].copy()

    logging.info("Getting explanatory features...")
    explanatory_features = get_explanatory_features(
        refs,
        anos,
        cluster,
        correlation_threshold,
        false_positive_filtering,
        max_distance,
    )
    explanatory_features_df = pd.DataFrame(
        list(explanatory_features.items()), columns=["index", "explanation"]
    )

    logging.info("Constructing explanations...")
    explanations = pd.merge(
        labels_df, explanatory_features_df, left_index=True, right_index=True
    )
    explanations.drop(["index"], axis=1, inplace=True)
    explanations["explanation"] = explanations["explanation"].apply(
        lambda x: get_features_integer_indice(x, anos)
    )
    explanations["exp_size"] = explanations["explanation"].apply(lambda x: len(x))

    logging.info("Computing instability...")
    test = get_explanations_instabilities(
        explanations,
        refs,
        anos,
        cluster,
        correlation_threshold,
        false_positive_filtering,
    )

    return test


def compute_instability(explanations: list):
    """
    Compute the instability of a list of explanations.

    Parameters:
    explanations (list): A list of explanations.

    Returns:
    float: The instability value, which is a measure of how stable the explanations are.
    """
    instability = 0
    flattened_explanations = [item for sublist in explanations for item in sublist]
    unique_explanations = set(flattened_explanations)
    for feature in unique_explanations:
        p = flattened_explanations.count(feature) / len(flattened_explanations)
        instability += -p * np.log2(p)
    # instability = 1 - len(unique_explanations) / len(explanations)

    return instability


def get_explanations_instabilities(
    explanations: pd.DataFrame,
    refs: pd.DataFrame,
    anos: pd.DataFrame,
    cluster: bool,
    correlation_threshold: float,
    false_positive_filtering: bool,
    max_distance: float,
):
    """
    Computes the instabilities of different types of explanations (bursty, stalled,
    and CPU)
    based on sampled references and anomalies.

    Returns:
        A tuple containing the instabilities of bursty explanations, stalled
        explanations, and CPU explanations.
    """

    for i in range(5):
        sampled_refs = refs.sample(frac=0.8)
        sampled_anos = anos.sample(frac=0.8)

        explanatory_features = get_explanatory_features(
            sampled_refs,
            sampled_anos,
            cluster,
            correlation_threshold,
            false_positive_filtering,
            max_distance,
        )

        col_name = "exp_" + str(i)
        explanations[col_name] = list(explanatory_features.values())

    explanations["exp_instability"] = explanations.apply(
        lambda x: compute_instability(
            [x["exp_0"], x["exp_1"], x["exp_2"], x["exp_3"], x["exp_4"]]
        ),
        axis=1,
    )

    explanations.drop(
        ["exp_0", "exp_1", "exp_2", "exp_3", "exp_4"], axis=1, inplace=True
    )

    return explanations


DATA_FOLDER = "data/folder_1"
LABEL_FILENAME = "labels"
VERBOSE = False
CORRELATION_THRESHOLD = 0.7
MAX_DISTANCE = 100.0

print("Without false positive filtering :")
print("Without clustering:")
csv_without_cluster = construct_explanations(
    DATA_FOLDER,
    LABEL_FILENAME,
    cluster=False,
    false_positive_filtering=False,
    correlation_threshold=CORRELATION_THRESHOLD,
    max_distance=MAX_DISTANCE,
    verbose=VERBOSE,
)
print(csv_without_cluster)
csv_without_cluster.to_csv(
    "data/folder_1_results/explanations_without_false_positive_filtering_"
    f"{MAX_DISTANCE}_without_cluster_{CORRELATION_THRESHOLD}.csv"
)

print("With clustering:")
csv_with_cluster = construct_explanations(
    DATA_FOLDER,
    LABEL_FILENAME,
    cluster=True,
    false_positive_filtering=False,
    correlation_threshold=CORRELATION_THRESHOLD,
    max_distance=MAX_DISTANCE,
    verbose=VERBOSE,
)
print(csv_with_cluster)
csv_with_cluster.to_csv(
    "data/folder_1_results/explanations_without_false_positive_filtering_"
    f"{MAX_DISTANCE}_with_cluster_{CORRELATION_THRESHOLD}.csv"
)


print("With false positive filtering :")
print("Without clustering:")
csv_without_cluster = construct_explanations(
    DATA_FOLDER,
    LABEL_FILENAME,
    cluster=False,
    false_positive_filtering=True,
    correlation_threshold=CORRELATION_THRESHOLD,
    max_distance=MAX_DISTANCE,
    verbose=VERBOSE,
)
print(csv_without_cluster)
csv_without_cluster.to_csv(
    "data/folder_1_results/explanations_with_false_positive_filtering_"
    f"{MAX_DISTANCE}_without_cluster_{CORRELATION_THRESHOLD}.csv"
)

print("With clustering:")
csv_with_cluster = construct_explanations(
    DATA_FOLDER,
    LABEL_FILENAME,
    cluster=True,
    false_positive_filtering=True,
    correlation_threshold=CORRELATION_THRESHOLD,
    max_distance=MAX_DISTANCE,
    verbose=VERBOSE,
)
print(csv_with_cluster)
csv_with_cluster.to_csv(
    "data/folder_1_results/explanations_with_false_positive_filtering_"
    f"{MAX_DISTANCE}_with_cluster_{CORRELATION_THRESHOLD}.csv"
)
