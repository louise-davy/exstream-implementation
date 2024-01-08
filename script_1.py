import numpy as np
import pandas as pd

from exstream.correlation_filtering import correlated_features_filter
from exstream.false_positive_filtering import false_positive_filter, assign_cols_per_ano
from exstream.entropy_based_single_reward_feature import (
    entropy_based_single_feature_reward,
    reward_leap_filter,
)
from utils.get_data import get_train_test_data, split_references_and_anomalies


def compute_explanatory_features(anos: pd.DataFrame, distances: dict) -> dict:
    selected_features = {}

    for ano_index in anos.index.unique():
        ano = anos.loc[ano_index]
        filtered_cols = ano["filtered_columns"].values
        for cols in np.unique(filtered_cols):
            cols = [s.replace("'", "") for s in cols]
            selected_distances = {
                feat: dist for feat, dist in distances.items() if feat in cols
            }
            if len(selected_distances) > 1:
                filtered_features = reward_leap_filter(selected_distances)
                selected_features[ano_index] = filtered_features
            else:
                selected_features[ano_index] = list(selected_distances.keys())

    return selected_features


# NOTEBOOK STOPED HERE


def get_explanatory_features(
    refs: pd.DataFrame,
    anos: pd.DataFrame,
    cluster: bool,
    false_positive_filtering: bool,
):
    all_data = pd.concat([refs, anos])

    filtered_features = correlated_features_filter(
        all_data, correlation_threshold=0.9, cluster=cluster
    )

    new_filtered_features = false_positive_filter(
        refs, anos, filtered_features, false_positive_filtering
    )
    new_anos = assign_cols_per_ano(anos, new_filtered_features)

    bursty_refs = refs[refs.index.str.startswith("bursty")]
    stalled_refs = refs[refs.index.str.startswith("stalled")]
    cpu_refs = refs[refs.index.str.startswith("CPU")]

    bursty_anos = new_anos[anos.index.str.startswith("bursty")]
    stalled_anos = new_anos[anos.index.str.startswith("stalled")]
    cpu_anos = new_anos[anos.index.str.startswith("CPU")]

    bursty_distances = entropy_based_single_feature_reward(bursty_refs, bursty_anos)
    stalled_distances = entropy_based_single_feature_reward(stalled_refs, stalled_anos)
    cpu_distances = entropy_based_single_feature_reward(cpu_refs, cpu_anos)

    bursty_features = compute_explanatory_features(bursty_anos, bursty_distances)
    stalled_features = compute_explanatory_features(stalled_anos, stalled_distances)
    cpu_features = compute_explanatory_features(cpu_anos, cpu_distances)

    explanatory_features = {**bursty_features, **stalled_features, **cpu_features}

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
    data_folder: str, label_filename: str, cluster: bool, false_positive_filtering: bool
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

    refs, anos = split_references_and_anomalies(data_folder, label_filename)
    _, labels = get_train_test_data(data_folder, label_filename)

    labels_df = labels[["trace_id", "ano_id"]].copy()

    explanatory_features = get_explanatory_features(
        refs, anos, cluster, false_positive_filtering
    )
    explanatory_features_df = pd.DataFrame(
        list(explanatory_features.items()), columns=["index", "explanation"]
    )

    explanations = pd.merge(
        labels_df, explanatory_features_df, left_index=True, right_index=True
    )
    explanations.drop(["index"], axis=1, inplace=True)
    explanations["explanation"] = explanations["explanation"].apply(
        lambda x: get_features_integer_indice(x, anos)
    )
    explanations["exp_size"] = explanations["explanation"].apply(lambda x: len(x))

    test = get_explanations_instabilities(
        explanations, refs, anos, cluster, false_positive_filtering
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
    false_positive_filtering: bool,
):
    """
    Computes the instabilities of different types of explanations (bursty, stalled,
    and CPU)
    based on sampled references and anomalies.

    Returns:
        A tuple containing the instabilities of bursty explanations, stalled
        explanations, and CPU explanations.
    """
    anos.drop("filtered_columns", axis=1, inplace=True)

    for i in range(5):
        sampled_refs = refs.sample(frac=0.8)
        sampled_anos = anos.sample(frac=0.8)

        explanatory_features = get_explanatory_features(
            sampled_refs, sampled_anos, cluster, false_positive_filtering
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

print("Without false positive filtering :")
print("Without clustering:")
csv_without_cluster = construct_explanations(
    DATA_FOLDER, LABEL_FILENAME, cluster=False, false_positive_filtering=False
)
print(csv_without_cluster)
csv_without_cluster.to_csv(
    "data/folder_1_results/explanations_without_filtering_without_cluster.csv"
)

print("With clustering:")
csv_with_cluster = construct_explanations(
    DATA_FOLDER, LABEL_FILENAME, cluster=True, false_positive_filtering=False
)
print(csv_with_cluster)
csv_with_cluster.to_csv(
    "data/folder_1_results/explanations_without_filtering_with_cluster.csv"
)


print("With false positive filtering :")
print("Without clustering:")
csv_without_cluster = construct_explanations(
    DATA_FOLDER, LABEL_FILENAME, cluster=False, false_positive_filtering=True
)
print(csv_without_cluster)
csv_without_cluster.to_csv(
    "data/folder_1_results/explanations_with_filtering_without_cluster.csv"
)

print("With clustering:")
csv_with_cluster = construct_explanations(
    DATA_FOLDER, LABEL_FILENAME, cluster=True, false_positive_filtering=True
)
print(csv_with_cluster)
csv_with_cluster.to_csv(
    "data/folder_1_results/explanations_with_filtering_with_cluster.csv"
)
