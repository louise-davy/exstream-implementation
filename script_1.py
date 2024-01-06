import numpy as np
import stumpy
import networkx as nx
import pandas as pd
import os


# IMPORT AND FORMAT DATA

def get_train_test_data(
        data_folder: str,
        label_filename: str
        ) -> (list, pd.DataFrame):

    files = [f.split(".")[0] for f in os.listdir(f"{data_folder}")]
    labels = pd.read_csv(f"{data_folder}/{label_filename}.csv", index_col=0)
    train_files = [file for file in files if file != label_filename]

    return train_files, labels


def from_files_to_anomaly_type(train_files: list) -> dict:

    labeled_files = {}

    for file in train_files:
        if file.startswith("1"):
            labeled_files[file] = "bursty input"
        elif file.startswith("2"):
            labeled_files[file] = "stalled input"
        elif file.startswith("3"):
            labeled_files[file] = "CPU contention"
        else:
            raise ValueError(f"Unknown file {file}.")

    return labeled_files


def split_references_and_anomalies(
        data_folder: str,
        label_filename: str
        ) -> (pd.DataFrame, pd.DataFrame):

    train_files, labels = get_train_test_data(data_folder, label_filename)
    labeled_files = from_files_to_anomaly_type(train_files)

    references = {}
    anomalies = {}

    for filename in train_files:
        train_file = pd.read_csv(f"{data_folder}/{filename}.csv", index_col=0)
        train_file["original_filename"] = filename
        train_file["timestamp"] = train_file.index

        label_file = labels.loc[labels["trace_id"] == filename, :]

        for i in label_file.index:
            ano_id = label_file.loc[i, "ano_id"]
            selection_ref = train_file.loc[
                (train_file["timestamp"] >= label_file["ref_start"][i])
                & (train_file["timestamp"] < label_file["ref_end"][i]),
                :,
            ].copy()
            selection_ref["ano_id"] = ano_id
            selection_ref["type_data"] = 0
            selection_ano = train_file.loc[
                (train_file["timestamp"] >= label_file["ano_start"][i])
                & (train_file["timestamp"] <= label_file["ano_end"][i]),
                :,
            ].copy()
            selection_ano["ano_id"] = ano_id
            selection_ano["type_data"] = 1
            references[f"{labeled_files[filename]}_{filename}_{i}"] = selection_ref
            anomalies[f"{labeled_files[filename]}_{filename}_{i}"] = selection_ano

    assert references.keys() == anomalies.keys()
    references = pd.concat(references).droplevel(1)
    anomalies = pd.concat(anomalies).droplevel(1)

    return references, anomalies


# USEFUL FUNCTIONS


def class_entropy(nb_ts_a: list, nb_ts_r: list) -> float:
    """
    Calculate the class entropy of a feature, which is the information
    needed to describe the class distributions between two time serie.

    Parameters
    ----------
    nb_ts_a : list
        Number of observations inside a time series belonging to the abnormal
        class.
    nb_ts_r : list
        Number of observations inside a time series belonging to the reference
        class.

    Returns
    -------
    float
        The class entropy.
    """

    if nb_ts_a == 0 or nb_ts_r == 0:
        raise ValueError(
            f"One of the time series is empty. Len of TSA is {nb_ts_a} and len of TSR is {nb_ts_r}."
        )
    p_a = nb_ts_a / (nb_ts_a + nb_ts_r)
    p_r = nb_ts_r / (nb_ts_a + nb_ts_r)
    h_class = p_a * np.log2(p_a) + p_r * np.log2(p_r)

    return h_class


def shuffle_observations_if_duplicates(
    sorted_values: pd.DataFrame, feature
) -> pd.DataFrame:
    """
    Shuffle the observations if there are duplicates in the sorted values.

    Parameters
    ----------
    sorted_values : pd.DataFrame
        The sorted values.

    Returns
    -------
    pd.DataFrame
        The sorted values with shuffled values for duplicates.
    """

    # On récupère le nombre de références et d'anomalies pour chaque modalité
    value_type_to_count = sorted_values.groupby(feature).value_counts().to_dict()

    # On récupère le nombre de valeurs distinctes pour chaque modalité (soit 1 lorsque pas de doublons, soit 2, lorsqu'il y a des doublons)
    value_to_count_distinct = (
        sorted_values.drop_duplicates().groupby(feature).count().to_dict()["type_data"]
    )

    # On récupère les modalités distinctes (les différentes valeurs prises par la feature)
    modalities = set(sorted_values[feature].tolist())
    # En fait, ce qui est un peu bizarre c'est qu'on a des valeurs continues, mais là on va les traiter comme des valeurs discrètes :
    # Par exemple si on considère une colonne qui prend les valeurs 501.03, 501.03, 502.4, 502.4, 505.0, on itère sur 501.03, 502.4, 505.0

    # On parcourt chaque modalité
    for modality in modalities:
        # On récupère le premier type de données observé (ano ou ref, donc 1 ou 0) (ça nous sera utile plus tard)
        last_type_data = sorted_values.loc[
            sorted_values[feature] == modality, "type_data"
        ].tolist()[0]

        # Cas où il n'y a pas de doublons
        if value_to_count_distinct[modality] == 1:
            # On ne fait rien
            continue

        # Cas où il y a des doublons
        else:
            # On va shuffle dans le pire ordre possible

            # D'abord on instancie les variables nécessaires
            list_values = []
            nb_refs = value_type_to_count[(modality, 0)]
            nb_anos = value_type_to_count[(modality, 1)]
            nb_total = nb_refs + nb_anos
            # Hop maintenant c'est parti pour le shuffle

            # Cas où il n'y a pas le même nombre de références et d'anomalies (cas le plus chiant)
            if nb_refs != nb_anos:
                # On instancie de nouveau des variables nécessaires
                biggest = int(
                    nb_refs < nb_anos
                )  # 1 si on a plus d'anomalies que de références, 0 sinon
                smallest = int(
                    nb_refs > nb_anos
                )  # 1 si on a plus de références que d'anomalies, 0 sinon
                nb_smallest = min(
                    nb_refs, nb_anos
                )  # Nombre de fois où on va mettre la valeur la moins représentée

                # On commence par mettre la valeur la plus représentée partout
                list_values = [biggest] * nb_total

                # Puis on cherche si le dernier type de donnée observé est le plus représenté ou le moins représenté pour savoir où commencer
                start_smallest = 0 if smallest != last_type_data else 1

                # On parcourt la liste 2 par 2 pour mettre la valeur la moins représentée
                for i in range(start_smallest, nb_smallest * 2, 2):
                    list_values[i] = smallest

            # Cas où il y a le même nombre de références et d'anomalies (cas le plus simple)
            else:
                # On parcourt le nombre total d'observations
                for i in range(nb_total):
                    # On alterne entre 0 et 1 en commençant par la valeur opposée à la dernière valeur observée
                    list_values.append(abs(last_type_data - i % 2 - 1))

            # On récupère le dernier type de donnée observé (toujours 1 ou 0)
            last_type_data = sorted_values.loc[
                sorted_values[feature] == modality, "type_data"
            ].tolist()[-1]

        # On vérifie que la longueur de la liste est bien égale au nombre d'observations pour la modalité
        assert (
            len(list_values)
            == sorted_values[sorted_values[feature] == modality].shape[0]
        ), f"Len of list_values {len(list_values)} is not equal to the number of observations for the modality {sorted_values[sorted_values[feature]==modality].shape[0]}."
        # On met à jour le type de donnée pour la modalité
        sorted_values.loc[sorted_values[feature] == modality, "type_data"] = list_values

    return sorted_values


def segmentation_entropy(shuffled_values: pd.DataFrame) -> float:
    """
    Calculate the segmentation entropy of a feature, which is the information
    needed to describe how merged points are segmented by class labels.

    Parameters:
    shuffled_values (pd.DataFrame): A DataFrame containing the shuffled values.

    Returns:
    float: The segmentation entropy of the feature.
    """
    # On récupère la time serie
    ts = shuffled_values["type_data"].tolist()

    # Stocke la première valeur de la liste
    past_value = ts[0]

    # Liste pour stocker les valeurs à l'intérieur d'un segment
    values_inside_segment = []

    # Variable pour stocker la segmentation entropy
    segmentation_ent = 0.0

    # Parcourt chaque valeur dans la time serie
    for value in ts:
        # Si la valeur est différente de la valeur précédente
        if value != past_value:
            # On a un nouveau segment, il faut calculer l'entropie de segmentation partielle du précédent segment
            pi = len(values_inside_segment) / shuffled_values.shape[0]
            segmentation_ent += pi * np.log(1 / pi)

            # On réinitialise la liste des valeurs à l'intérieur du segment avec la nouvelle valeur
            values_inside_segment = [value]
        else:
            # On stocke les valeurs à l'intérieur du segment tant qu'un nouveau segment n'est pas créé
            values_inside_segment.append(value)

        # On met à jour la valeur précédente avec la valeur actuelle
        past_value = value

    return segmentation_ent


def entropy_based_single_feature_reward(refs: pd.DataFrame, anos: pd.DataFrame) -> dict:
    """
    Calculates the reward function for a single feature based on the reference
    data and the annotated data.

    Parameters:
    refs (pandas.DataFrame): The reference data.
    anos (pandas.DataFrame): The abnormal data.

    Returns:
    dict: A dictionary containing the single reward function for each feature.
    """

    distances = {}
    # On calcule la class entropy
    class_ent = class_entropy(refs.shape[0], anos.shape[0])
    # On calcule la segmentation entropy pour chaque feature sauf type_data
    for feature in [col for col in refs.columns if col != "type_data"]:
        # On concatène les références et les anomalies pour la feature
        all_values = pd.concat(
            [refs[[feature, "type_data"]], anos[[feature, "type_data"]]]
        )
        # On trie les valeurs par feature puis par type_data
        sorted_values = all_values.sort_values(by=[feature, "type_data"])
        # On shuffle les valeurs si on a des doublons
        shuffled_values = shuffle_observations_if_duplicates(sorted_values, feature)
        # On calcule la segmentation entropy
        segmentation_ent = segmentation_entropy(shuffled_values)
        # On calcule la single reward function
        distance = class_ent / segmentation_ent
        # On stocke la single reward function dans le dictionnaire
        distances[feature] = distance

        if all(value < 0 for value in distances.values()):
            positive_distances = {key: np.abs(value) for key, value in distances.items()}

        sorted_distances = dict(sorted(positive_distances.items(), key=lambda x: x[1], reverse=True))

    return sorted_distances


def correlated_features_filter(df: pd.DataFrame, correlation_threshold: float = 0.9) -> pd.DataFrame:
    """
    Identify and remove correlated features using clustering. Similar features
    are identified using pairwise correlation. A feature is represented as a
    node. Two nodes are connected if the pairwise correlation of the two
    features exceeds a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing both the references and the anomalies.

    correlation_threshold : float
        The threshold used to identify the correlated features.

    Returns
    -------
    selected_features : pd.DataFrame
        The initial dataframe cleared of its correlated features.
    """

    correlation_matrix = df.corr()

    G = nx.Graph()
    G.add_nodes_from(correlation_matrix.columns)

    for i in range(len(correlation_matrix.columns[:-4])):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j])

    clusters = list(nx.connected_components(G))

    selected_features = [cluster.pop() for cluster in clusters]

    return selected_features


def false_positive_filter(refs: pd.DataFrame, anos: pd.DataFrame, cols: list) -> list:
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

    new_cols = []
    refs_df = refs[cols]
    anos_df = anos[cols]
    cols_to_visit = list(anos_df.columns[:-4])

    for ano in anos_df.index.unique():
        for col in cols_to_visit:
            pattern = anos_df.loc[ano, col]
            ts = refs_df.loc[:, col]
            matches = stumpy.match(pattern, ts, max_distance=28.0)
            if not list(matches):
                if col not in new_cols:
                    new_cols.append(col)
            else:
                cols_to_visit.remove(col)
                # print(f"Found {len(matches)} match(es) for {col} in ano {ano}")

    return new_cols


def maximum_leap(distances: dict) -> float:
    """
    Calculate the maximum leap between the distances of the neighboring ranked
    features.

    Parameters
    ----------
    distances : dict
        The ranking of all remaining features based on their individual reward.

    Returns
    -------
    maximum_leap : float
        The maximum leap.

    """

    leaps = [last_distance - distance for last_distance, distance in zip(distances.values(), list(distances.values())[1:])]

    maximum_leap = max(leaps)

    return maximum_leap


def reward_leap_filter(distances: dict) -> dict:
    """
    Discard the features that rank below a sharp drop in the reward.

    Parameters
    ----------
    distances : dict
        The ranking of all remaining features based on their individual reward.

    Returns
    -------
    filtered_features : dict
        The remaining features after filtering.

    """

    threshold = maximum_leap(distances)
    to_be_discarded = set()

    last_distance = 0
    for feature, distance in distances.items():
        if last_distance != 0:
            leap = last_distance - distance
            if leap <= threshold:
                to_be_discarded.update([feature])
        last_distance = distance

    filtered_features = [feature for feature in distances.keys() if feature not in to_be_discarded]

    return filtered_features


def get_explanatory_features():
    DATA_FOLDER = "folder_1"
    LABEL_FILENAME = "labels"

    references, anomalies = split_references_and_anomalies(DATA_FOLDER, LABEL_FILENAME)

    bursty_refs = references[references.index.str.startswith("bursty")]
    bursty_anos = anomalies[anomalies.index.str.startswith("bursty")]
    bursty_df = pd.concat([bursty_refs, bursty_anos])

    stalled_refs = references[references.index.str.startswith("stalled")]
    stalled_anos = anomalies[anomalies.index.str.startswith("stalled")]
    stalled_df = pd.concat([stalled_refs, stalled_anos])

    cpu_refs = references[references.index.str.startswith("CPU")]
    cpu_anos = anomalies[anomalies.index.str.startswith("CPU")]
    cpu_df = pd.concat([cpu_refs, cpu_anos])

    all_dfs = [bursty_df, stalled_df, cpu_df]

    # STEP 1 : FILTERING BY CORRELATION CLUSTERING

    filtered_features = []

    for df in all_dfs:
        filtered_features.append(correlated_features_filter(df))

    # STEP 2 : FALSE POSITIVE FILTERING

    new_filtered_features_bursty = false_positive_filter(
        bursty_refs, bursty_anos, filtered_features[0]
    )
    new_filtered_features_stalled = false_positive_filter(
        stalled_refs, stalled_anos, filtered_features[1]
    )
    new_filtered_features_cpu = false_positive_filter(
        cpu_refs, cpu_anos, filtered_features[2]
    )

    # STEP 3 : REWARD LEAP FILTERING

    # STEP 3.1 : SINGLE-FEATURE REWARD RANKING

    distances_bursty = entropy_based_single_feature_reward(
        bursty_refs.loc[:, new_filtered_features_bursty + ["type_data"]],
        bursty_anos.loc[:, new_filtered_features_bursty + ["type_data"]],
    )

    distances_stalled = entropy_based_single_feature_reward(
        stalled_refs.loc[:, new_filtered_features_stalled + ["type_data"]],
        stalled_anos.loc[:, new_filtered_features_stalled + ["type_data"]],
    )

    distances_cpu = entropy_based_single_feature_reward(
        cpu_refs.loc[:, new_filtered_features_cpu + ["type_data"]],
        cpu_anos.loc[:, new_filtered_features_cpu + ["type_data"]],
    )

    # STEP 3.2 : REMOVING LOW-RANKED FEATURES

    features_bursty = reward_leap_filter(distances_bursty)
    features_stalled = reward_leap_filter(distances_stalled)
    features_cpu = reward_leap_filter(distances_cpu)

    return features_bursty, features_stalled, features_cpu


def get_features_integer_indice(features: list, anomalies: pd.DataFrame):
    indices = []
    for feature in features:
        indice = anomalies.columns.get_loc(feature)
        indices.append(indice)

    return indices


def construct_explanations(labels: pd.DataFrame):
    columns_to_add = ["exp_size", "exp_instability", "explanation"]
    explanations = labels[["trace_id", "ano_id", "ano_type"]].copy()

    features_bursty, features_stalled, features_cpu = get_explanatory_features()

    explanations.loc[explanations["ano_type"] == "bursty_input", "explanation"] = features_bursty.values()
    explanations.loc[explanations["ano_type"] == "stalled_input", "explanation"] = features_stalled.values()
    explanations.loc[explanations["ano_type"] == "cpu_contention", "explanation"] = features_cpu.values()

    return explanations


# _, labels = get_train_test_data("folder_1", "labels")

# print(construct_explanations(labels))
features_bursty, features_stalled, features_cpu = get_explanatory_features()
_, anomalies = split_references_and_anomalies("folder_1", "labels")

indices = get_features_integer_indice(features_bursty, anomalies)
print(indices)
print(features_bursty)
