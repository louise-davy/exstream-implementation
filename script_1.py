import numpy as np
import stumpy
import networkx as nx
import pandas as pd
import os


# IMPORT AND FORMAT DATA


def get_train_test_data(data_folder: str, label_filename: str) -> (list, pd.DataFrame):
    """
    Retrieve the train files and labels from the given data folder.

    Parameters:
    data_folder (str): The path to the folder containing the data files.
    label_filename (str): The name of the label file.

    Returns:
    tuple: A tuple containing the list of train files and the labels DataFrame.
    """
    files = [f.split(".")[0] for f in os.listdir(f"{data_folder}")]
    labels = pd.read_csv(f"{data_folder}/{label_filename}.csv", index_col=0)
    train_files = [file for file in files if file != label_filename]

    return train_files, labels


def from_files_to_anomaly_type(train_files: list) -> dict:
    """
    Maps the given list of train files to their corresponding anomaly types.

    Args:
        train_files (list): A list of train files.

    Returns:
        dict: A dictionary mapping each train file to its corresponding anomaly type.
            The keys are the train file names, and the values are the anomaly types.

    Raises:
        ValueError: If an unknown file is encountered in the train_files list.
    """

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
    data_folder: str, label_filename: str
) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the data into references and anomalies based on the provided label file.

    Args:
        data_folder (str): The path to the folder containing the data files.
        label_filename (str): The filename of the label file.

    Returns:
        references (pd.DataFrame): The references data.
        anomalies (pd.DataFrame): The anomalies data.
    """
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


# STEP 1 : FUNCTIONS FOR CORRELATION CLUSTERING


def correlated_features_filter(
    df: pd.DataFrame, correlation_threshold: float = 0.9, cluster=True
) -> pd.DataFrame:
    """
    Identify and remove correlated features using (or not) clustering.

    When using clustering, similar features
    are identified using pairwise correlation. A feature is represented as a
    node. Two nodes are connected if the pairwise correlation of the two
    features exceeds a threshold.

    When not using clustering, we simply remove the features that are
    correlated.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing both the references and the anomalies.

    correlation_threshold : float
        The threshold used to identify the correlated features.

    clustering : bool
        Whether to use clustering or not.

    Returns
    -------
    selected_features : list
        The list of features that are not correlated.
    """

    if cluster:
        # Step 1: Calculate the correlation matrix
        correlation_matrix = df.corr()

        # Step 2: Create a graph based on pairwise correlations
        G = nx.Graph()

        # Add nodes (features) to the graph
        G.add_nodes_from(correlation_matrix.columns)

        # Add edges between nodes if the correlation exceeds a threshold
        for i in range(
            len(correlation_matrix.columns[:-4])
        ):  # Last 4 columns are metadata
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    G.add_edge(
                        correlation_matrix.columns[i], correlation_matrix.columns[j]
                    )

        # Step 3: Extract clusters from the graph
        clusters = list(nx.connected_components(G))

        # Step 4: Select one representative feature from each cluster
        selected_features = [cluster.pop() for cluster in clusters]

    else:
        # Calculate the correlation matrix
        correlation_matrix = df.loc[:, :-4].corr()

        correlation_mask = (correlation_matrix.abs() < correlation_threshold) & (
            correlation_matrix > 0.0
        )

        # Create a set of features to remove
        selected_features = []
        for feature in correlation_matrix.columns:
            correlated_features = correlation_matrix.index[
                correlation_mask[feature]
            ].tolist()
            selected_features.append(correlated_features)

        selected_features = list(set(selected_features))

    return selected_features


# STEP 2 : FUNCTIONS FOR FALSE POSITIVES FILTERING


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
    return new_cols


def assign_cols_per_ano(anos, new_filtered_features):
    anos["filtered_columns"] = None
    for i, ano in enumerate(anos.index.unique()):
        anos.loc[ano, "filtered_columns"] = str(new_filtered_features[i])
    anos["filtered_columns"] = anos["filtered_columns"].apply(
        lambda x: x.strip("][").split(", ")
    )
    return anos


# STEP 3: FUNCTIONS FOR REWARD LEAP FILTERING

# STEP 3.1 : FUNCTIONS FOR SINGLE FEATURE REWARDS RANKING


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
            f"One of the time series is empty. Len of TSA is {nb_ts_a} and len of TSR"
            f" is {nb_ts_r}."
        )
    p_a = nb_ts_a / (nb_ts_a + nb_ts_r)
    p_r = nb_ts_r / (nb_ts_a + nb_ts_r)
    h_class = p_a * np.log2(1 / p_a) + p_r * np.log2(1 / p_r)

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

    # On récupère le nombre de valeurs distinctes pour chaque modalité (soit 1 lorsque
    # pas de doublons, soit 2, lorsqu'il y a des doublons)
    value_to_count_distinct = (
        sorted_values.drop_duplicates().groupby(feature).count().to_dict()["type_data"]
    )

    # On récupère les modalités distinctes (les différentes valeurs prises par la
    # feature)
    modalities = set(sorted_values[feature].tolist())
    # En fait, ce qui est un peu bizarre c'est qu'on a des valeurs continues, mais là
    # on va les traiter comme des valeurs discrètes :
    # Par exemple si on considère une colonne qui prend les valeurs 501.03, 501.03,
    # 502.4, 502.4, 505.0, on itère sur 501.03, 502.4, 505.0

    # On parcourt chaque modalité
    for modality in modalities:
        # On récupère le premier type de données observé (ano ou ref, donc 1 ou 0) (ça
        # nous sera utile plus tard)
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

            # Cas où il n'y a pas le même nombre de références et d'anomalie
            #  (cas le plus chiant)
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

                # Puis on cherche si le dernier type de donnée observé est le plus
                # représenté ou le moins représenté pour savoir où commencer
                start_smallest = 0 if smallest != last_type_data else 1

                # On parcourt la liste 2 par 2 pour mettre la valeur la moins
                # représentée
                for i in range(start_smallest, nb_smallest * 2, 2):
                    list_values[i] = smallest

            # Cas où il y a le même nombre de références et d'anomalies
            # (cas le plus simple)
            else:
                # On parcourt le nombre total d'observations
                for i in range(nb_total):
                    # On alterne entre 0 et 1 en commençant par la valeur opposée à la
                    # dernière valeur observée
                    list_values.append(abs(last_type_data - i % 2 - 1))

            # On récupère le dernier type de donnée observé (toujours 1 ou 0)
            last_type_data = sorted_values.loc[
                sorted_values[feature] == modality, "type_data"
            ].tolist()[-1]

        # On vérifie que la longueur de la liste est bien égale au nombre
        # d'observations pour la modalité
        assert (
            len(list_values)
            == sorted_values[sorted_values[feature] == modality].shape[0]
        ), (
            f"Len of list_values {len(list_values)} is not equal to the number of"
            " observations for the modality"
            f" {sorted_values[sorted_values[feature]==modality].shape[0]}."
        )
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
            # On a un nouveau segment, il faut calculer l'entropie de segmentation
            # partielle du précédent segment
            pi = len(values_inside_segment) / shuffled_values.shape[0]
            segmentation_ent += pi * np.log(1 / pi)

            # On réinitialise la liste des valeurs à l'intérieur du segment avec la
            # nouvelle valeur
            values_inside_segment = [value]
        else:
            # On stocke les valeurs à l'intérieur du segment tant qu'un nouveau segment
            # n'est pas créé
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

        # if all(value < 0 for value in distances.values()):
        #    positive_distances = {
        #        key: np.abs(value) for key, value in distances.items()
        #    }

        sorted_distances = dict(
            sorted(distances.items(), key=lambda x: x[1], reverse=True)
        )

    return sorted_distances


# STEP 3.2 : FUNCTIONS FOR REMOVING LOW-RANKED FEATURES


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

    leaps = [
        last_distance - distance
        for last_distance, distance in zip(
            distances.values(), list(distances.values())[1:]
        )
    ]

    maximum_leap = max(leaps)

    return maximum_leap


def reward_leap_filter(distances: dict) -> list:
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
    if len(distances) > 0:  # j'ai ajouté ça au cas où
        threshold = maximum_leap(distances)
        to_keep = set()
        last_distance = 0

        for feature, distance in distances.items():
            if last_distance != 0:
                leap = last_distance - distance
                if leap == threshold:
                    break
            last_distance = distance
            to_keep.update([feature])

        filtered_features = [
            feature for feature in distances.keys() if feature in to_keep
        ]

        return filtered_features

    else:
        return None


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


def get_explanatory_features(refs: pd.DataFrame, anos: pd.DataFrame, cluster: bool):
    all_data = pd.concat([refs, anos])

    filtered_features = correlated_features_filter(all_data, cluster)

    new_filtered_features = false_positive_filter(refs, anos, filtered_features)
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


def construct_explanations(data_folder: str, label_filename: str, cluster: bool):
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

    explanatory_features = get_explanatory_features(refs, anos, cluster)
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

    test = get_explanations_instabilities(explanations, refs, anos, cluster)

    return test


def compute_instability(explanations: list):
    """
    Compute the instability of a list of explanations.

    Parameters:
    explanations (list): A list of explanations.

    Returns:
    float: The instability value, which is a measure of how stable the explanations are.
    """

    flattened_explanations = [item for sublist in explanations for item in sublist]
    unique_explanations = set(flattened_explanations)
    instability = 1 - len(unique_explanations) / len(explanations)

    return instability


def get_explanations_instabilities(
    explanations: pd.DataFrame, refs: pd.DataFrame, anos: pd.DataFrame, cluster: bool
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
            sampled_refs, sampled_anos, cluster
        )

        col_name = "exp_" + str(i)
        explanations[col_name] = list(explanatory_features.values())

    # instabilty_bursty = compute_instability(explanations_bursty)

    return explanations


DATA_FOLDER = "data/folder_1"
LABEL_FILENAME = "labels"

print("With clustering:")
print(construct_explanations(DATA_FOLDER, LABEL_FILENAME, cluster=True))

print("Without clustering:")
print(construct_explanations(DATA_FOLDER, LABEL_FILENAME, cluster=False))
