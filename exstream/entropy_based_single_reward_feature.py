import logging
import pandas as pd
import numpy as np

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
    value_type_to_count = sorted_values[[feature, "type_data"]].value_counts().to_dict()
    # On récupère le nombre de valeurs distinctes pour chaque modalité (soit 1 lorsque
    # pas de doublons, soit 2, lorsqu'il y a des doublons)
    value_to_count_distinct = (
        sorted_values[feature].drop_duplicates().value_counts().to_dict()
    )
    # print("value_to_count_distinct", value_to_count_distinct)

    # On récupère les modalités distinctes (les différentes valeurs prises par la
    # feature)
    modalities = set(sorted_values[feature].tolist())
    # print("modalities", modalities)
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


def entropy_based_single_feature_reward(
    refs: pd.DataFrame, anos: pd.DataFrame, all_data: pd.DataFrame
) -> dict:
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
        # On trie les valeurs par feature puis par type_data
        sorted_values = all_data.sort_values(by=[feature, "type_data"])
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
        logging.warning("Dict distances is empty.")
        return None
