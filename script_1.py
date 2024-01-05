import numpy as np
import math


def class_entropy(ts_a: list, ts_r: list) -> float:
    """Calculate the entropy of a class, which is the information needed to describe the class distributions between two time series.

    Parameters
    ----------
    tsa : list
        A time series belonging to the abnormal class.
    tsr : list
        A time series belong to the reference class.

    Returns
    -------
    float
        The entropy of the class.
    """
    nb_ts_a = len(ts_a)
    nb_ts_r = len(ts_r)
    if nb_ts_a == 0 or nb_ts_r == 0:
        raise ValueError(f"One of the class is empty. Len of TSA is {nb_ts_a} and len of TSR is {nb_ts_r}.")
    p_a = nb_ts_a / (nb_ts_a + nb_ts_r)
    p_r = nb_ts_r / (nb_ts_a + nb_ts_r)
    h_class = p_a * np.log2(p_a) + p_r * np.log2(p_r)
    return h_class


def segmentation_entropy(segmentations):
    """
    Là j'ai pas trop compris c'était quoi l'input.
    """
    h_segmentation = 0
    for pi in segmentations:
        if pi == 0:
            raise ValueError("One of the segmentations is empty.")
        h_segmentation += pi * math.log(1/pi)
    return h_segmentation


def single_feature_reward(ts_a, ts_r, segmentations):
    """
    Pareil du coup là.
    """
    d = class_entropy(ts_a, ts_r) / segmentation_entropy(segmentations)
    return d


def absolute_and_sort_distances(distances: dict) -> dict:

    if all(value < 0 for value in distances.values()):
        positive_distances = {key: np.abs(value) for key, value in distances.items()}

    sorted_distances = dict(sorted(positive_distances.items(), key=lambda x: x[1], reverse=True))

    return sorted_distances


def maximum_leap(distances: dict) -> float:
    ranked_distances = absolute_and_sort_distances(distances)

    leaps = [last_distance - distance for last_distance, distance in zip(ranked_distances.values(), list(ranked_distances.values())[1:])]

    maximum_leap = max(leaps)

    return maximum_leap


def reward_leap_filter(distances: dict) -> dict:
    threshold = maximum_leap(distances)
    to_be_discarded = set()

    for a, i in distances.items():
        for b, j in distances.items():
            if a != b:
                leap = j - i
                print(leap)
                if leap > threshold:
                    to_be_discarded.update([a, b])

    filtered_features = {feature: distance for feature, distance in distances.items() if feature not in to_be_discarded}

    return filtered_features
