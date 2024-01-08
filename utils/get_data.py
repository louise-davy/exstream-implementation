import os
import pandas as pd


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
