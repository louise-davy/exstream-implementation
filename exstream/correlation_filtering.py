import pandas as pd
import networkx as nx

# STEP 1 : FUNCTIONS FOR CORRELATION CLUSTERING


def correlated_features_filter(
    df: pd.DataFrame, correlation_threshold: float, cluster: bool
) -> pd.DataFrame:
    """
    Identify and remove correlated features using (or not) clustering.

    When using clustering, similar features
    are identified using pairwise correlation. A feature is represented as a
    node. Two nodes are connected if the pairwise correlation of the two
    features exceeds a threshold.

    When not using clustering, we do not remove correlated features.

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
        selected_features = df.columns[:-4]

    return selected_features
