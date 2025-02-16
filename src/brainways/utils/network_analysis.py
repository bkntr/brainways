import networkx as nx
import numpy as np
import pandas as pd
from networkx import Graph
from scipy.stats import t
from statsmodels.sandbox.stats.multicomp import multipletests


def correlation_matrix_to_network_graph(corr: pd.DataFrame) -> nx.Graph:
    corr_graph = nx.from_numpy_array(corr.values)
    corr_graph.remove_edges_from(nx.selfloop_edges(corr_graph))

    node_labels = {
        index: struct
        for index, struct in enumerate(corr.columns)
        if index in corr_graph.nodes
    }
    nx.set_node_attributes(corr_graph, node_labels, "name")
    return corr_graph


def calculate_correlation_graph(cell_counts: pd.DataFrame):
    corr = cell_counts.corr(numeric_only=True, min_periods=3)
    corr.values[np.tril_indices_from(corr.values)] = 0

    corr_graph = nx.from_numpy_array(corr.values)
    corr_graph.remove_edges_from(nx.selfloop_edges(corr_graph))

    node_labels = {
        index: struct
        for index, struct in enumerate(corr.columns)
        if index in corr_graph.nodes
    }
    nx.set_node_attributes(corr_graph, node_labels, "name")
    return corr_graph


def add_null_network_bootstrap_pvalues(
    graph: Graph,
    cell_counts: pd.DataFrame,
    n_bootstraps: int,
    multiple_comparison_correction_method: str = "fdr_bh",
) -> Graph:
    """
    Add edge bootstrap p-values to a NetworkX graph based on the correlation of cell count data.
    This function performs bootstrap sampling on the provided cell counts, computes null
    correlation matrices, and calculates p-values by comparing the observed edge correlations
    to those in the null distribution. The p-values are then adjusted for multiple comparisons
    and stored as the "null_pvalue" attribute on each edge in the graph.

    Args:
        graph : nx.Graph
            A NetworkX graph whose edges will be annotated with bootstrap-based p-values.
        cell_counts : pd.DataFrame
            A pandas DataFrame that includes at least two columns, "animal_id" and
            node count columns corresponding to nodes in the graph.
        n_bootstraps : int
            The number of bootstrap iterations to perform.
        multiple_comparison_correction_method : str, optional
            The method used to correct for multiple comparisons (default is "fdr_bh").
    Returns:
        nx.Graph
            The original graph with an additional "null_pvalue" edge attribute derived from
            the bootstrap analysis.
    """
    null_corrs = []
    animal_ids = cell_counts.index.unique().to_numpy()

    node_id_to_name = nx.get_node_attributes(graph, "name")
    name_to_node_id = {name: node_id for node_id, name in node_id_to_name.items()}
    node_names = list(node_id_to_name.values())
    node_ids = [name_to_node_id[name] for name in node_names]

    assert set(node_names) == set(
        cell_counts.columns
    ), "Node names in `cell_counts` do not match those in `graph`"

    for _ in range(n_bootstraps):
        bootstrap_animal_ids = np.random.choice(
            animal_ids, len(animal_ids), replace=True
        )
        bootstrap_counts = cell_counts.loc[bootstrap_animal_ids]
        corr = bootstrap_counts.corr(numeric_only=True).values
        corr[np.tril_indices_from(corr)] = 0
        null_corrs.append(corr)
    null_corrs = np.stack(null_corrs)

    graph_corrs = nx.to_pandas_adjacency(graph, nodelist=node_ids)

    p_values = np.mean(np.abs(null_corrs) > np.abs(graph_corrs.values), axis=0)
    p_values_flat = p_values[np.triu_indices_from(p_values, k=1)]
    _, p_values_corrected_flat, _, _ = multipletests(
        p_values_flat, method=multiple_comparison_correction_method
    )
    p_values_corrected = np.ones(p_values.shape)
    p_values_corrected[np.triu_indices_from(p_values, k=1)] = p_values_corrected_flat

    edge_attributes = {}
    for i, j in zip(*np.triu_indices_from(p_values, k=1)):
        node_i = node_ids[i]
        node_j = node_ids[j]
        p_value = p_values_corrected[i, j]
        edge_attributes[(node_i, node_j)] = p_value
    nx.set_edge_attributes(graph, edge_attributes, name="null_pvalue")

    return graph


def calculate_network_graph(
    cell_counts: pd.DataFrame,
    n_bootstraps: int = 1000,
    multiple_comparison_correction_method: str = "fdr_bh",
) -> nx.Graph:
    graph = calculate_correlation_graph(cell_counts)
    edge_to_r_values = nx.get_edge_attributes(graph, "weight")
    r_values = np.array(list(edge_to_r_values.values()))

    n = len(cell_counts)
    degrees_of_freedom = n - 2
    t_values = r_values * np.sqrt(degrees_of_freedom / (1 - r_values**2))
    p_values = 2 * t.cdf(-np.abs(t_values), degrees_of_freedom)

    _, p_values_corrected, _, _ = multipletests(
        p_values, method=multiple_comparison_correction_method
    )

    edge_to_p_values_corrected = {
        edge: p_value_corrected
        for edge, p_value_corrected in zip(edge_to_r_values, p_values_corrected)
    }

    nx.set_edge_attributes(graph, edge_to_p_values_corrected, name="pvalue")

    add_null_network_bootstrap_pvalues(
        graph,
        cell_counts,
        n_bootstraps=n_bootstraps,
        multiple_comparison_correction_method=multiple_comparison_correction_method,
    )

    return graph
