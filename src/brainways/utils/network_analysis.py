from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pingouin  # noqa: F401


def calculate_network_graph(cell_counts: pd.DataFrame, alpha: float, output_path: Path):
    corr = cell_counts.rcorr(stars=False, padjust="fdr_bh")
    np.fill_diagonal(corr.values, 0)
    corr = corr.astype(float)
    corr.values[np.tril_indices_from(corr, k=-1)] = 0
    corr[corr > alpha] = 0

    corr_graph = nx.from_numpy_array(corr.values)
    corr_graph.remove_edges_from(nx.selfloop_edges(corr_graph))

    node_labels = {
        index: struct
        for index, struct in enumerate(corr.columns)
        if index in corr_graph.nodes
    }
    nx.set_node_attributes(corr_graph, node_labels, "name")
    nx.write_graphml(corr_graph, output_path)
