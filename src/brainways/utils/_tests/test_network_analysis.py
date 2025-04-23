import networkx as nx
import numpy as np
import pandas as pd

from brainways.utils.network_analysis import calculate_network_graph


def test_calculate_network_graph():
    region_1 = np.random.rand(100)
    region_2 = region_1 * 100 + np.random.rand(100)
    region_3 = np.random.rand(100)
    # Create synthetic data
    data = {
        "animal_id": [str(i) for i in range(100)],
        "region1": region_1,
        "region2": region_2,
        "region3": region_3,
    }
    df = pd.DataFrame(data).set_index("animal_id")

    # Run function
    graph = calculate_network_graph(df, n_bootstraps=100)

    # Validate graph structure
    assert len(graph.nodes) == 3  # region1, region2, region3
    assert len(graph.edges) > 0

    # Check pvalues
    p_values = nx.get_edge_attributes(graph, "pvalue")
    assert all(
        0 <= val <= 1 for val in p_values.values()
    ), "pvalues must be between 0 and 1"
    assert (
        p_values[(0, 1)] < 1e-5
    ), "pvalue between region1 and region2 should be significant"
    assert (
        p_values[(0, 2)] > 0.5
    ), "pvalue between region1 and region3 should not be significant"
    assert (
        p_values[(1, 2)] > 0.5
    ), "pvalue between region2 and region3 should not be significant"

    # Check null pvalues
    null_pvalues = nx.get_edge_attributes(graph, "null_pvalue")
    assert len(null_pvalues) == len(p_values)
    assert all(
        0 <= val <= 1 for val in null_pvalues.values()
    ), "null pvalues must be between 0 and 1"

    # Check node names
    assert nx.get_node_attributes(graph, "name") == {
        0: "region1",
        1: "region2",
        2: "region3",
    }
