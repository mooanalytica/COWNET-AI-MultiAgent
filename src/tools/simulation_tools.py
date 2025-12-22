from typing import Dict, Any, Tuple, Annotated
from langgraph.prebuilt import InjectedState
from langchain.tools import tool
import networkx as nx
from .sna_tools import get_demo_sna_results

@tool
def remove_cow_from_network(cow_id: str):
    """
    Remove a single cow from the social network graph and return the modified graph.

    This tool:
    - Reads the baseline sna graph from state["sna_graph"].
    - Creates a copy of the graph and removes the specified cow_id node.
    - Computes basic connectivity changes (nodes/edges removed).
    - Returns:
        content: Summary of what was removed for the LLM.
        artifact: {"modified_graph": dict representation of G_after} for state.

    Args:
        cow_id: The cow ID (string) to remove from the network.

    """
    return cow_id
    
def _remove_cow_from_network(
    cow_id: str,
    sna_graph: dict
) -> Tuple[str, Dict[str, Any]]:
    """
    Remove a single cow from the social network graph and return the modified graph.

    This tool:
    - Reads the baseline sna graph from state["sna_graph"].
    - Creates a copy of the graph and removes the specified cow_id node.
    - Computes basic connectivity changes (nodes/edges removed).
    - Returns:
        content: Summary of what was removed for the LLM.
        artifact: {"modified_graph": dict representation of G_after} for state.

    Args:
        cow_id: The cow ID (string) to remove from the network.

    """
    # Get baseline graph from state
    if sna_graph is None:
        content = (
            "No baseline sna graph found in state. "
            "Please run the SNA agent first to populate state['sna_graph']."
        )
        return content, {}

    # Rebuild graph from dict representation
    G_base = nx.from_dict_of_dicts(sna_graph)
    
    # Check if cow exists in graph
    if cow_id not in G_base.nodes:
        content = (
            f"Cow {cow_id} is not present in the current social network graph. "
            f"Available cows: {len(G_base.nodes)} total nodes."
        )
        return content, {}

    # Create modified graph (copy + remove node)
    G_modified = G_base.copy()
    original_degree = G_modified.degree(cow_id)
    G_modified.remove_node(cow_id)
    
    # Compute changes
    nodes_removed = 1
    edges_removed = original_degree  # All edges connected to this cow
    
    # Build content summary for LLM
    content = (
        f"Removed cow {cow_id} (degree {original_degree}) from the social network. "
        f"Network now has {G_modified.number_of_nodes()} nodes "
        f"(↓{nodes_removed}) and {G_modified.number_of_edges()} edges "
        f"(↓{edges_removed})."
    )
    
    # Serialize modified graph for state
    modified_graph_dict = nx.to_dict_of_dicts(G_modified)
    
    modified_metrics_dict = get_demo_sna_results(G_modified)
    

    return (content, {"modified_graph": modified_graph_dict, "modified_metrics": modified_metrics_dict})

