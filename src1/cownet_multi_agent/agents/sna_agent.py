from typing import Dict, Any, List
import pandas as pd
import networkx as nx
from langchain_core.messages import AIMessage
from langgraph.types import Command

from ..state import GraphState
from ..tools.sna_tools import (  # adjust import path to your file
    create_social_network_graph,
    get_demo_sna_results,
)


def sna_node(state: GraphState) -> Command:
    """
    SNA node for the CowNet multi-agent system.

    Responsibilities:
    - Use loaded interaction records in state['interactions'] to build an aggregated
      interaction_counts DataFrame.
    - Build the social network graph (sna_graph) with inverse-distance weights.
    - Compute per-cow and herd-level SNA metrics + risk scores.
    - Write results into state['sna_graph'] and state['sna_metrics'].
    - Route back to the supervisor.
    """
    messages: List[Any] = list(state.get("messages", []))
    interactions = state.get("interactions")

    if not interactions:
        # No data → cannot run SNA
        return Command(
            update={
                "messages": messages + [
                    AIMessage(
                        content=(
                            "SNA Agent: No interaction data found in state['interactions']. "
                            "Please run the data_loader_agent first."
                        ),
                        name="sna_agent",
                    )
                ]
            },
            goto="supervisor",
        )

    # interactions is a list[dict] from data_loader_node
    interactions_df = pd.DataFrame(interactions)

    # Expect columns 'cow_i', 'cow_j' at minimum
    if not {"cow_i", "cow_j"}.issubset(interactions_df.columns):
        return Command(
            update={
                "messages": messages + [
                    AIMessage(
                        content=(
                            "SNA Agent: Interaction data is missing 'cow_i' or 'cow_j' columns. "
                            "Cannot build the social network graph."
                        ),
                        name="sna_agent",
                    )
                ]
            },
            goto="supervisor",
        )

    # Aggregate interaction counts per pair (no week separation for demo)
    interaction_counts = (
        interactions_df.groupby(["cow_i", "cow_j"])
        .size()
        .reset_index(name="count")
    )

    if interaction_counts.empty:
        return Command(
            update={
                "messages": messages + [
                    AIMessage(
                        content=(
                            "SNA Agent: Interaction dataset is empty after aggregation. "
                            "No graph or metrics were computed."
                        ),
                        name="sna_agent",
                    )
                ]
            },
            goto="supervisor",
        )

    # 1) Build social network graph (dict-of-dicts) using your helper
    sna_graph_dict = create_social_network_graph(interaction_counts)

    # 2) Rebuild a NetworkX graph from that dict for metric computation
    G = nx.from_dict_of_dicts(sna_graph_dict)

    # 3) Compute per-cow / herd-level metrics and risk scores
    sna_results = get_demo_sna_results(G)
    # sna_results structure:
    # {
    #   'herd_metrics': {...},
    #   'per_cow_metrics': {cow_id: {...}},
    #   'risk_scores': {cow_id: {...}},
    #   'top_risk_cows': [...]
    # }

    # Prepare a short summary message
    herd_metrics = sna_results.get("herd_metrics", {})
    num_cows = herd_metrics.get("num_cows", G.number_of_nodes())
    num_edges = herd_metrics.get("num_edges", G.number_of_edges())
    top_risks = sna_results.get("top_risk_cows", [])

    if top_risks:
        top_str = ", ".join(
            f"{r['cow_id']} (conflict={r['conflict_risk']:.2f}, isolation={r['isolation_risk']:.2f})"
            for r in top_risks
        )
    else:
        top_str = "No high-risk cows identified."

    summary = (
        f"SNA Agent: Computed social network with {num_cows} cows and {num_edges} edges. "
        f"Top risk cows: {top_str}"
    )

    # 4) Update GraphState
    return Command(
        update={
            "sna_graph": sna_graph_dict,
            "sna_metrics": sna_results,  # store full metrics bundle
            "messages": messages + [
                AIMessage(content=summary, name="sna_agent")
            ],
        },
        goto="supervisor",
    )
