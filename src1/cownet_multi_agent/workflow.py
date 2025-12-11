from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .state import GraphState
from ..cownet_multi_agent.agents.supervisor import cownet_supervisor_node
from ..cownet_multi_agent.agents.data_loader import data_loader_node
from ..cownet_multi_agent.agents.sna_agent import sna_node
from ..cownet_multi_agent.agents.simulation_agent import simulation_node
from ..cownet_multi_agent.agents.explainer_agent import explainer_agent_node
from ..cownet_multi_agent.agents.research_agent import research_node

# Build the multi-agent graph
def build_cownet_workflow():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("supervisor", cownet_supervisor_node)
    graph.add_node("data_loader_agent", data_loader_node)
    graph.add_node("sna_agent", sna_node)
    graph.add_node("simulation_agent", simulation_node)
    graph.add_node("explainer_agent", explainer_agent_node)
    graph.add_node("research_agent", research_node)

    # Edges: each worker agent returns goto="supervisor"
    graph.add_edge("data_loader_agent", "supervisor")
    graph.add_edge("sna_agent", "supervisor")
    graph.add_edge("simulation_agent", "supervisor")
    graph.add_edge("research_agent", "supervisor")
    graph.add_edge("explainer_agent", END)  # explainer ends the workflow

    # Start at supervisor
    graph.set_entry_point("supervisor")

    # In-memory checkpointer
    checkpointer = MemorySaver()

    return graph

graph = build_cownet_workflow().compile(checkpointer=MemorySaver())