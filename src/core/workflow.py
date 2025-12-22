from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from core.state import GraphState
from ..agents.supervisor import cownet_supervisor_node
from ..agents.data_loader import data_loader_node
from ..agents.sna_agent import sna_node
from ..agents.simulation_agent import simulation_node
from ..agents.response_agent import response_agent_node
from ..agents.research_agent import research_node
from ..agents.report_agent import report_agent_node

# Build the multi-agent graph
def build_cownet_workflow():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("supervisor", cownet_supervisor_node)
    graph.add_node("data_loader_agent", data_loader_node)
    graph.add_node("sna_agent", sna_node)
    graph.add_node("simulation_agent", simulation_node)
    graph.add_node("response_agent", response_agent_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("report_agent", report_agent_node)

    # Edges: each worker agent returns goto="supervisor"
    graph.add_edge("data_loader_agent", "supervisor")
    graph.add_edge("sna_agent", "supervisor")
    graph.add_edge("simulation_agent", "supervisor")
    graph.add_edge("research_agent", "supervisor")
    graph.add_edge("response_agent", END)  # response ends the workflow
    graph.add_edge("report_agent", END)  # report ends the workflow

    # Start at supervisor
    graph.set_entry_point("supervisor")

    return graph

graph = build_cownet_workflow().compile()

