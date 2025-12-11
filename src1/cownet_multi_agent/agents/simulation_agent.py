from typing import Annotated, List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.types import Command
from ...core.language_models import LanguageModelManager
from .base_agent import BaseAgent
from ..config import WORKING_DIRECTORY
from ..state import GraphState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..tools.simulation_tools import remove_cow_from_network

class SimulationResponse(BaseModel):
    """Structured decision for simulation agent."""
    cow_id: Annotated[str, Field(
        description="The specific cow ID to remove from the social network. "
                   "Must be one of the existing cows in the current graph."
    )]

def simulation_node(state: GraphState) -> Command:
    # Initialize the LLM with structured output
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0,
    )
    llm_structured = llm.with_structured_output(SimulationResponse)

    # Build a concise system + user prompt using the message history and current graph context
    messages: List[BaseMessage] = state.get("messages", [])
    sna_graph = state.get("sna_graph")

    # Provide context about available nodes to help the LLM pick a valid cow_id
    graph_summary = ""
    if sna_graph is None or not sna_graph:
        graph_summary = "No proximity graph found."
    else:
        # List a small sample of cow IDs to ground the choice
        node_ids = list(sna_graph.keys())
        sample_ids = node_ids[:10]
        graph_summary = (
            f"Current network has {len(node_ids)} cows. "
            f"Example cow IDs: {', '.join(sample_ids)}."
        )

    system_prompt = (
        "You are the simulation agent. Based on the prior discussion and analysis in the message history, "
        "choose the single cow ID that should be removed from the social network to run a simulation. "
        "Return only a valid existing cow_id present in the current graph."
    )

    user_prompt = (
        f"Context: {graph_summary}\n"
        "Pick the cow_id to remove now."
    )

    prompt_messages: List[BaseMessage] = [
        AIMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    # Include prior messages to give the LLM history
    prompt_messages = messages + prompt_messages

    # Ask the LLM for the cow_id
    try:
        sim_decision: SimulationResponse = llm_structured.invoke(prompt_messages)
        cow_id = sim_decision.cow_id
    except Exception as e:
        error_msg = f"Simulation agent failed to select a cow_id: {e}"
        return Command(
            update={
                "messages": messages + [AIMessage(content=error_msg)]
            }
        )

    # Run the simulation tool to remove the cow and get the modified graph + summary
    tool_result = remove_cow_from_network(cow_id=cow_id, sna_graph=sna_graph)

    # tool_result follows:
    # {
    #   "content": <summary string>,
    #   "modified_graph": <dict of dicts>
    # }
    content = tool_result.get("content", "Simulation completed.")
    modified_graph = tool_result.get("modified_graph", {})
    modified_metrics = tool_result.get("modified_metrics", {})

    # Update state with the simulation graph and append a summary message
    return Command(
        update={
            "simulation_graph": modified_graph,
            "simulation_metrics": modified_metrics,
            "messages": messages + [
                AIMessage(content=f"Simulation: {content}"),
                AIMessage(content=f"Removed cow_id: {cow_id}")
            ],
        },
        goto="supervisor"
    )