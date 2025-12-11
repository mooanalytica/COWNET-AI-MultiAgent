from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from ..state import GraphState


class CowNetSupervisor(BaseModel):
    next: Literal[
        "data_loader_agent",
        "sna_agent",
        "explainer_agent",
        "simulation_agent",
        "research_agent",
    ] = Field(
        description=(
            "Determines which specialist agent to activate next in the workflow: "
            "'data_loader_agent' to (re)load or refresh interaction data; "
            "'sna_agent' for computing or updating social network analysis metrics; "
            "'explainer_agent' for farmer-friendly explanations; "
            "'simulation_agent' for what-if network scenarios; "
            "'research_agent' for consulting rules/intents and combining them with metrics."
        )
    )
    reason: str = Field(
        description=(
            "Detailed justification for the routing decision, explaining why this agent "
            "is the best next step given the current conversation and available state."
        )
    )


def cownet_supervisor_node(
    state: GraphState,
) -> Command[
    Literal[
        "data_loader_agent",
        "sna_agent",
        "explainer_agent",
        "simulation_agent",
        "research_agent",
    ]
]:
    """
    Supervisor node for the CowNet multi-agent system.

    Routes among:
      - data_loader_agent: (re)load interaction data from source files.
      - sna_agent: social network analysis (build/update graph, compute metrics).
      - explainer_agent: explain metrics/simulations to the farmer.
      - simulation_agent: run what-if graph manipulation scenarios.
      - research_agent: retrieve/apply rules/intents and combine them with metrics.
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0,
    )

    system_prompt = """
You are the supervisor for the CowNet multi-agent dairy herd decision support system.

You manage five specialized agents:

1) Data Loader Agent ("data_loader_agent"):
   - Loads or refreshes the raw cow interaction data (e.g., interactions.csv).
   - Should be used when there is no interaction data in state['interactions'],
     or when the farmer indicates that new weekly data is available and needs to be loaded.

2) SNA Agent ("sna_agent"):
   - Builds and updates the cow social network (NetworkX graph) from interaction data.
   - Computes herd-level and cow-level metrics (centralities, risk components, etc.).
   - Should be used when metrics or the network need to be computed or refreshed
     and interaction data is already loaded.

3) Explainer Agent ("explainer_agent"):
   - Turns technical metrics, simulation outputs, and rule-based conclusions into
     clear, farmer-friendly explanations and recommendations.
   - Should be used when the farmer needs an explanation, summary, or narrative answer.

4) Simulation Agent ("simulation_agent"):
   - Runs what-if scenarios by manipulating the social network graph
     (e.g., removing a cow) and recomputing metrics.
   - Should be used when the farmer asks about hypothetical changes or interventions.

5) Research Agent ("research_agent"):
   - Consults the rules/intents knowledge base (JSON rules) and combines them
     with SNA and simulation metrics to justify decisions or interpret risk scores.
   - Should be used when a question depends on formal rules, policies, or intents.

Your responsibilities:
- Inspect the conversation history and current context (interaction data, metrics, simulations, rules).
- If there is no interaction data yet, or if the user mentions new data, route to 'data_loader_agent'.
- If interaction data exists but SNA metrics/graph are missing or outdated, route to 'sna_agent'.
- If the user asks "why", "explain", or wants a recommendation, route to 'explainer_agent'.
- If the user asks about hypothetical removals or changes, route to 'simulation_agent'.
- If the user asks about rules, policies, or justifications tied to the rule base, route to 'research_agent'.
- Avoid redundant work (do not re-run data loading, SNA, or simulations without need).
- Provide a brief, explicit reason for your routing decision.

Return:
- 'next': one of ["data_loader_agent", "sna_agent", "explainer_agent", "simulation_agent", "research_agent"]
- 'reason': short but precise justification referencing the farmer's question and available state.
"""

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    response: CowNetSupervisor = llm.with_structured_output(CowNetSupervisor).invoke(
        messages
    )

    goto = response.next
    reason = response.reason

    print(f"--- CowNet Workflow Transition: Supervisor → {goto.upper()} ---")

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,
    )
