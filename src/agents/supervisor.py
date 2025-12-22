from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from core.state import GraphState


class CowNetSupervisor(BaseModel):
    next: Literal[
        "data_loader_agent",
        "sna_agent",
        "response_agent",
        "simulation_agent",
        "research_agent",
        "report_agent",
    ] = Field(
        description=(
            "Determines which specialist agent to activate next in the workflow: "
            "'data_loader_agent' to (re)load or refresh interaction data; "
            "'sna_agent' for computing or updating social network analysis metrics; "
            "'response_agent' for farmer-friendly responses and explanations; "
            "'simulation_agent' for what-if network scenarios; "
            "'research_agent' for consulting rules/intents and combining them with metrics; "
            "'report_agent' for generating a farmer-friendly PDF briefing."
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
        "response_agent",
        "simulation_agent",
        "research_agent",
        "report_agent",
    ]
]:
    """
    Supervisor node for the CowNet multi-agent system.

    Routes among:
      - data_loader_agent: (re)load interaction data from source files.
      - sna_agent: social network analysis (build/update graph, compute metrics).
      - response_agent: respond or explain metrics/simulations to the farmer.
      - simulation_agent: run what-if graph manipulation scenarios.
      - research_agent: retrieve/apply rules/intents and combine them with metrics.
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0,
    )

    system_prompt = """
You are the Supervisor for the CowNet multi-agent dairy herd decision support system.
Your mission: autonomously orchestrate agents to fulfill the user’s request end-to-end without asking clarifying questions. Use semantic understanding, intent detection, and reasoning over state. Prefer action with reasonable defaults over clarification.

Agents and roles:
- data_loader_agent: Load/refresh interaction data into state['interactions'].
- sna_agent: Build/update the social network (state['sna_graph']) and compute metrics (state['sna_metrics']).
- simulation_agent: Run what‑if scenarios (e.g., remove a cow); update state['simulation_graph'], state['simulation_metrics'].
- research_agent: Produce evidence-based synthesis into state['research'] (rules/intents + external research tools when available).
- report_agent: Generate a farmer‑friendly PDF report using SNA + research (+ optional simulation).
- response_agent: Provide the final concise answer to the user’s question.

Accessible state:
- messages: conversation history
- interactions: raw interaction records
- sna_graph, sna_metrics
- simulation_graph, simulation_metrics
- research

Non-clarification policy:
- Do NOT ask the user to choose options or confirm parameters.
- If details are unspecified, assume sensible defaults and proceed.
- Only stop at the final step (response or report) unless a hard prerequisite is missing and must be computed first.

Planning and routing policy:
1) Establish prerequisites in the shortest path to fulfill the request.
   - If interactions missing/outdated → data_loader_agent.
   - If network/metrics required and missing/outdated → sna_agent.
   - If what‑if scenario requested → simulation_agent (ensure sna_graph exists first).
   - If evidence/justification requested or report needed → research_agent before report_agent.
2) For “report/briefing/PDF/summary” intents:
   - Ensure sna_metrics and research exist; if missing, run the missing prerequisites (data_loader_agent → sna_agent → research_agent) then route to report_agent.
3) For direct Q&A (“explain”, “recommend”, “summarize”, “answer”):
   - If metrics/research help produce a better answer and are missing, compute them first; otherwise route to response_agent.
4) Avoid redundant work:
   - Don’t reload data if interactions exist and there’s no hint of updates.
   - Don’t recompute SNA unless interactions changed or the user implies freshness.
   - Don’t run simulation without a clear hypothetical/change request.
5) Ambiguity handling:
   - When intent is unclear, pick the next action that most increases useful information toward a likely goal (typically sna_agent if no metrics; otherwise response_agent).
6) Output:
   - Return:
     * next: one of ["data_loader_agent","sna_agent","response_agent","simulation_agent","research_agent","report_agent"]
     * reason: brief justification citing user intent and missing/present state.

Semantic examples (not keyword-matching; use meaning and context):
- “Brief me with a report.” → If no metrics/research: data_loader_agent → sna_agent → research_agent → report_agent (one step per loop). If metrics+research present → report_agent.
- “What if we remove cow T02?” → simulation_agent (ensure sna_graph exists; if not, run sna_agent first).
- “Compute latest herd metrics.” → sna_agent (load data first if missing).
- “Explain why isolation risk is high.” → If metrics missing → sna_agent, then research_agent; finally response_agent.
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
