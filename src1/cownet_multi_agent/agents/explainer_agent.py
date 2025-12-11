from typing import Sequence, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from ..state import GraphState  # your GraphState model


def explainer_agent_node(state: GraphState) -> Command:
    """
    Explainer agent node.

    Takes existing SNA metrics, simulation metrics, and research text from the
    GraphState and generates a farmer-friendly explanation and answer.
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.2,
    )

    # Unpack state
    messages: Sequence[BaseMessage] = state.get("messages", [])
    sna_metrics: Dict[str, Any] | None = state.get("sna_metrics")
    simulation_metrics: Dict[str, Any] | None = state.get("simulation_metrics")
    research_text: str | None = state.get("research")

    # Build context string for the explainer
    context_parts = []
    if sna_metrics:
        context_parts.append(
            "SNA metrics (per cow and/or herd) are available in 'sna_metrics'. "
            "They may include centralities, risk scores, or herd summaries."
        )
    if simulation_metrics:
        context_parts.append(
            "Simulation results are available in 'simulation_metrics', describing "
            "before/after network metrics when specific cows are removed."
        )
    if research_text:
        context_parts.append(
            "Rule-based or research interpretation is available in 'research', "
            "summarizing how formal rules/intents apply to the current situation."
        )
    if not context_parts:
        context_parts.append(
            "No structured metrics or research text are present in the state; "
            "base your explanation only on the conversation so far."
        )

    state_context_summary = "\n".join(context_parts)

    system_prompt = f"""
You are the Explainer Agent in the CowNet multi-agent system.

Your role:
- Read the conversation so far and any available analysis in the shared state.
- Turn technical outputs into clear, farmer-friendly explanations and recommendations.
- Use SNA metrics, simulation metrics, and rule-based research when available.

State context:
{state_context_summary}

Guidelines:
- Do not invent new metrics or rules; only interpret what is already computed or described.
- If SNA metrics exist, explain what they mean for individual cows and the herd.
- If simulation results exist, explain how the 'after' network differs from 'before'
  and what that implies for welfare or management decisions.
- If research text exists, ground your explanation in those rules/intents.
- Use short paragraphs and plain language suitable for a dairy farmer.
"""

    explainer_messages: list[BaseMessage] = [
        AIMessage(content=system_prompt),
    ] + list(messages)

    explanation = llm.invoke(explainer_messages)

    return Command(
        update={
            "messages": list(messages) + [
                AIMessage(
                    content=explanation.content,
                    name="explainer_agent",
                )
            ]
        },
        goto=END,
    )
