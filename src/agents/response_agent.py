from typing import Sequence, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.graph import END

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.state import GraphState

def response_agent_node(state: GraphState) -> Command:
    """
    Response agent node.

    Formulates an appropriate final response to the user's initial query
    using all available data from other agents, adapting format to query type.
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
    )

    # Unpack state
    messages: Sequence[BaseMessage] = state.get("messages", [])
    sna_metrics: Dict[str, Any] | None = state.get("sna_metrics")
    simulation_graph: Dict[str, Any] | None = state.get("simulation_graph")
    simulation_metrics: Dict[str, Any] | None = state.get("simulation_metrics")
    research_text: str | None = state.get("research")

    # Build context string for the response agent
    context_parts = []
    if sna_metrics:
        context_parts.append("SNA METRICS available: per-cow risk scores, centrality, herd summaries")
    if simulation_metrics:
        context_parts.append("SIMULATION RESULTS available: network changes after cow removal")
    if research_text:
        context_parts.append("RESEARCH FINDINGS available: rule-based insights + academic references")
    if not context_parts:
        context_parts.append("No metrics/research available - use conversation history only")

    state_context_summary = " | ".join(context_parts)

    system_prompt = f"""
You are the Response Agent in the CowNet multi-agent system.

Your job: Provide the FINAL answer to the user's original question using all available information.

RULES:
- Answer the USER'S INITIAL QUERY directly and completely
- Use appropriate format for the query type:
  * Simple questions → Direct answer
  * Complex analysis → Organized bullets/sections  
  * Out-of-scope topics → Polite redirect to CowNet capabilities
- Be concise, farmer-friendly, and actionable
- Reference available data: {state_context_summary}
- No clarifications; proceed with reasonable defaults. Do not ask followup questions.

DETERMINE QUERY TYPE:
1. Simple/direct → Straight answer
2. Analysis/risks → Bulleted key findings + recommendations  
3. Out-of-scope → "CowNet focuses on social network analysis. For [topic], consult [resource]."

Always answer the actual user question using whatever data is available.
"""

    response_messages: list[BaseMessage] = [
        AIMessage(content=system_prompt),
    ] + list(messages)

    final_response = llm.invoke(response_messages)

    return Command(
        update={
            "messages": list(messages) + [
                AIMessage(
                    content=final_response.content,
                    name="response_agent",
                )
            ]
        },
        goto=END,
    )
