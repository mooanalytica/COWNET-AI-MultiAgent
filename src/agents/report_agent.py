from typing import Any, List, Dict, Tuple

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from .base_agent import BaseAgent
from llm.language_models import LanguageModelManager
from config import WORKING_DIRECTORY
from core.state import GraphState

from ..tools.report_tools import markdown_to_pdf


class ReportAgent(BaseAgent):
	"""Agent responsible for generating a farmer-friendly PDF briefing.

	The agent will:
	- Read `sna_metrics`, `research`, and optional `simulation_metrics` from state
	- Compose a concise, farmer-friendly markdown report
	- Call the `markdown_to_pdf` tool to create and save a PDF
	- Return a summary message with the saved path
	"""

	def __init__(
		self,
		language_model_manager: LanguageModelManager,
		team_members: List[str],
		working_directory: str = WORKING_DIRECTORY,
	):
		super().__init__(
			agent_name="report_agent",
			language_model_manager=language_model_manager,
			team_members=team_members,
			working_directory=working_directory,
			response_format=None,
		)

	def _get_system_prompt(self) -> str:
		return (
			"SYSTEM_PROMPT:"
			"You are the Report Agent."
			" Generate a farmer-friendly markdown briefing using available state. Assume farmers know nothing about social network analysis"
            " Use language understandable to anyone in the dairy farming field"
			" SNA metrics, research findings, and optional simulation results."
			" Keep language simple and actionable."
			" Structure: # Title, ## Executive Summary, ## Key Insights,"
			" ## Actionable Recommendations, ## Additional Notes."
            " Always include a reminder to consult professionals and experts before taking action"
            " Ensure the report is no longer than one page"
            " Cite any external resources used to obtain information (mainly from research knowledge) at the end."
			" After composing the markdown, call the `markdown_to_pdf` tool with"
			" the markdown and a reasonable filename (e.g., cownet_briefing.pdf)."
			" Return to the workflow with a short success message."
		)

	def _get_tools(self) -> List:
		return [markdown_to_pdf]


# Instantiate a shared agent instance
report_agent = ReportAgent(
	language_model_manager=LanguageModelManager(),
	team_members=[
		"supervisor",
		"sna_agent",
		"research_agent",
		"simulation_agent",
		"response_agent",
	],
)


def _compose_markdown_from_state(state: GraphState) -> str:
	"""Compose a minimal markdown report from the state as fallback.

	Used if the LLM does not produce the markdown automatically. Ensures the
	tool can still run and produce a reasonable PDF.
	"""
	lines: List[str] = []
	lines.append("# CowNet Herd Health Briefing")
	lines.append("## Executive Summary")
	lines.append("Your herd shows stable connectivity with identified risk signals.")
	lines.append("")
	lines.append("## Key Network Insights")
	sna = state.get("sna_metrics") or {}
	herd = sna.get("herd_metrics", {})
	lines.append(f"- Herd size: {herd.get('num_cows', 'N/A')}")
	lines.append(f"- Network density: {herd.get('density', 'N/A')}")
	lines.append(f"- Average degree: {herd.get('avg_degree', 'N/A')}")
	lines.append("")
	lines.append("## High Risk Cows")
	top = sna.get("top_risk_cows", [])
	if isinstance(top, list) and top:
		for entry in top:
			cid = entry.get("cow_id", "unknown")
			cr = entry.get("conflict_risk", 0)
			ir = entry.get("isolation_risk", 0)
			lines.append(f"- {cid}: conflict={cr:.2f}, isolation={ir:.2f}")
	else:
		lines.append("- No high-risk cows identified.")
	lines.append("")
	lines.append("## Actionable Recommendations")
	lines.append("1. Monitor high-risk cows and adjust grouping to reduce conflicts.")
	lines.append("2. Reinforce social reintegration for isolated cows.")
	lines.append("")
	lines.append("## Additional Notes")
	research = state.get("research")
	if research:
		lines.append("Research findings:")
		lines.append(research)
	sim = state.get("simulation_metrics")
	if sim:
		lines.append("Simulation summary:")
		lines.append("- Simulation results available.")

	return "\n".join(lines)


def report_agent_node(state: GraphState) -> Command:
	"""Node that drives the report agent to produce a PDF and end the workflow.

	Preconditions:
	- Expect `sna_metrics` and `research` in state for a meaningful report.
	"""
	if not state.get("sna_metrics") or not state.get("research"):
		return Command(
			update={
				"messages": [
					AIMessage(
						content=(
							"Report Agent: Missing SNA metrics or research. "
							"Please run SNA and Research agents first."
						),
						name="report_agent",
					)
				]
			},
			goto="supervisor",
		)

	# Let the LLM agent run; it should call markdown_to_pdf.
	response = report_agent.invoke(state)

	# Extract ToolMessage produced by `markdown_to_pdf` if present.
	pdf_summary = "Report generated."
	pdf_path = None

	for msg in response.get("messages", []):
		if isinstance(msg, ToolMessage):
			artifact = getattr(msg, "artifact", None) or {}
			if "path" in artifact:
				pdf_path = artifact.get("path")
				pdf_summary = f"✅ Report generated: {artifact.get('filename')}\n📁 Saved: {pdf_path}"
				break

	# Fallback: if tool wasn't called, compose markdown and call tool directly.
	if pdf_path is None:
		fallback_md = _compose_markdown_from_state(state)
		content, artifact = markdown_to_pdf.invoke({
			"markdown": fallback_md,
			"filename": "cownet_briefing.pdf",
			"title": "CowNet Report",
		})
		pdf_path = artifact.get("path")
		pdf_summary = content if content else pdf_summary

	updates = {
		"messages": [AIMessage(content=pdf_summary, name="report_agent")],
	}

	return Command(update=updates, goto="supervisor")

