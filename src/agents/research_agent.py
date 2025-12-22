from typing import List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.types import Command
from semanticscholar import SemanticScholar
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools
from ..llm.language_models import LanguageModelManager
from ..agents.base_agent import BaseAgent
from config import WORKING_DIRECTORY
from core.state import GraphState

INTENTS = {
  "rules": [
    {
      "id": 1,
      "intent": "top_conflict_risk_cows",
      "action": "List top 5 cows in Pen X with highest conflict risk.",
      "required_metrics": [
        "conflict_risk_score"
      ],
      "metric_derivation": "Derived from proximity logs, behavior labels, frequency and duration of negative/aggressive interactions. Supported by automated behavior classification and SNA research.",
      "recommended_action": "Flag top cows for monitoring or intervention.",
      "action_justification": "Early identification reduces stress and injury risk."
    },
    {
      "id": 2,
      "intent": "isolation_increase_from_last_week",
      "action": "List cows whose average distance from herd increased week-over-week.",
      "required_metrics": [
        "weekly_isolation_score"
      ],
      "metric_derivation": "Derived from UWB location data, mean/min/max herd distance, and SNA edge density. Isolation correlates with welfare decline.",
      "recommended_action": "Flag cows for welfare check or social reintegration.",
      "action_justification": "Prevents welfare decline and enables early intervention."
    },
    {
      "id": 3,
      "intent": "communities_and_dominant_hubs",
      "action": "Detect communities and hubs; identify dominant hubs.",
      "required_metrics": [
        "community_membership",
        "centrality_scores"
      ],
      "metric_derivation": "Derived from interaction graphs using Louvain/clique detection and centrality metrics.",
      "recommended_action": "Highlight dominant hubs for monitoring or management.",
      "action_justification": "Central cows influence group dynamics and disease spread."
    },
    {
      "id": 8,
      "intent": "top_5_central_cows",
      "action": "List cows with highest centrality and explain their roles.",
      "required_metrics": [
        "degree_centrality",
        "betweenness_centrality",
        "eigenvector_centrality"
      ],
      "metric_derivation": "Standard SNA centrality measures.",
      "recommended_action": "Flag for monitoring or targeted management.",
      "action_justification": "Central cows influence group stability and disease risk."
    },
    {
      "id": 9,
      "intent": "new_bridges_between_communities",
      "action": "Identify new bridge cows and compare to previous week.",
      "required_metrics": [
        "betweenness_centrality",
        "edge_bridge_score"
      ],
      "metric_derivation": "SNA bridge detection identifies connectors between communities.",
      "recommended_action": "Monitor bridge cows for welfare and biosecurity.",
      "action_justification": "Bridges facilitate information and disease flow."
    },
    {
      "id": 10,
      "intent": "persistent_high_conflict_risk",
      "action": "Flag cows with high conflict risk over multiple weeks.",
      "required_metrics": [
        "historical_conflict_risk_scores"
      ],
      "metric_derivation": "Longitudinal SNA and behavior logs.",
      "recommended_action": "Recommend intervention or management change.",
      "action_justification": "Persistent issues predict injury or stress."
    },
    {
      "id": 11,
      "intent": "isolated_high_risk_lactation",
      "action": "Flag isolated cows at risk given lactation stage.",
      "required_metrics": [
        "isolation_score",
        "lactation_stage",
        "lactation_risk_threshold"
      ],
      "metric_derivation": "Derived from registry, SNA, and milk data.",
      "recommended_action": "Monitor or reintegrate socially.",
      "action_justification": "Supports welfare and productivity during vulnerable stages."
    },
    {
      "id": 15,
      "intent": "feeding_congestion_risk",
      "action": "Compute and flag congestion risk at feeding times.",
      "required_metrics": [
        "cows_feeding_together",
        "dominant_cow_presence",
        "congestion_risk_equation"
      ],
      "metric_derivation": "Derived from UWB, head direction, and behavior data.",
      "recommended_action": "Suggest management to reduce congestion.",
      "action_justification": "Reduces aggression and improves welfare."
    },
    {
      "id": 16,
      "intent": "abrupt_centrality_drop",
      "action": "Flag cows with significant centrality decrease.",
      "required_metrics": [
        "centrality_change",
        "connectedness",
        "isolation_risk"
      ],
      "metric_derivation": "SNA week-over-week comparison.",
      "recommended_action": "Recommend welfare check or intervention.",
      "action_justification": "Early detection prevents social or health decline."
    },
  ]
}


class ResearchAgent(BaseAgent):
    """Agent responsible for gathering and summarizing research information."""

    def __init__(self, language_model_manager: LanguageModelManager, team_members: List[str], working_directory: str = WORKING_DIRECTORY):
        """
        Initialize the SearchAgent.

        Args:
            language_model_manager: Manager for language model configuration.
            team_members: List of team member roles for collaboration.
            working_directory: The directory where the agent's data will be stored.
        """
        super().__init__(
            agent_name="research_agent",
            language_model_manager=language_model_manager,
            team_members=team_members,
            working_directory=working_directory
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for information retrieval and summarization."""
        return f'''
SYSTEM PROMPT:
You are the Research Agent. Act autonomously and do not ask for clarifications.
Objective: Produce evidence-based synthesis into state["research"] supporting CowNet decisions.

Capabilities:
- FIXED INTENTS: {INTENTS} — use as foundational, structured domain rules.
- TOOLS: SemanticScholar, PubMed, arXiv — fetch peer-reviewed or respected sources.

Protocol:
1) First, match relevant FIXED INTENTS to the user’s need or current state.
2) Immediately query tools to supplement with external evidence (titles, authors, venues).
3) Synthesize: [Intent findings] → [External research] → [Actionable synthesis].
4) Cite compactly (Title; Authors; Venue, Year). Prefer veterinary/agriculture sources.

Rules:
- No clarifications; proceed with reasonable defaults.
- Be concise, farmer-friendly, and directly relevant to dairy herd social networks.
- If tools return nothing, state that and rely on intents + sound reasoning.
- Output plain text, suitable for downstream report/response agents.

CONSTRAINTS:
- Focus on EVIDENCE-BASED insights relevant to dairy cow social networks
- Structure responses: [Fixed Intent Findings] → [External Research] → [Synthesis]
- When tools return no results, explicitly state this and rely on intents + domain reasoning
        '''

    def _get_tools(self):
        """Get the list of tools for information retrieval and summarization."""
        return [SemanticScholarQueryRun(api_wrapper=SemanticScholarAPIWrapper())] + load_tools(["arxiv"]) + load_tools(["pubmed"])
      
research_agent = ResearchAgent(
    language_model_manager=LanguageModelManager(),
    team_members=["supervisor", "sna_agent", "simulation_agent", "response_agent"]
)

def research_node(state: GraphState) -> Command:
    """Extract research information from the GraphState.

    Args:
        state: The current GraphState containing research data.
    """
    
    response = research_agent.invoke(state)
    
    message = response.get("messages")[-1]
    output = message.content
    
    updates = {
        "messages": [AIMessage(content=output)],
        "research": output
    }
    
    return Command(
      update=updates,
      goto="supervisor"
    )