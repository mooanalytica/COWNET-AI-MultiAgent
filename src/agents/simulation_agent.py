from typing import List
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from ..llm.language_models import LanguageModelManager
from ..agents.base_agent import BaseAgent
from config import WORKING_DIRECTORY
from core.state import GraphState
from ..tools.simulation_tools import remove_cow_from_network, _remove_cow_from_network

class SimulationAgent(BaseAgent):
    """Agent responsible for running social network simulations by removing cows."""

    def __init__(self, language_model_manager: LanguageModelManager, team_members: List[str], working_directory: str = WORKING_DIRECTORY):
        """
        Initialize the SimulationAgent.

        Args:
            language_model_manager: Manager for language model configuration.
            team_members: List of team member roles for collaboration.
            working_directory: The directory where the agent's data will be stored.
        """
        super().__init__(
            agent_name="simulation_agent",
            language_model_manager=language_model_manager,
            team_members=team_members,
            working_directory=working_directory
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for simulation tasks."""
        return f'''
SYSTEM PROMPT:
You are the simulation agent specializing in dairy cow social network "what-if" analysis.

Your ONLY job: Use the remove_cow_from_network tool to simulate removing ONE specific cow from the current network.

WORKFLOW:
1. Review conversation history and SNA metrics to identify the target cow
2. IMMEDIATELY call remove_cow_from_network(cow_id="COW_ID") with a valid cow from sna_graph
3. The tool returns modified_graph + modified_metrics artifacts automatically
4. Summarize the simulation impact in your final response

RULES:
- ONLY use remove_cow_from_network tool (no other actions/tools)
- Select ONE cow_id based on: high conflict risk, isolation risk, or supervisor instruction
- If cow_id invalid, tool will report error - select different cow
- ALWAYS complete simulation and return to supervisor

Focus on high-risk cows from prior SNA analysis or explicit supervisor instructions.
'''

    def _get_tools(self):
        """Get the simulation tool."""
        return [remove_cow_from_network]
     
simulation_agent = SimulationAgent(
    language_model_manager=LanguageModelManager(),
    team_members=["supervisor", "sna_agent", "research_agent", "response_agent"]
)

def simulation_node(state: GraphState) -> Command:
    """Run simulation agent and extract content/artifact from tool message.

    Args:
        state: The current GraphState containing messages and sna_graph.
    """
    
    response = simulation_agent.invoke(state)
    
    # Extract all new agent messages
    agent_messages = response.get("messages", [])
    
    # Find the ToolMessage with content_and_artifact from remove_cow_from_network
    simulation_content = ""
    simulation_graph = None
    simulation_metrics = None
    
    for msg in agent_messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
            cow_id = msg.content
            break
        
    if cow_id:
        simulation_content, artifacts = _remove_cow_from_network(
            cow_id=cow_id,
            sna_graph=state.get("sna_graph")
        )
        simulation_graph = artifacts.get("modified_graph")
        simulation_metrics = artifacts.get("modified_metrics")
    
    # Prepare updates with extracted artifacts matching GraphState schema
    updates = {
        "messages": [AIMessage(content=simulation_content or "Simulation completed.", name="simulation_agent")],
        "simulation_graph": simulation_graph,
        "simulation_metrics": simulation_metrics
    }
    
    return Command(
        update=updates,
        goto="supervisor"
    )
