from typing import Sequence, Dict, Any, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from mem0 import Memory

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.state import GraphState
from logger import get_logger

logger = get_logger(__name__)


class CowNetMemory:
    """
    Long-term memory layer for CowNet using mem0 with pgvector database.
    Provides persistent memory storage and retrieval for contextual responses.
    """
    
    _instance: Optional["CowNetMemory"] = None
    _memory: Optional[Memory] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single memory instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize mem0 memory with pgvector configuration."""
        if self._memory is not None:
            return
            
        # pgvector configuration from environment variables
        pg_host = os.getenv("PGVECTOR_HOST", "localhost")
        pg_port = os.getenv("PGVECTOR_PORT", "5432")
        pg_user = os.getenv("PGVECTOR_USER", "postgres")
        pg_password = os.getenv("PGVECTOR_PASSWORD", "postgres")
        pg_database = os.getenv("PGVECTOR_DATABASE", "cownet_memory")
        
        config = {
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "host": pg_host,
                    "port": int(pg_port),
                    "user": pg_user,
                    "password": pg_password,
                    "dbname": pg_database,
                    "embedding_model_dims": 1536,
                    "collection_name": "cownet_memories",
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "embedding_dims": 1536,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                }
            },
            "version": "v1.1"
        }
        
        try:
            self._memory = Memory.from_config(config)
            logger.info("CowNet long-term memory initialized with pgvector")
        except Exception as e:
            logger.error(f"Failed to initialize mem0 memory: {e}")
            self._memory = None
    
    @property
    def is_available(self) -> bool:
        """Check if memory is available."""
        return self._memory is not None
    
    def search(self, query: str, user_id: str = "cownet_user", limit: int = 5) -> list[Dict[str, Any]]:
        """
        Search for relevant memories based on query.
        
        Args:
            query: The search query string
            user_id: User identifier for memory isolation
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory entries
        """
        if not self.is_available:
            logger.warning("Memory not available, skipping search")
            return []
        
        try:
            results = self._memory.search(query=query, user_id=user_id, limit=limit)
            logger.info(f"Retrieved {len(results.get('results', []))} memories for query")
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def add(self, messages: list[Dict[str, str]], user_id: str = "cownet_user", metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Add a conversation to long-term memory.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            user_id: User identifier for memory isolation
            metadata: Optional metadata to attach to the memory
            
        Returns:
            Memory addition result or None if failed
        """
        if not self.is_available:
            logger.warning("Memory not available, skipping add")
            return None
        
        try:
            result = self._memory.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata or {}
            )
            logger.info(f"Added conversation to long-term memory")
            return result
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return None
    
    def get_all(self, user_id: str = "cownet_user") -> list[Dict[str, Any]]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of all memory entries for the user
        """
        if not self.is_available:
            return []
        
        try:
            results = self._memory.get_all(user_id=user_id)
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []


def _get_user_query(messages: Sequence[BaseMessage]) -> str:
    """Extract the original user query from messages."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _format_memory_context(memories: list[Dict[str, Any]]) -> str:
    """Format retrieved memories into context string for the prompt."""
    if not memories:
        return ""
    
    memory_texts = []
    for i, mem in enumerate(memories, 1):
        memory_content = mem.get("memory", "")
        if memory_content:
            memory_texts.append(f"{i}. {memory_content}")
    
    if not memory_texts:
        return ""
    
    return "RELEVANT PAST INTERACTIONS:\n" + "\n".join(memory_texts)


# Initialize singleton memory instance
_cownet_memory = CowNetMemory()


def response_agent_node(state: GraphState) -> Command:
    """
    Response agent node with long-term memory integration.

    Formulates an appropriate final response to the user's initial query
    using all available data from other agents, adapting format to query type.
    Leverages mem0 long-term memory with pgvector for contextual awareness.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # Unpack state
    messages: Sequence[BaseMessage] = state.get("messages", [])
    sna_metrics: Dict[str, Any] | None = state.get("sna_metrics")
    simulation_graph: Dict[str, Any] | None = state.get("simulation_graph")
    simulation_metrics: Dict[str, Any] | None = state.get("simulation_metrics")
    research_text: str | None = state.get("research")

    # Extract user query for memory operations
    user_query = _get_user_query(messages)
    
    # Search long-term memory for relevant context
    memory_context = ""
    if user_query and _cownet_memory.is_available:
        retrieved_memories = _cownet_memory.search(
            query=user_query,
            user_id="cownet_user",
            limit=5
        )
        memory_context = _format_memory_context(retrieved_memories)
        if memory_context:
            logger.info(f"Retrieved {len(retrieved_memories)} relevant memories for context")

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

    # Build system prompt with memory context
    memory_section = ""
    if memory_context:
        memory_section = f"""
LONG-TERM MEMORY CONTEXT:
{memory_context}

Use the above past interactions to provide more personalized and contextually aware responses when relevant.
"""

    system_prompt = f"""
You are the Response Agent in the CowNet multi-agent system.

Your job: Provide the FINAL answer to the user's original question using all available information.
{memory_section}
RULES:
- Answer the USER'S INITIAL QUERY directly and completely
- Use appropriate format for the query type:
  * Simple questions → Direct answer
  * Complex analysis → Organized bullets/sections  
  * Out-of-scope topics → Polite redirect to CowNet capabilities
- Be concise, farmer-friendly, and actionable
- Reference available data: {state_context_summary}
- No clarifications; proceed with reasonable defaults. Do not ask followup questions.
- If relevant past interactions are available, use them to provide continuity and context.

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
    
    # Add the conversation to long-term memory
    if user_query and _cownet_memory.is_available:
        conversation_for_memory = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": final_response.content}
        ]
        
        # Add metadata for better memory organization
        metadata = {
            "agent": "response_agent",
            "has_sna_metrics": sna_metrics is not None,
            "has_simulation": simulation_metrics is not None,
            "has_research": research_text is not None,
        }
        
        _cownet_memory.add(
            messages=conversation_for_memory,
            user_id="cownet_user",
            metadata=metadata
        )
        logger.info("Stored conversation in long-term memory")

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
