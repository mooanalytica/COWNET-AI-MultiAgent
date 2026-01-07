import os
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from psycopg_pool import AsyncConnectionPool

from core.state import GraphState
from ..agents.supervisor import cownet_supervisor_node
from ..agents.data_loader import data_loader_node
from ..agents.sna_agent import sna_node
from ..agents.simulation_agent import simulation_node
from ..agents.response_agent import response_agent_node
from ..agents.research_agent import research_node
from ..agents.report_agent import report_agent_node
from ..logger import get_logger

logger = get_logger(__name__)


class CowNetCheckpointerConfig:
    """
    Configuration for async PostgreSQL checkpointer with user and thread management.
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        """
        Initialize checkpointer configuration.
        
        Args:
            user_id: Unique identifier for the user. Auto-generated if not provided.
            thread_id: Unique identifier for the conversation thread. Auto-generated if not provided.
        """
        self.user_id = user_id or str(uuid.uuid4())
        self.thread_id = thread_id or str(uuid.uuid4())
    
    @property
    def config(self) -> dict:
        """Get the configuration dictionary for graph invocation."""
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "user_id": self.user_id,
            }
        }
    
    def new_thread(self) -> "CowNetCheckpointerConfig":
        """Create a new configuration with a fresh thread_id but same user_id."""
        return CowNetCheckpointerConfig(
            user_id=self.user_id,
            thread_id=str(uuid.uuid4())
        )


def _get_postgres_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    host = os.getenv("PGVECTOR_HOST", "localhost")
    port = os.getenv("PGVECTOR_PORT", "5432")
    user = os.getenv("PGVECTOR_USER", "postgres")
    password = os.getenv("PGVECTOR_PASSWORD", "postgres")
    database = os.getenv("PGVECTOR_DATABASE", "cownet_memory")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


# Global connection pool for async checkpointer
_connection_pool: Optional[AsyncConnectionPool] = None


async def get_connection_pool() -> AsyncConnectionPool:
    """Get or create the async connection pool."""
    global _connection_pool
    if _connection_pool is None:
        connection_string = _get_postgres_connection_string()
        _connection_pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=10,
            min_size=1,
        )
        await _connection_pool.open()
        logger.info("Created async PostgreSQL connection pool for checkpointer")
    return _connection_pool


async def close_connection_pool():
    """Close the connection pool gracefully."""
    global _connection_pool
    if _connection_pool is not None:
        await _connection_pool.close()
        _connection_pool = None
        logger.info("Closed async PostgreSQL connection pool")


@asynccontextmanager
async def get_async_checkpointer():
    """
    Async context manager for PostgreSQL checkpointer.
    
    Yields:
        AsyncPostgresSaver: Configured async checkpointer instance
    """
    pool = await get_connection_pool()
    checkpointer = AsyncPostgresSaver(pool)
    
    # Setup the checkpointer tables if they don't exist
    await checkpointer.setup()
    
    try:
        yield checkpointer
    finally:
        # Pool is managed globally, not closed here
        pass


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


def compile_graph_with_memory_saver():
    """
    Compile graph with in-memory checkpointer (for testing/development).
    
    Returns:
        Compiled graph with MemorySaver checkpointer
    """
    memory_saver = MemorySaver()
    return build_cownet_workflow().compile(checkpointer=memory_saver)


async def compile_graph_with_postgres_checkpointer():
    """
    Compile graph with async PostgreSQL checkpointer for production use.
    
    Returns:
        Compiled graph with AsyncPostgresSaver checkpointer
    """
    async with get_async_checkpointer() as checkpointer:
        compiled_graph = build_cownet_workflow().compile(checkpointer=checkpointer)
        return compiled_graph


async def run_workflow_async(
    messages: list,
    config: Optional[CowNetCheckpointerConfig] = None,
):
    """
    Run the CowNet workflow asynchronously with PostgreSQL checkpointing.
    
    Args:
        messages: List of messages to process
        config: Checkpointer configuration with user_id and thread_id.
                If not provided, creates new config with auto-generated IDs.
    
    Returns:
        The final state from the workflow execution
    """
    if config is None:
        config = CowNetCheckpointerConfig()
    
    async with get_async_checkpointer() as checkpointer:
        compiled_graph = build_cownet_workflow().compile(checkpointer=checkpointer)
        
        initial_state = {
            "messages": messages,
        }
        
        result = await compiled_graph.ainvoke(
            initial_state,
            config=config.config
        )
        
        logger.info(
            f"Workflow completed for user_id={config.user_id}, "
            f"thread_id={config.thread_id}"
        )
        
        return result


async def get_thread_history(
    config: CowNetCheckpointerConfig,
) -> list:
    """
    Retrieve conversation history for a specific thread.
    
    Args:
        config: Checkpointer configuration with thread_id
    
    Returns:
        List of state checkpoints for the thread
    """
    async with get_async_checkpointer() as checkpointer:
        history = []
        async for checkpoint in checkpointer.alist(config.config):
            history.append(checkpoint)
        return history


# Default compiled graph (with in-memory saver for backward compatibility)
graph = build_cownet_workflow().compile()

