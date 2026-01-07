"""
FastAPI backend for CowNet Multi-Agent System.

Provides REST API endpoints for interacting with the CowNet AI system,
including chat functionality with PostgreSQL checkpointing and session management.
"""

import os
import sys
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.workflow import (
    run_workflow_async,
    CowNetCheckpointerConfig,
    get_thread_history,
    close_connection_pool,
)
from logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message to send to CowNet AI")
    user_id: Optional[str] = Field(None, description="User identifier for session tracking")
    thread_id: Optional[str] = Field(None, description="Thread identifier for conversation continuity")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI response from CowNet")
    user_id: str = Field(..., description="User identifier used for this request")
    thread_id: str = Field(..., description="Thread identifier used for this request")


class MessageHistory(BaseModel):
    """Model for a single message in history."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class ThreadHistoryResponse(BaseModel):
    """Response model for thread history endpoint."""
    user_id: str
    thread_id: str
    messages: List[MessageHistory]


class NewThreadResponse(BaseModel):
    """Response model for creating a new thread."""
    user_id: str
    thread_id: str
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting CowNet API server...")
    yield
    # Shutdown
    logger.info("Shutting down CowNet API server...")
    await close_connection_pool()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="CowNet AI API",
    description="REST API for interacting with the CowNet Multi-Agent System for cattle disease risk analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status and version information.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the CowNet AI system.
    
    This endpoint processes user messages through the multi-agent workflow,
    utilizing PostgreSQL checkpointing for conversation persistence.
    
    - **message**: The user's question or request
    - **user_id**: Optional user identifier (auto-generated if not provided)
    - **thread_id**: Optional thread identifier for conversation continuity
    
    Returns the AI response along with session identifiers.
    """
    try:
        # Create checkpointer config
        config = CowNetCheckpointerConfig(
            user_id=request.user_id,
            thread_id=request.thread_id
        )
        
        # Create message
        messages = [HumanMessage(content=request.message)]
        
        # Run the workflow
        result = await run_workflow_async(
            messages=messages,
            config=config
        )
        
        # Extract the final response
        response_messages = result.get("messages", [])
        if not response_messages:
            raise HTTPException(
                status_code=500,
                detail="No response generated from workflow"
            )
        
        # Get the last AI message
        final_response = ""
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage):
                final_response = msg.content
                break
        
        if not final_response:
            final_response = "I apologize, but I couldn't generate a response. Please try again."
        
        logger.info(f"Chat completed for user={config.user_id}, thread={config.thread_id}")
        
        return ChatResponse(
            response=final_response,
            user_id=config.user_id,
            thread_id=config.thread_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@app.post("/threads/new", response_model=NewThreadResponse, tags=["Threads"])
async def create_new_thread(user_id: Optional[str] = None):
    """
    Create a new conversation thread.
    
    Generates a new thread ID for starting a fresh conversation.
    Optionally accepts a user_id to associate with the thread.
    
    - **user_id**: Optional user identifier (auto-generated if not provided)
    
    Returns the new thread configuration.
    """
    config = CowNetCheckpointerConfig(user_id=user_id)
    
    return NewThreadResponse(
        user_id=config.user_id,
        thread_id=config.thread_id,
        message="New thread created successfully"
    )


@app.get("/threads/{thread_id}/history", response_model=ThreadHistoryResponse, tags=["Threads"])
async def get_conversation_history(
    thread_id: str,
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Retrieve conversation history for a specific thread.
    
    - **thread_id**: The thread identifier to retrieve history for
    - **X-User-ID**: Optional user ID header for validation
    
    Returns the list of messages in the conversation thread.
    """
    try:
        config = CowNetCheckpointerConfig(
            user_id=user_id or "anonymous",
            thread_id=thread_id
        )
        
        history = await get_thread_history(config)
        
        # Extract messages from checkpoints
        messages = []
        for checkpoint in history:
            if checkpoint and "channel_values" in checkpoint:
                channel_messages = checkpoint["channel_values"].get("messages", [])
                for msg in channel_messages:
                    if isinstance(msg, HumanMessage):
                        messages.append(MessageHistory(role="user", content=msg.content))
                    elif isinstance(msg, AIMessage):
                        messages.append(MessageHistory(role="assistant", content=msg.content))
        
        return ThreadHistoryResponse(
            user_id=config.user_id,
            thread_id=thread_id,
            messages=messages
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve thread history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream a response from the CowNet AI system.
    
    This endpoint provides Server-Sent Events (SSE) for streaming responses.
    Currently returns a placeholder - implement with async generator for full streaming.
    """
    # TODO: Implement streaming with async generator and SSE
    raise HTTPException(
        status_code=501,
        detail="Streaming endpoint not yet implemented. Use /chat for synchronous responses."
    )


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
