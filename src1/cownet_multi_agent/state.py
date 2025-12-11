from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired
import pandas as pd
import networkx as nx

class GraphState(TypedDict):
    
    messages = Annotated[Sequence[BaseMessage], add_messages]
    
    interactions: Optional[dict]
    
    sna_graph: Optional[dict]
    sna_metrics: Optional[dict]
    
    research: Optional[str]
    
    simulation_graph: Optional[dict]
    simulation_metrics: Optional[dict]