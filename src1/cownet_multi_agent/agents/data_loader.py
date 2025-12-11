from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
import pandas as pd
from ..state import GraphState

def data_loader_node(state: GraphState) -> Command:
    """
    Data loader node for initializing or updating the cow interaction data
    and social network graph in the CowNet multi-agent system.
    """
    interactions_df = pd.read_csv("D:\DrSuresh_Co-op\CowNet_AI_Repo\COWNET-AI-MultiAgent\data")
    interactions_dict = interactions_df.to_dict(orient="records")
    
    return Command(
        update={
            "interactions": interactions_dict,
            "messages": [AIMessage(content="Loaded interaction data")]
        },
        goto="supervisor"
    )