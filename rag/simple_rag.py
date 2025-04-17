from langgraph.graph import START, END, StateGraph
from langchain_openai import OpenAIEmbeddings
from RAG.chains import simple_chain
from langchain_core.messages import BaseMessage
from typing import TypedDict, Optional, Dict, List, Union

from RAG.tools import json_to_table, goal_feasibility
import re
from dotenv import load_dotenv
load_dotenv()


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=384  
)

class Graph(TypedDict):
    user_data : Optional[Dict]
    allocations : Optional[Dict]
    query : str
    result : Optional[str]
    pdf: Optional[Union[str, bytes]]
    memory : List[BaseMessage]

def generate(state):
    user_data = state['user_data']
    allocations = state['allocations']
    query = state['query']
    docs = state['pdf']
    memory = state['memory']

    print("Invoking simple_chain...")  # Debugging line
    response = simple_chain.invoke({
        'user_data':user_data,
        'query':query,
        'allocations':allocations,
        'data':docs,
        'chat_history':memory
    })
    print(f"simple_chain response: {response}")  # Debugging line

    return {
        'query':query,
        'result':response
    }

workflow = StateGraph(Graph)
workflow.add_node('generate', generate)
workflow.add_edge(START, 'generate')
workflow.add_edge('generate', END)


graph = workflow.compile()

inputs = {
                    "query": "I want to buy a car of 12L within a year",
                    "user_data": None,
                    "allocations": None,
                    "pdf":None,
                    "memory": [""]
                }
#print(graph.invoke(inputs))
