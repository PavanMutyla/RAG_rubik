from langgraph.graph import START, END, StateGraph
from langchain_openai import OpenAIEmbeddings
from RAG.chains import simple_chain
from langchain_core.messages import BaseMessage
from typing import TypedDict, Optional, Dict, List, Union
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

    response = simple_chain.invoke({
        'user_data':user_data,
        'query':query,
        'allocations':allocations,
        'data':docs,
        'chat_history':memory
    })
    #response = re.sub(
     ####flags=re.DOTALL
    #)

    return {
        'query':query,
        'result':response
    }

workflow = StateGraph(Graph)
workflow.add_node('generate', generate)
workflow.add_edge(START, 'generate')
workflow.add_edge('generate', END)

graph = workflow.compile()