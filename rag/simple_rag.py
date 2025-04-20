from langgraph.graph import START, END, StateGraph
from langchain_openai import OpenAIEmbeddings
from RAG.chains import simple_chain, llm_with_tools
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import TypedDict, Optional, Dict, List, Union, Annotated
from langchain_core.messages import AnyMessage #human or AI message
from langgraph.graph.message import add_messages # reducer in langgraph 
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langgraph.checkpoint.memory import MemorySaver
import json
from RAG.tools import json_to_table, goal_feasibility
import re

from dotenv import load_dotenv
load_dotenv()

memory = MemorySaver()
config = {"thread_id":"sample"}
tools = [json_to_table, goal_feasibility]
#tool_executor = ToolExecutor([json_to_table, goal_feasibility])

class Graph(TypedDict):
    query: Annotated[list[AnyMessage], add_messages]
    user_data : Dict
    allocations : Dict 
    data : str 
    output : Dict
def tools_condition(state):
    last_message = state["query"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END

def chat(state):
    inputs = {
        "query": state['query'],
        "user_data": state['user_data'],
        "allocations": state['allocations'],
        "data": state['data']
    }
    result = simple_chain.invoke(inputs)

    new_query = state["query"] + [AIMessage(content=result.content)]

    return {
        "query": new_query,
        "user_data": state['user_data'],
        "allocations": state['allocations'],
        "data": state['data'],
        "output": result.content
    }


graph = StateGraph(Graph)
graph.add_node("chat", chat)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")
graph.add_edge("chat", END)

app = graph.compile(checkpointer=memory)


with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    data = json.load(f)
with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    allocs = json.load(f)
inputs = {
    "query":"I got a hike of 10k, help me reallocate my investments.",
    "user_data":data,
    "allocations":allocs,
    "data":"",
}


#print(app.invoke(inputs, config={"configurable": {"thread_id": "sample"}}))