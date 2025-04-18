from langgraph.graph import START, END, StateGraph
from langchain_openai import OpenAIEmbeddings
from RAG.chains import simple_chain, llm_with_tools
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import TypedDict, Optional, Dict, List, Union, Annotated
from langchain_core.messages import AnyMessage #human or AI message
from langgraph.graph.message import add_messages # reducer in langgraph 
from langgraph.prebuilt import ToolNode
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

import json
from RAG.tools import json_to_table, goal_feasibility
import re

from dotenv import load_dotenv
load_dotenv()

tools = [json_to_table, goal_feasibility]
#tool_executor = ToolExecutor([json_to_table, goal_feasibility])

class GraphState(TypedDict):
    query: str
    user_data: dict
    allocations: dict
    data: dict
    chat_history: list
    messages: list
def custom_agent_node(state: GraphState) -> GraphState:
    chain_input = {
        "query": state["query"],
        "user_data": state["user_data"],
        "allocations": state["allocations"],
        "data": state["data"],
        "chat_history": state["chat_history"],
    }
    result = simple_chain.invoke(chain_input)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=result.content)],
    }
# ⚙️ Tool execution node
def tool_node(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    tool_outputs = []
    for call in tool_calls:
        tool_outputs.append(tool_executor.invoke(call))

    tool_messages = [ToolMessage(content=str(out), tool_call_id=call.id)
                     for call, out in zip(tool_calls, tool_outputs)]

    return {
        **state,
        "messages": state["messages"] + tool_messages,
    }

# ➕ Update memory with latest user additions (goals etc.)
def update_state(state: GraphState) -> GraphState:
    # Example logic to store new goal from message
    for msg in state["messages"]:
        if "goal" in msg.content.lower():
            state["user_data"]["goals"].append(msg.content)  # custom memory update
    return state

# 🧠 LangGraph setup
graph = StateGraph(GraphState)

graph.add_node("agent", custom_agent_node)
graph.add_node("tools", tool_node)
graph.add_node("update_memory", update_state)

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", "update_memory")
graph.add_edge("update_memory", END)

app = graph.compile()
'''

class Graph(TypedDict):
    user_data: Optional[Dict]
    allocations: Optional[Dict]
    pdf: Optional[Union[str, bytes]]
    memory: List[BaseMessage]
    messages: List[BaseMessage]

def tool_call_llm(state):
    user_data = state['user_data']
    allocations = state['allocations']
    docs = state['pdf']
    memory = state['memory']
    messages = state['messages']

    # Extract latest HumanMessage as the actual query
    query_content = [msg.content for msg in messages if isinstance(msg, HumanMessage)][-1]

    inputs = {
        'user_data': user_data,
        'query': query_content,
        'allocations': allocations,
        'data': docs,
        'chat_history': memory
    }

    response = simple_chain.invoke(inputs)

    return {
        # Required for the next prompt
        "query": query_content,
        "user_data": user_data,
        "allocations": allocations,
        "data": docs,
        "chat_history": memory,

        # Update memory/messages as state
        "messages": messages + [response],
        "memory": memory + [response]
    }

def tools_condition(state):
    latest_msg = state["messages"][-1]

    try:
        parsed = json.loads(latest_msg.content)
        return "tools" if parsed.get("tool_calls") else "tool_calling_llm"
    except Exception:
        return "tool_calling_llm"


workflow = StateGraph(Graph)
from langchain_core.runnables import RunnableLambda

workflow.add_node(
    "tool_calling_llm",
    RunnableLambda(tool_call_llm).with_config({
        "input_keys": ["query", "user_data", "allocations", "data", "chat_history", "messages", "memory"]
    })
)

workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, 'tool_calling_llm')

workflow.add_conditional_edges(
    "tool_calling_llm", 
    tools_condition
)

workflow.add_edge('tools', 'tool_calling_llm')

graph = workflow.compile()


inputs = {
                    "query": "I want to buy a car of 12L within a year",
                    "user_data": None,
                    "allocations": None,
                    "pdf":None,
                    "memory": [""],

                }
print(agent.invoke(inputs))
'''