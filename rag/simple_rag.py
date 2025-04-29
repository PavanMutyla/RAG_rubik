from langgraph.graph import START, END, StateGraph
from langchain_openai import OpenAIEmbeddings
from RAG.chains import simple_chain
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import TypedDict, Optional, Dict, List, Union, Annotated, Any
from langchain_core.messages import AnyMessage #human or AI message
from langgraph.graph.message import add_messages # reducer in langgraph 
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langgraph.checkpoint.memory import MemorySaver
import json
from langchain.tools.base import StructuredTool
import langchain
from RAG.tools import json_to_table, goal_feasibility, rag_tool, save_data
import re

from dotenv import load_dotenv
load_dotenv()

memory = MemorySaver()
config = {"thread_id":"sample"}
tools = [json_to_table, rag_tool]
table_tool = Tool(name = "table", func = json_to_table, description="takes in a dict of changed allocations and returns a dataframe.")
table_tool = StructuredTool.from_function(json_to_table)
rag = Tool(name = 'rag', func=rag_tool, description="RAG with external knowledge.")
#tool_executor = ToolExecutor([json_to_table, goal_feasibility])
json_to_table_node = ToolNode([table_tool])

rag_tool_node = ToolNode([rag])
class Graph(TypedDict):
    query: Annotated[list[Any], add_messages]
    user_data: Dict
    allocations: Dict
    output: Dict
    retrieved_context: str

# --- NODES ---
def call_model(state: Graph):
    inputs = {
        "query": state["query"],
        "user_data": state["user_data"],
        "allocations": state["allocations"],
        "chat_history": state["query"],
        "retrieved_context": state.get("retrieved_context", "")
    }
    result = simple_chain.invoke(inputs)
    return {
        "query": state["query"],
        "user_data": state["user_data"],
        "allocations": state["allocations"],
        "retrieved_context": "",  # clear after use
        "output": result.content
    }

def tools_node(state: Graph):
    last_message = state["query"][-1]  # last AI/user message
    tool_calls = getattr(last_message, "tool_calls", None)
    
    if tool_calls:
        tool_name = tool_calls[0].get('name', '')
        
        if tool_name == "json_to_table":
            print('tool---------------------------------------------------------------------------------------')
            tool_output = json_to_table(state["allocations"])
            new_message = AIMessage(content=tool_output)
        
        elif tool_name == "rag_tool":
            print('tool---------------------------------------------------------------------------------------')
            tool_output = rag_tool(state['query'])
            new_message = AIMessage(content=tool_output)
        
        else:
            print('unk---------------------------------------------------------------------------------------')
            # Unknown tool fallback
            new_message = AIMessage(content="Unknown tool called.")
        
        return {
            **state,
            "query": state["query"] + [new_message]
        }
    
    return state

# --- CONDITIONAL FLOW ---
def should_continue(state: Graph):
    last_message = state["query"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# --- WORKFLOW SETUP ---
workflow = StateGraph(Graph)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tools_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# --- COMPILE ---
app = workflow.compile(checkpointer=memory)
'''

from IPython.display import Image, display
try:
    image_data = app.get_graph().draw_mermaid_png()
    # Save the image
    with open("graph.png", "wb") as f:
        f.write(image_data)

    # Optionally, still display it
    display(Image(image_data))
except Exception as e:
    print(f"An error occurred: {e}")
    pass
'''


with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    data = json.load(f)
with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    allocs = json.load(f)
inputs = {
    "query":"Show me my allocations.",
    "user_data":data,
    "allocations":allocs,
    "data":"",
    "chat_history": [],
    
}

langchain.debug = True

#print(app.invoke(inputs, config={"configurable": {"thread_id": "sample"}}))
#print(json_to_table.args_schema.model_json_schema())
#print(table_tool.invoke({"input_data": allocs}))
#print(rag.invoke("What is tax?"))