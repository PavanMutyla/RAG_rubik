from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from rag.RAG.tools import json_to_table, goal_feasibility
from rag.RAG.chains import simple_prompt
import os
llm = ChatOpenAI(
    model='gpt-4.1-nano',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2
)
class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]
tools = [json_to_table, goal_feasibility]
llm_with_prompt_and_tools = simple_prompt | llm.bind_tools(tools=tools)

def chatbot(state: BasicChatBot):
    last_user_input = state["messages"][-1].content
    return {
        "messages": [
            llm_with_prompt_and_tools.invoke({
                "query": last_user_input,
                "user_data": state.get("user_data", ""),
                "allocations": state.get("allocations", ""),
                "data": state.get("data", "")
            })
        ]
    }


def tools_router(state: BasicChatBot):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatBot)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

app = graph.compile()