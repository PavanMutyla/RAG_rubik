from rag.RAG.tools import json_to_table, goal_feasibility, update_user_state
from rag.RAG.chains import template, simple_prompt
from langchain.tools import tool 
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
import os 
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
load_dotenv()

tools = [json_to_table, goal_feasibility, update_user_state]

llm = llm = ChatOpenAI(
    model='gpt-4.1-nano',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2
)



agent = create_tool_calling_agent(llm, tools, simple_prompt)
inputs = {
    "query": "I want to buy a car of 12L within a year",
    "user_data": None,  # or some mock data like {"income": 60000, ...}
    "allocations": None,  # or {"FD": 300000, "Mutual Funds": 200000}
    "data": None,  # if no PDF or other external data, keep it None
    "chat_history": [],  # start with empty history if fresh
    "agent_scratchpad": ""  # always include this
}


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
"""
print(agent_executor.invoke({
    "query": "I want to buy a car of 12L within a year",
    "user_data": None,  # or some mock data like {"income": 60000, ...}
    "allocations": None,  # or {"FD": 300000, "Mutual Funds": 200000}
    "data": None,  # if no PDF or other external data, keep it None
    "chat_history": [],  # start with empty history if fresh
    "agent_scratchpad": ""  # always include this
}))
"""
#print(agent_executor.invoke(inputs))