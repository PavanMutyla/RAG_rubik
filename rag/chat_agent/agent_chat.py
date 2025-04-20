from rag.RAG.tools import json_to_table, goal_feasibility
from rag.RAG.chains import template, simple_prompt, simple_chain
from langchain.tools import tool 
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, create_react_agent, AgentType
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_tool_calling_agent
import os 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

tools = [json_to_table]

llm = llm = ChatOpenAI(
    model='gpt-4.1-nano',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2,
)

memory = ChatMessageHistory(session_id="test-session")

agent = initialize_agent(llm = llm,tools = tools, prompt=simple_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,  # Associates memory with session_id
    input_messages_key="query",
    history_messages_key="chat_history",
)