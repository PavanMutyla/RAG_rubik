from rag.RAG.tools import json_to_table, goal_feasibility,rag_tool
from rag.RAG.chains import template, simple_prompt, simple_chain
from langchain.tools import tool 
import json
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

tools = [
    Tool(name="DisplayJSON", func=json_to_table, description="Displays JSON as table"),
    Tool(name="RAGSearch", func=rag_tool, description="Retrieves documents related to the query")
]

llm = llm = ChatOpenAI(
    model='gpt-4.1-nano',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True,
    agent_kwargs={"prompt": simple_prompt}
)


with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    data = json.load(f)
with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    allocs = json.load(f)
inputs = {
    "query":"Display my investments.",
    "user_data":data,
    "allocations":allocs,
   
    "chat_history": [],
    
}
