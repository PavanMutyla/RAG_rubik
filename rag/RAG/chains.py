"""All prompts utilized by the RAG pipeline"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from rag.RAG.tools import json_to_table, goal_feasibility
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from dotenv import load_dotenv
load_dotenv()


gemini = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
llm = ChatOpenAI(
    model='gpt-4.1-nano',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2
)

# Schema for grading documents
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {data} \n\n User question: {query}")
])

retrieval_grader = grade_prompt | structured_llm_grader


prompt = PromptTemplate(
    template='''
    You are a SEBI-Registered Investment Advisor (RIA) specializing in Indian financial markets and client relationship management.

    Your task is to understand and respond to the user's financial query using the following inputs:
    - Query: {query}
    - Documents: {data}
    - User Profile: {user_data}
    - Savings Allocations: {allocations}

    

    Instructions:
    1. Understand the User's Intent: Carefully interpret what the user is asking about their investments.
    2. Analyze Allocations: Evaluate the savings allocation data to understand the user's current financial posture.
    3. Personalized Response:
    - If detailed user profile and allocation data are available, prioritize your response based on this data.
    - If profile or allocation data is sparse, rely more heavily on the query context.
    4. Use Supporting Documents: Extract relevant insights from the provided documents ({data}) to support your answer.
    5. When Unsure: If the documents or data do not contain the necessary information, say "I don't know" rather than guessing.

    Always aim to give a response that is:
    - Data-informed
    - Client-centric
    - Aligned with Indian financial regulations and norms


    ''',
    input_variables=['query', 'data', 'user_data', 'allocations']
)

rag_chain = prompt | gemini | StrOutputParser()


# Prompt
system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human",
            "Here is the initial question: \n\n {query} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


from pydantic import BaseModel, Field, RootModel
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser

# Define the Pydantic model using RootModel
class CategoryProbabilities(RootModel):
    """Probabilities for different knowledge base categories."""
    root: Dict[str, float] = Field(description="Dictionary mapping category names to probability scores")

system_classifier = """You are a query classifier that determines the most relevant knowledge bases (KBs) for a given user query. 
Analyze the semantic meaning and intent of the query and assign probability scores (between 0 and 1) to each KB.

Ensure the probabilities sum to 1 and output a JSON dictionary with category names as keys and probabilities as values.
"""

classification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_classifier),
        (
            "human",
            "Here is the user query: \n\n {query} \n\n Assign probability scores to each of the following KBs:\n"
            "{categories}\n\nReturn a JSON object with category names as keys and probability scores as values."
        ),
    ]
)

# Create a JSON output parser
json_parser = JsonOutputParser(pydantic_object=CategoryProbabilities)

# Create the chain with the structured output parser
query_classifier = classification_prompt | llm | json_parser


#query_classifier = classification_prompt | llm | StrOutputParser()

"""
name: str
    
    position: Dict[str, int]
    riskiness: int
    illiquidity: int
    
    amount: float
    currency: str = "inr"
    percentage: float
    explanation: Dict[str, str]
    
    assets: List[AssetAllocation]
"""

tools = [ json_to_table, goal_feasibility
]



template = """
        You are a SEBI-Registered Investment Advisor (RIA) specializing in Indian financial markets and client relationship management.

        Your task is to understand and respond to the user's financial query using the following inputs:
        - Query: {query}
        - User Profile: {user_data}
        - Savings Allocations: {allocations}
        - Additional Data: {data}
        - Chat History : {chat_history}

        ------------------------
        🧠 GENERAL INSTRUCTIONS:
        ------------------------
        1. Understand the user's intent — identify whether they need advice, review, calculations, visualization, or planning.
        2. Analyze {user_data}, {allocations} and **{chat_history}** to provide **personalized** and **context-aware** responses.
        3. Refer to the {data} document if required factual support or financial decision making.
        4. Update your understanding as the conversation progresses — later messages may override previous ones.
        5. If the query needs a calculation or a table, **generate the inputs needed for the correct tool.**
        6. If the user shares new financial goals or updates in the chat, extract and include them in the updated analysis. Do not ignore chat updates even if they’re not in `user_data`.
        7. If the user provides new or updated information via chat (e.g., new income, goal, or asset class), use it to **update your understanding**, and if relevant, apply the tools or recalculate previous advice.
        8. If new allocation inputs or strategy changes are mentioned mid-chat, treat them as updates to the `allocations` structure.
        9. Reuse or revise the previous allocation table using the latest updates in `chat_history` and `user_data`.
        10. For general or informational queries (e.g., “What is tax?”, “What is an ELSS?”, “Explain PPF”), do **not** follow the full structured response format. Answer in a clear, concise, and user-friendly tone — like an expert talking to a curious client.
        11. Only use the full format (🎯 Overview, 🧾 Insights, 🧐 Goal Evaluation, etc.) when the query is personal, strategic, or involves calculations, savings review, planning, or goal tracking.
        12. You may call `update_user_state` at any time to refresh financial data from user messages.
        13. After updating, continue reasoning with the new values returned by the tool.


        -------------------------------
        ⚙️ AVAILABLE TOOLS AND FORMATS:
        -------------------------------

        1. `goal_feasibility`
        Use this tool when:
        - A new goal is mentioned not present in {user_data}.
        - The user provides a target amount and timeline for a purchase, retirement, education, etc.
        - Use data from {chat_history} to resolve missing values (e.g., income or current savings).

        - Inputs:
        - goal_amount: float (₹)
        - timeline: float (months)
        - current_savings: float (₹)
        - income: float (₹)


        2. `json_to_table`
        Use this tool *whenever* there is a need to display, rearrange, recalculate, or visualize the user's current savings allocations — including comparisons, new strategies, or reallocations.

        - Always present allocations in tabular format via this tool, instead of plain text.

        - Input: JSON object or list, the tool returns a pandas dataframe object.


        3. `update_user_state`
        Use this tool if the user shares new or updated financial information in the chat (e.g., salary changes, expense updates, or new savings goals).

        - This tool will extract updated values from {chat_history}, modify {user_data} and {allocations}, save the updates into the `Data/updated_json/` directory, and return the new state.
        - Always use this tool **before** feasibility or planning tools if a user gives any financial update via chat.

        - Inputs:
        - chat_history: list
        - user_data: dict (optional)
        - allocations: dict (optional)

        - Returns:
        - A dictionary with updated {user_data} and {allocations}


        📝 RESPONSE FORMAT:
        Your actual answer should follow this structure:

        🎯 Overview — Generalized comment on the query

        🧾 Personalized Insights — Use user_data and allocations to tailor your response

        🧐 Goal Evaluation — Honest take: is the goal logical or not? Use numbers, be blunt but respectful.

        Display the output dataframe of the tool `json_to_table`.

        🧠 Data Update Acknowledgement — If the user added new data, mention the updated memory (like: goals, salary, etc)

        🧠 Summary — Summarize key takeaways and next steps

        🔖 REFERENCES (use the external data at least once before generating response):
        If you used the data document, include:
        "references": [
                        {{
                            "title": "<Title of the document>",
                            "summary": "<Brief explanation of the section used>",
                            "cited_text": "<Quoted or paraphrased text>",
                            "context": "<How it was used in your response>"
                        }}
                    ]
        If no references used: 
        "references":[]




"""
llm_with_tools = llm.bind_tools(tools =tools)

simple_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template=template),
    
    # Preserve memory of previous turns
    MessagesPlaceholder(variable_name="chat_history", optional=True),

    # Use a separate message for the user query
    HumanMessagePromptTemplate.from_template("User Query: {query}"),

    # Pass structured data cleanly
    HumanMessagePromptTemplate.from_template("Current User Profile:\n{user_data}"),
    HumanMessagePromptTemplate.from_template("Current Allocations:\n{allocations}"),
    HumanMessagePromptTemplate.from_template("Reference Data:\n{data}"),

    # Agent scratchpad for tool calls
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


simple_chain = simple_prompt | llm_with_tools 