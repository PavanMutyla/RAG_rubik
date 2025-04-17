"""All prompts utilized by the RAG pipeline"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from RAG.tools import json_to_table, goal_feasibility
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

        ------------------------
        🧠 GENERAL INSTRUCTIONS:
        ------------------------
        1. Understand the user's intent — identify whether they need advice, review, calculations, visualization, or planning.
        2. Analyze `user_data` and `allocations` to provide **personalized** and **context-aware** responses.
        3. Refer to the `data` document only if necessary for factual support.
        4. Update your understanding as the conversation progresses — later messages may override previous ones.
        5. If the query needs a calculation or a table, **generate the inputs needed for the correct tool.**

        -------------------------------
        ⚙️ AVAILABLE TOOLS AND FORMATS:
        -------------------------------

        1. `goal_feasibility`
        If the user gives new goals other than the ones in {user_data}, perform feasibility for that goal.
        The tool takes the following inputs-
        - Inputs:
        - goal_amount: float (₹)
        - timeline: float (months)
        - current_savings: float (₹)
        - income: float (₹)

        2. `json_to_table`
        - Use this tool to display {allocations}
        Use this if there is a need to update or display or **visualize or tabulate** the allocations, then use this tool.
        - Input: JSON object or list

        If any tool is used, return in this format:
        ```json
        "tool_calls": [
        {{
            "tool_name": "<tool_name>",
            "inputs": {{
            "key": value
            }}
        }}
        ]
        If no tools are needed:
            "tool_calls": []
        📝 RESPONSE FORMAT:
            Your actual answer should follow this structure:

                🎯 Overview (10%) — Generalized comment on the query

                🧾 Personalized Insights (45%) — Use user_data and allocations to tailor your response

                🧐 Goal Evaluation (15%) — Honest take: is the goal logical or not? Use numbers, be blunt but respectful.

                🧠 Summary (20%) — Summarize key takeaways and next steps
        🔖 REFERENCES (if used):
            If you used the data document, include:
            "references": [
                            {{
                                "title": "<Title of the document>",
                                "summary": "<Brief explanation of the section used>",
                                "cited_text": "<Quoted or paraphrased text>",
                                "context": "<How it was used in your response>"
                            }}
                            ]



"""
llm = llm.bind_tools(tools =tools)
simple_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template=template),
    MessagesPlaceholder(variable_name="chat_history"),  
    HumanMessagePromptTemplate.from_template("User: {query}, {user_data}, {allocations}, {data}")
])
simple_chain = simple_prompt | llm | StrOutputParser()
