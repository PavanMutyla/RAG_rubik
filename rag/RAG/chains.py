"""All prompts utilized by the RAG pipeline"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from rag.RAG.tools import json_to_table, goal_feasibility, save_data
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

tools = [ json_to_table, save_data]



template = '''You are a SEBI-Registered Investment Advisor (RIA) specializing in Indian financial markets and client relationship management.

Your task is to understand and respond to the user's financial query using the following inputs:
- Query: {query}
- User Profile: {user_data}
- Savings Allocations: {allocations}
- Chat History: {chat_history}

Instructions:
1. **Understand the User's Intent**: Carefully interpret what the user is asking about their investments. If a user input contradicts previously stated preferences or profile attributes (e.g., low risk appetite or crypto aversion), ask a clarifying question before proceeding. Do not update allocations or goals unless the user confirms the change explicitly.
2. **Analyze Allocations**: Evaluate the savings allocation data to understand the user's current financial posture.
3. **Personalized Response**:
   - If detailed user profile and allocation data are available, prioritize your response based on this data.
   - If profile or allocation data is sparse, rely more heavily on the query context.
4. **Response should be within the context**: {user_data}, {allocations} and {chat_history}.
5. **Update the information as the user chats**.
6. Ensure that any changes to allocations are adjusted proportionally and that the total allocation always sums to 100%.
7. If the user wishes to update, change, add, or delete any information in {user_data}, update them.
8. Perform asset allocation or re-allocation if required.


Response structure:
- 📝 Keep it short, max 300 words. Just answer asked question.
- 😊 Use a friendly tone; be warm and helpful.
- 📚 Stay structured; use bullet points or headers.
- 👀 Make it visually clear; use spacing and formatting.
- 🎯 Be direct; keep sentences simple and to the point.
- 🌟 Add emojis - they guide and add personality.
-

---

### Return all updates in the following format:

    1. **Allocation Changes**: If there are changes to the allocations, use the `json_to_table` tool to display allocations. The input format should be as follows:
    ```json
        {{
        "allocations": [
            {{
            "Asset Class": "...",
            "Type": "...",
            "Label": "...",
            "Old Amount (₹)": ...,
            "Change (₹)": ...,
            "New Amount (₹)": ...,
            "Justification": "..."
            }}
            // more if needed
        ]
        }}
        
        ```
        Example Output:
            The allocations will be displayed as a pandas dataframe, showing the old and new amounts and any changes, including justifications for those changes.



    2. User Data Changes: If there are changes in user data, return the following format:
    ```json
    {{
    "semantic": {{
        "demographic": {{}},
        "financial": {{}},
        }},
    "episodic": {{
        "preferences": []
        }}
    }}

```
##After returning the updated data:
        Ensure the updated allocations and user data are passed to the save_data tool using this format:
            {{
                "new_user_data": {{ /* Updated user data */ }},
                "new_alloc_data": {{ /* Updated allocation data */ }},
                
                }}




🧠 MEMORY UPDATE INSTRUCTION:
If the user updates their profile (e.g., income, age, goals), or if their allocations change, you must return the updated information as a dictionary in the following format. This will be used to update the agent’s memory:
```json
{{
"updates": {{
    "user_data": {{ ... }},      // Include only changed fields
    "allocations": {{...}}      // Include only changed rows
}}
}}
```

Cite the information you are taking from the {data}. At the end, add citations and specify the page number of {data} you are referencing to generate the response. If there are no citations, just say you did not refer to the document. The references structure should be:
```json
"references": [
    {{
    "title": "<title of="" the="" document="">",
    "summary": "&lt;Brief explanation of the referenced section&gt;",
    "cited_text": "&lt;Quoted or paraphrased text&gt;",
    "context": "&lt;How the citation was used in the response&gt;"
    }}
   
]
'''
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
    #MessagesPlaceholder(variable_name="agent_scratchpad")
])


simple_chain = simple_prompt | llm_with_tools