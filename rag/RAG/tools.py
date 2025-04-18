from langchain.tools import tool
import pandas as pd 
import json 
import re
import os
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()

@tool
def json_to_table(json_data):
    """Convert JSON data to a markdown table. Use when user asks to visualise or tabulate structured data."""
    
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    
    
    df = pd.json_normalize(json_data)
    print(f"json_to_table output: {df.to_markdown()}")  # Debugging line
    return df.to_markdown()

@tool
def goal_feasibility(goal_amount: float, timeline: float, current_savings: float, income : float) -> dict:
    """Evaluate if a financial goal is feasible based on user income, timeline, and savings. Use when user asks about goal feasibility."""
    # Input checks
    if timeline <= 0:
        return {
            "feasible": False,
            "status": "Invalid",
            "monthly_required": 0,
            "reason": "Timeline must be greater than 0 months."
        }

    # Calculate the remaining amount
    remaining_amount = goal_amount - current_savings
    if remaining_amount <= 0:
        return {
            "feasible": True,
            "status": "Already Achieved",
            "monthly_required": 0,
            "reason": "You have already met or exceeded your savings goal."
        }

    monthly_required = remaining_amount / timeline
    income_ratio = monthly_required / income

    # Feasibility classification
    if income_ratio <= 0.3:
        status = "Feasible"
        feasible = True
        reason = "The required savings per month is manageable for an average income."
    elif income_ratio <= 0.7:
        status = "Difficult"
        feasible = False
        reason = "The required monthly saving is high but may be possible with strict budgeting."
    else:
        status = "Infeasible"
        feasible = False
        reason = "The required monthly saving is unrealistic for an average income."

    return {
        "feasible": feasible,
        "status": status,
        "monthly_required": round(monthly_required, 2),
        "reason": reason
    }



@tool
def update_user_state(chat_history: list, user_data: dict = None, allocations: dict = None) -> dict:
    """
    Update user_data and allocations based on chat history.
    Saves updated JSONs to a folder named 'updated_json' inside the path set by env var 'Data'.
    """
    user_data = deepcopy(user_data or {
        "sematic": {"financial": {}, "demographic": {}},
        "episodic": {"goals": [], "prefrences": []}
    })

    allocations = deepcopy(allocations or {})

    for msg in chat_history:
        if isinstance(msg, dict) and "content" in msg:
            content = msg["content"].lower()

            # Salary update
            if match := re.search(r"(salary|income) (is|to be)? ?₹?([\d,]+)", content):
                user_data["sematic"]["financial"]["salary"] = int(match.group(3).replace(",", ""))

            # Expenses update
            if match := re.search(r"(monthly expenses|spend) (is|are)? ?₹?([\d,]+)", content):
                user_data["sematic"]["financial"]["monthly_expenses"] = int(match.group(3).replace(",", ""))

            # New goal
            if match := re.search(r"(save | buy) ₹?([\d,]+) for (.*?) in (\d+) (months|years|year)", content):
                amount = int(match.group(1).replace(",", ""))
                label = match.group(2).strip().title()
                timeline = int(match.group(3))
                if "year" in match.group(4):
                    timeline *= 12
                goal = {
                    "priority": "medium",
                    "target": amount,
                    "tenuer": f"{timeline} months",
                    "label": label
                }
                if goal not in user_data["episodic"]["goals"]:
                    user_data["episodic"]["goals"].append(goal)

            # Add MF
            if "add mutual fund" in content:
                mf = {
                    "type": "mutual_fund",
                    "amount": 50000,
                    "category": "index_fund",
                    "label": "User Added MF",
                    "portfolio": []
                }
                allocations.setdefault("equity", []).append(mf)

    # Save to folder
    base_path = os.getenv("Data", ".")
    save_path = os.path.join(base_path, "updated_json")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "updated_user_data.json"), "w") as f:
        json.dump(user_data, f, indent=2)

    with open(os.path.join(save_path, "updated_allocations.json"), "w") as f:
        json.dump(allocations, f, indent=2)

    return {
        "user_data": user_data,
        "allocations": allocations
    }
