from agents import function_tool
import json 
from typing import Annotated
import os 


def load_memory(memory_file : str):
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            return json.load(f)
    return {}

def save_memory(data, memory_file):
    with open(memory_file, "w") as f:
        json.dump(data, f, indent=2)

def smart_update(category: str, key: str, value: str):
    data = load_memory()

    if category not in data:
        data[category] = {}

    data[category][key] = value
    save_memory(data)
    return f"Memory updated: [{category}] {key} → {value}"

@function_tool
def update_memory(
    category: Annotated[str, "Categories"],
    key: Annotated[str, "Key to update"],
    value: Annotated[str, "The value to store or update"]
) -> str:
    """Update or add information to the agent's memory JSON file."""
    return smart_update(category, key, value)