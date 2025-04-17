from agents import Runner, Agent
import asyncio
import json 
import os 
from typing import List
from pydantic import BaseModel
from Agent.structure import Response
from Agent.tools import update_memory
from dotenv import load_dotenv
load_dotenv()


model = os.environ.get('MODEL')
print(model)