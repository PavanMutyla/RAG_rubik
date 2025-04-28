from enum import IntEnum, StrEnum, Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class KBCategory(str, Enum):
    ProductCategory = "product_category"
    InvestmentRegulations = "investment_regulations"
    TaxationDetails = "taxation_details"
    MarketSegments = "market_segments"
    CulturalAspects = "cultural_aspects"
    General = "general"

class THRESHOLD(Enum):
    threshold = 0.2


class TextResponseSection(BaseModel):
    general: str = Field(..., description="General introduction (around 15% of the text, ~50 words)")
    user_related: str = Field(..., description="User-specific analysis and details (around 50% of the text, ~175 words)")
    suggestions_review: str = Field(..., description="Suggestions or review section (around 15% of the text, ~50 words)")
    summary: str = Field(..., description="Summary and conclusion (around 20% of the text, ~70 words)")

class AllocationDelta(BaseModel):
    asset_class : str = Field(..., description='The class of the proudct.')
    type_asset : str = Field(..., description="The type of proudct under asset class.")
    label : str = Field(..., description="The exact investment element.")
   
    old_value: float = Field(..., description="The previous allocation value")
    change : float = Field(..., description="The change in funds compared to old value.")
    new_value: float = Field(..., description="The updated allocation value")
    justification: str = Field(..., description="Reason for the change in allocation")

class Citations(BaseModel):
    url: str = Field(..., description="URL pointing to the source")
    context: str = Field(..., description="Relevant text snippet from the source")
    title: str = Field(..., description="Title of the source")

class Response(BaseModel):
    text_response: TextResponseSection = Field(..., description="Structured text response broken into 4 sections")
    allocation_delta: Optional[AllocationDelta] = Field(None, description="Returns updated allocation if any")
    citations: List[Citations] = Field(default_factory=list, description="List of citations supporting the response")