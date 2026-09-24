import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

load_dotenv()




llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))





class Classification(BaseModel):
    category: Literal[
        "billing",
        "technical",
        "refund",
        "account",
        "general",
    ]

    urgency: Literal[
        "low",
        "medium",
        "high",
    ]

    sentiment: Literal[
        "positive",
        "neutral",
        "negative",
    ]


class Extraction(BaseModel):
    issue: str
    customer_name: str | None
    product: str | None
    important_details: list[str]


class SupportResponse(BaseModel):
    response: str
    needs_human: bool




CLASSIFICATION_PROMPT = """
You are a customer support classification system.

Classify the customer's message.

Choose:
- billing: payment, invoice, or subscription charges
- technical: bugs, errors, crashes, or technical problems
- refund: requests to return money
- account: login, account, or profile problems
- general: anything that does not fit the categories above

Determine:
- category
- urgency
- sentiment

Only classify the message.
Do not answer the customer.
"""


EXTRACTION_PROMPT = """
You are a customer support information extraction system.

Extract useful information from the customer's message.

Identify:
- the main issue
- customer name if available
- product if available
- important details

Do not invent information.

If something is not available, return null.
"""


RESPONSE_PROMPT = """
You are a professional customer support agent.

Write a helpful response to the customer.

Use the classification and extracted information provided.

Rules:
- Be polite.
- Be concise.
- Do not invent policies.
- Do not promise refunds or actions that were not confirmed.
- If the issue requires human intervention, set needs_human to true.
- Otherwise set needs_human to false.
"""