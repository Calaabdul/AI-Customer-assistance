import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


load_dotenv()



"""                 START
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
        Safety Check     Relevance Check
             │                 │
             └────────┬────────┘
                      ▼
                Classification
                      │
                      ▼
                   Analysis
                      │
                      ▼
                 Final Response
                      │
                     END
                     
"""



# So the three main processing steps are:

# 1. Classification
# 2. Analysis
# 3. Generation
# Before those, there are two parallel guardrails:

# 1. Safety guardrail
# 2. Customer-support relevance guardrail




# ============================================================
# PYDANTIC OUTPUTS
# ============================================================


class SafetyResult(BaseModel):
    safe: bool = Field(description="Whether the message is safe for a customer-support system")
    confidence: float = Field(description="Confidence between 0 and 1", ge=0.0, le=1.0)
    reason: str = Field(description="Reason for the safety assessment")


class RelevanceResult(BaseModel):
    relevant: bool = Field(description="Whether the message is relevant to customer support")
    confidence: float = Field(description="Confidence between 0 and 1", ge=0.0, le=1.0)
    reason: str = Field(description="Reason for the relevance assessment")


class ClassificationResult(BaseModel):
    category: Literal[
        "billing",
        "technical",
        "account",
        "order",
        "refund",
        "general",
    ] = Field(description="Category of the customer-support message")

    urgency: Literal[
        "low",
        "medium",
        "high",
    ] = Field(description="Urgency of the customer-support message")

    summary: str = Field(
        description="One-sentence summary of the customer-support message"
    )


class AnalysisResult(BaseModel):
    customer_need: str = Field(description="What the customer needs")
    recommended_action: str = Field(description="The recommended action to take")
    missing_information: str = Field(
        description="Any missing information that would help resolve the issue"
    )


# ============================================================
# GRAPH STATE
# ============================================================


class WorkflowState(TypedDict, total=False):
    message: str

    safety: SafetyResult
    relevance: RelevanceResult

    reject: Literal["classify", "reject"]

    classification: ClassificationResult
    analysis: AnalysisResult

    final_response: str


# ============================================================
# MODEL
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# PROMPTS
# ============================================================

safety_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Check whether this message is safe for a customer-support system.

Unsafe: threats, illegal activity, harmful instructions, or malicious
cyber activity.

Normal complaints and angry customers are safe.
""",
        ),
        ("human", "{message}"),
    ]
)


relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Check whether this message is about customer support.

Relevant: billing, accounts, orders, refunds, technical problems,
products, services, or general customer help.

Reject unrelated topics.
""",
        ),
        ("human", "{message}"),
    ]
)


classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Classify this customer-support message.

Choose the closest category and urgency.
Give a one-sentence summary.
""",
        ),
        (
            "human",
            """
Message:
{message}

Safety:
{safety}

Relevance:
{relevance}
""",
        ),
    ]
)


analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Analyze the customer problem.

State what they need, the recommended action,
and any missing information.
Do not invent policies.
""",
        ),
        (
            "human",
            """
Message:
{message}

Classification:
{classification}
""",
        ),
    ]
)


final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Write a short, professional customer-support response.

Do not mention the AI, workflow, guardrails, or internal analysis.
Do not invent policies or claim an action was completed.
""",
        ),
        (
            "human",
            """
Message:
{message}

Analysis:
{analysis}
""",
        ),
    ]
)


# ============================================================
# CHAINS
# ============================================================

safety_chain = safety_prompt | llm.with_structured_output(SafetyResult)

relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResult)

classification_chain = classification_prompt | llm.with_structured_output(ClassificationResult)

analysis_chain = analysis_prompt | llm.with_structured_output(AnalysisResult)

final_chain = final_prompt | llm


# ============================================================
# GUARDRAILS
# ============================================================


def safety_guardrail(state: WorkflowState):
    result = safety_chain.invoke({"message": state["message"]})

    return {"safety": result}


def relevance_guardrail(state: WorkflowState):
    result = relevance_chain.invoke({"message": state["message"]})

    return {"relevance": result}


# guardrail check
def check_guardrails(state: WorkflowState):
    if not state["safety"].safe: 
        return {"route": "reject"} 

    if not state["relevance"].relevant: 
        return { "route": "reject"} 

    return {"route": "classify"}

# reject message if it fails guardrail check
def reject_message(state: WorkflowState): 
    if not state["safety"].safe: 
        response = ( "I can only help with customer-support requests." ) 
    else: 
        response = ( """I can only help with customer-support questions "
                       about products, services, accounts, orders,
                       billing, refunds, or technical issues.""" ) 

    return { "final_response": response }

# ============================================================
# MAIN STEPS
# ============================================================


def classify_message(state: WorkflowState):
    result = classification_chain.invoke(
        {
            "message": state["message"],
            "safety": state["safety"].model_dump_json(),
            "relevance": state["relevance"].model_dump_json(),
        }
    )

    return {"classification": result}


def analyze_message(state: WorkflowState):
    result = analysis_chain.invoke(
        {
            "message": state["message"],
            "classification": state["classification"].model_dump_json(),
        }
    )

    return {"analysis": result}


def generate_response(state: WorkflowState):
    result = final_chain.invoke(
        {
            "message": state["message"],
            "analysis": state["analysis"].model_dump_json(),
        }
    )

    return {"final_response": result.content}


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(WorkflowState)

builder.add_node( "safety_guardrail", safety_guardrail, )

builder.add_node("relevance_guardrail", relevance_guardrail)

builder.add_node( "check_guardrails", check_guardrails)

builder.add_node( "reject", reject_message)

builder.add_node( "classify", classify_message)

builder.add_node( "analyze", analyze_message) 

builder.add_node( "generate", generate_response)


# Two independent nodes begin from START.


#parallel guardrails
builder.add_edge( START, "safety_guardrail")

builder.add_edge(START, "relevance_guardrail")

# Conditional routing
# After check_guardrails finishes, look at the state and decide which node to go to next

builder.add_conditional_edges("check_guardrails", lambda state: state["route"],
                               { "classify": "classify",
                                "reject": "reject", })


# def get_route(state):
    # return state["route"]

# state = {
#     "message": "I want to hack your system",
#     "route": "reject"
# }



builder.add_edge("classify","analyze")

builder.add_edge("analyze","generate")


builder.add_edge( "reject", END)
builder.add_edge("generate", END)


workflow = builder.compile()


# ============================================================
# PUBLIC FUNCTION
# ============================================================


def run_workflow(message: str) -> str:

    result = workflow.invoke({"message": message})

    return result["final_response"]
