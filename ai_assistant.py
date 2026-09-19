import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

load_dotenv()


# LANGSMITH = fregrekrfjner7548952p3443258954594532458745234355544

class State(TypedDict):
    question: str
    answer: str


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))


def answer_question(state: State):
    response = llm.invoke(state["question"])

    return {
        "answer": response.content
    }


builder = StateGraph(State)

builder.add_node("answer", answer_question)

builder.add_edge(START, "answer")
builder.add_edge("answer", END)

workflow = builder.compile()


def run_workflow(question: str):
    result = workflow.invoke({
        "question": question
    })

    return result["answer"]