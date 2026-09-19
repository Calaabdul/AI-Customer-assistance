from fastapi import FastAPI
from pydantic import BaseModel

from ai_assistant import run_workflow


app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(data: Question):

    answer = run_workflow(data.question)

    return {
        "answer": answer
    }



# def sum(a, b):
#     return a // b

