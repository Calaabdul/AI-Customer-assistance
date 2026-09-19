from fastapi import FastAPI
from pydantic import BaseModel

from customer_support import run_workflow


app = FastAPI(title="Customer Support AI", version="1.0.0")


class CustomerRequest(BaseModel):
    message: str


class CustomerResponse(BaseModel):
    response: str

#display the status of the service on home page
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Customer Support AI",
    }


@app.post("/chat", response_model=CustomerResponse)
def chat(request: CustomerRequest):

    response = run_workflow(request.message)

    return CustomerResponse(response=response)