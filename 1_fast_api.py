"""
 why FastAPI is fast, explained simply:

👉 1. Built on Starlette (ASGI)
FastAPI is built on Starlette, an asynchronous web framework using ASGI (Asynchronous Server Gateway Interface). This lets it handle many requests at the same time without waiting—unlike older WSGI frameworks like Flask or Django, which process requests one at a time.

👉 2. Uses Pydantic for fast data validation
When you send JSON or form data, FastAPI uses Pydantic, a highly optimized library written partly in Cython, to parse and validate data much faster than pure Python libraries.

👉 3. Async support = no blocking I/O
By supporting async and await natively, FastAPI lets you call APIs, databases, or other I/O tasks without blocking the server. This means other requests don’t have to wait while one request is fetching something.

👉 4. Less overhead
Compared to older frameworks, FastAPI has a lighter core with fewer built-in layers of middleware, keeping it lean and direct—so requests pass through fewer steps internally.

👉 5. Modern Python optimizations
Because it uses type hints and modern Python features, FastAPI can pre-check and compile some operations at runtime, reducing errors and unnecessary work at request time.

In short, it’s fast because it’s asynchronous, optimized, and uses modern tools designed for speed from the start.
"""


from fastapi import FastAPI, requests
import pickle

from pydantic import BaseModel

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    class_name: str

with open("model.pkl", "rb") as f:
    model_rf = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(irisData: IrisData):
    result = model_rf.predict(
        [[
            irisData.sepal_length, 
            irisData.sepal_width, 
            irisData.petal_length,
            irisData.petal_width
        ]]
    )
    return IrisResponse(class_name=str(result[0]))
    
