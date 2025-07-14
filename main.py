from fastapi import FastAPI, Request
from chromadb import PersistentClient
from embed import embed
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

chroma = PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("iss_knowledge")

@app.post("/query")
async def query_handler(request: Request):
    data = await request.json()
    question = data.get("message")
    embedding = embed(question)

    results = collection.query(query_embeddings=[embedding], n_results=1, include=["distances", "documents"])
    print("💬 Question:", question)
    print("🧠 Distance:", results["distances"][0][0])

    if results["distances"][0][0] < 0.3:
        return {"reply": results["documents"][0][0]}
    else:
        return {"reply": "I'm still learning how to answer that! 😊"}
