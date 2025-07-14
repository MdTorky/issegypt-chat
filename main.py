from fastapi import FastAPI, Request
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

app = FastAPI()

chroma = PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("iss_knowledge")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

@app.post("/query")
async def query_handler(request: Request):
    data = await request.json()
    question = data.get("message")
    embedding = model.encode(question).tolist()

    results = collection.query(query_embeddings=[embedding], n_results=1, include=["distances", "documents"])

    print("💬 Question:", question)
    print("🧠 Distance:", results["distances"][0][0])

    # Lower = better (0 is exact match). Try threshold = 0.3
    if results["distances"][0][0] < 0.3:
        return {"reply": results["documents"][0][0]}
    else:
        return {"reply": "I'm still learning how to answer that! 😊"}

