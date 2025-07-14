

# from chromadb import PersistentClient
# from chromadb.config import Settings
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer

# client = MongoClient("mongodb://issEAC:iss-eacademic-24@ac-bh3n5pj-shard-00-00.myjfgpj.mongodb.net:27017,ac-bh3n5pj-shard-00-01.myjfgpj.mongodb.net:27017,ac-bh3n5pj-shard-00-02.myjfgpj.mongodb.net:27017/?ssl=true&replicaSet=atlas-i70w1b-shard-0&authSource=admin&retryWrites=true&w=majority")
# db = client["iss"]
# collection = db["knowledges"]

# chroma = PersistentClient(path="./chroma_db")

# # Reset collection
# try:
#     chroma.delete_collection("iss_knowledge")
# except:
#     pass

# chroma_collection = chroma.get_or_create_collection("iss_knowledge")

# # Embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# docs = list(collection.find({}))
# for idx, item in enumerate(docs):
#     question = item["question"]
#     answer = item["answer"]
#     embedding = model.encode(question).tolist()

#     chroma_collection.add(
#         documents=[answer],
#         metadatas=[{"question": question}],
#         ids=[f"q{idx}"],
#         embeddings=[embedding]
#     )

# print("✅ ChromaDB knowledge saved.")


from chromadb import PersistentClient
from chromadb.config import Settings
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")


mongo = MongoClient(MONGO_URI)
db = mongo["test"]
collection = db["knowledges"]

# Start Chroma
chroma = PersistentClient(path="./chroma_db")
try:
    chroma.delete_collection("iss_knowledge")
except:
    pass
chroma_collection = chroma.get_or_create_collection("iss_knowledge")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Index data
docs = list(collection.find({}))
print(f"🔍 Found {len(docs)} knowledge items")
for idx, item in enumerate(docs):
    question = item["question"]
    answer = item["answer"]
    embedding = model.encode(question).tolist()
    chroma_collection.add(
        documents=[answer],
        metadatas=[{"question": question}],
        ids=[f"q{idx}"],
        embeddings=[embedding]
    )

print("✅ ChromaDB knowledge saved.")
