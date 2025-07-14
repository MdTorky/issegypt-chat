import sys
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

query = sys.argv[1]

chroma = PersistentClient(path="./chroma_db")
collection = chroma.get_collection("iss_knowledge")

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode(query).tolist()
results = collection.query(query_embeddings=[embedding], n_results=1)

if results["documents"]:
    print(results["documents"][0][0])
else:
    print("NO_MATCH")
