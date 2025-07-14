import sys
from chromadb import PersistentClient
from embed import embed

query = sys.argv[1]

chroma = PersistentClient(path="./chroma_db")
collection = chroma.get_collection("iss_knowledge")

embedding = embed(query)
results = collection.query(query_embeddings=[embedding], n_results=1)

if results["documents"]:
    print(results["documents"][0][0])
else:
    print("NO_MATCH")
