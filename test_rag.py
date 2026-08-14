from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


VECTOR_DIR = "vectorstore"


# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Load existing ChromaDB
db = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)


# Ask a question
query = "What is DRDO?"


# Find the 3 most relevant chunks
results = db.similarity_search(
    query,
    k=3
)


print("\nQuestion:")
print(query)

print("\nRelevant information:\n")

for i, document in enumerate(results, start=1):

    print(f"--- Result {i} ---")

    print("Source:", document.metadata.get("source"))

    print(document.page_content[:1000])

    print()