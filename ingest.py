from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Folder containing your 209 text files
DATA_DIR = Path("data/extracted")

# Where ChromaDB will store the vectors
VECTOR_DIR = "vectorstore"


# -----------------------------
# STEP 1: Load documents
# -----------------------------

documents = []

for file in DATA_DIR.glob("*.txt"):
    try:
        loader = TextLoader(
            str(file),
            encoding="utf-8"
        )

        docs = loader.load()

        # Save filename as metadata
        for doc in docs:
            doc.metadata["source"] = file.name

        documents.extend(docs)

        print(f"Loaded: {file.name}")

    except Exception as e:
        print(f"Error loading {file.name}: {e}")


print("\nTotal documents loaded:", len(documents))


# -----------------------------
# STEP 2: Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("Total chunks created:", len(chunks))


# -----------------------------
# STEP 3: Create embeddings
# -----------------------------

print("\nCreating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# -----------------------------
# STEP 4: Store in ChromaDB
# -----------------------------

print("Creating ChromaDB...")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DIR
)


print("\n================================")
print("RAG DATABASE CREATED SUCCESSFULLY")
print("================================")
print("Documents:", len(documents))
print("Chunks:", len(chunks))