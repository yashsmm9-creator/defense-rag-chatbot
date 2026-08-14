import os

from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError(
        "NVIDIA_API_KEY not found. "
        "Please add it to your .env file."
    )


# ============================================================
# 2. CONNECT TO NVIDIA NIM
# ============================================================

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ============================================================
# 4. LOAD EXISTING CHROMADB
# ============================================================

db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)


# ============================================================
# 5. GET USER QUESTION
# ============================================================

query = input("\nAsk your question: ")


# ============================================================
# 6. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

results = db.similarity_search(
    query,
    k=3
)


# ============================================================
# 7. CREATE CONTEXT FROM RETRIEVED DOCUMENTS
# ============================================================

context_parts = []

for i, document in enumerate(results, start=1):

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    context_parts.append(
        f"""
SOURCE {i}: {source}

{document.page_content}
"""
    )


context = "\n".join(context_parts)


# ============================================================
# 8. RAG PROMPT
# ============================================================

prompt = f"""
You are Defense-RAG, an AI assistant for defense and
security information.

Your job is to answer questions using ONLY the information
provided in the retrieved knowledge base below.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not invent or assume facts.
3. If the answer is not present in the context, say:
   "I could not find this information in the knowledge base."
4. Give a clear and concise answer.
5. Do not provide instructions that could enable someone
   to carry out harmful or illegal activities.
6. When possible, mention the source document used.

RETRIEVED KNOWLEDGE:

{context}


USER QUESTION:

{query}
"""


# ============================================================
# 9. SEND RAG PROMPT TO NVIDIA
# ============================================================

try:

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful and factual "
                    "defense information assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=1024
    )


    # ========================================================
    # 10. GET AI ANSWER
    # ========================================================

    answer = completion.choices[0].message.content


    print("\n" + "=" * 60)
    print("DEFENSE-RAG AI ANSWER")
    print("=" * 60)

    print(answer)


    # ========================================================
    # 11. SHOW SOURCES
    # ========================================================

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for i, document in enumerate(results, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        print(f"{i}. {source}")


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(e)