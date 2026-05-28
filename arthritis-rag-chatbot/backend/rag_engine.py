import os
import time
import pickle
import faiss
import streamlit as st
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

import google.generativeai as genai

# ==========================
# CONFIG
# ==========================

TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 4

MIN_RERANK_SCORE = 1.8
MAX_HISTORY = 10

FAISS_PATH = "vector_store/faiss_index.bin"
CHUNKS_PATH = "vector_store/chunks.pkl"

# ==========================
# LOAD ENV
# ==========================

load_dotenv()

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

genai.configure(
    api_key=API_KEY
)

# ==========================
# LOAD MODELS
# ==========================

@st.cache_resource
def load_models():

    embedding_model = (
        SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    llm = genai.GenerativeModel(
        "gemini-1.5-flash"
    )

    return (
        embedding_model,
        reranker,
        llm
    )


@st.cache_resource
def load_vector_db():

    index = faiss.read_index(
        FAISS_PATH
    )

    with open(
        CHUNKS_PATH,
        "rb"
    ) as f:

        data = pickle.load(f)

    chunks = data["chunks"]
    metadata = data["metadata"]

    return (
        index,
        chunks,
        metadata
    )

embedding_model, reranker, llm = (
    load_models()
)

index, chunks, metadata = (
    load_vector_db()
)
# ==========================
# CHAT MEMORY
# ==========================

chat_history = []
conversation_topic = ""
topic_history = []

# ==========================
# QUERY REWRITE
# ==========================

def rewrite_query(user_query):

    global conversation_topic
    global topic_history

    lower_query = (
        user_query.lower().strip()
    )

    # ==========================
    # Follow-up terms
    # ==========================

    vague_terms = [
        "it",
        "this",
        "these",
        "that",
        "which one",
        "they",
        "them",
        "its",
        "this process",
        "that process",
        "this treatment",
        "that treatment",
        "does it",
        "can it",
        "how does it",
        "what about that",
        "can treatment",
        "does treatment"
    ]

    # ==========================
    # New topic phrases
    # ==========================

    new_topic_phrases = [
        "tell me about",
        "what is",
        "explain",
        "define",
        "overview of",
        "symptoms of",
        "treatment of",
        "causes of",
        "biomarkers of"
    ]

    # ==========================
    # Medical entities
    # ==========================

    medical_keywords = [
        "arthritis",
        "osteoarthritis",
        "rheumatoid arthritis",
        "osteoporosis",
        "bone fracture",
        "cartilage",
        "joint",
        "bone",
        "skeletal",
        "musculoskeletal"
    ]

    # ==========================
    # Detect NEW topic
    # ==========================

    is_new_topic = False

    if any(
        phrase in lower_query
        for phrase in new_topic_phrases
    ):
        is_new_topic = True

    if any(
        keyword in lower_query
        for keyword in medical_keywords
    ):
        is_new_topic = True

    # ==========================
    # TOPIC SWITCH
    # ==========================

    if is_new_topic:

        conversation_topic = user_query

        topic_history.append(
            user_query
        )

        return user_query

    # ==========================
    # Follow-up detection
    # ==========================

    is_followup = any(
        term in lower_query
        for term in vague_terms
    )

    if (
        is_followup
        and conversation_topic
    ):

        return (
            f"{user_query}\n"
            f"Context Topic: "
            f"{conversation_topic}"
        )

    return user_query


# ==========================
# RETRIEVE + RERANK
# ==========================

def retrieve_context(
    query
):

    start = time.time()

    # ----------------------
    # Embed query
    # ----------------------

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    # ----------------------
    # Retrieve top 20
    # ----------------------

    distances, indices = (
        index.search(
            query_embedding,
            TOP_K_RETRIEVAL
        )
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append({
            "chunk":
                chunks[idx],

            "metadata":
                metadata[idx]
        })

    # ----------------------
    # Rerank chunks
    # ----------------------

    pairs = []

    for item in (
        retrieved_chunks
    ):

        pairs.append(
            (
                query,
                item["chunk"]
            )
        )

    scores = reranker.predict(
        pairs
    )

    reranked = []

    for item, score in zip(
        retrieved_chunks,
        scores
    ):

        reranked.append({
            "chunk":
                item["chunk"],

            "metadata":
                item["metadata"],

            "score":
                score
        })

    reranked = sorted(
        reranked,
        key=lambda x:
            x["score"],
        reverse=True
    )

    best_score = reranked[0]["score"]

    if best_score < MIN_RERANK_SCORE:

        return None

    final_chunks = (
        reranked[:TOP_K_RERANK]
    )

    end = time.time()

    print(
        "\n" + "=" * 80
    )

    print(
        "TOP RERANKED CHUNKS"
    )

    print(
        "=" * 80
    )

    for rank, item in enumerate(
        final_chunks
    ):

        source = (
            item["metadata"]
            .get(
                "source_file",
                "Unknown"
            )
        )

        print(
            f"\nResult "
            f"#{rank + 1}"
        )

        print(
            f"Source: "
            f"{source}"
        )

        print(
            f"Rerank Score: "
            f"{item['score']:.4f}"
        )

        print(
            "\nChunk:\n"
        )

        print(
            item["chunk"][:700]
        )

        print(
            "\n" +
            "-" * 80
        )

    print(
        f"\nRetrieval + "
        f"Rerank Time: "
        f"{end - start:.2f}s"
    )

    compressed_context = []

    for item in final_chunks:

        chunk = item["chunk"][:1000]

        compressed_context.append(
            chunk
        )

    context = "\n\n".join(
        compressed_context
    )

    return context


# ==========================
# CHAT LOOP
# ==========================

def ask_question(user_query):

    global chat_history

    rewritten_query = (
        rewrite_query(
            user_query
        )
    )

    print(
        f"Current Topic: "
        f"{conversation_topic}"
    )

    print(
        f"\nRewritten Query: "
        f"{rewritten_query}"
    )

    retrieved_context = (
        retrieve_context(
            rewritten_query
        )
    )

    if retrieved_context is None:

        return (
            "This question seems "
            "outside the medical "
            "research knowledge base."
        )

    recent_history = (
        chat_history[
            -MAX_HISTORY:
        ]
    )

    history_text = ""

    for msg in recent_history:

        history_text += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    prompt = f"""
You are an expert medical
research assistant.

Use:

1. Chat history
2. Retrieved papers

IMPORTANT:

- Use research context
as primary evidence.

- Understand references
like:

"it"
"this"
"which one"

using chat history.

- Infer carefully when
exact wording is absent.

- Do NOT hallucinate.

- Answer naturally and clearly.
- Explain medical concepts in simple language.
- Do not sound overly robotic or academic.
- Summarize research findings in a human-friendly way.
- If possible, combine retrieved evidence into one coherent explanation.

- If evidence is weak,
say so honestly.

==================
CHAT HISTORY
==================

{history_text}

==================
RESEARCH CONTEXT
==================

{retrieved_context}

==================
QUESTION
==================

{user_query}
"""

    print("\nThinking...\n")

    try:

        response = (
            llm.generate_content(
                prompt
            )
        )

        answer = response.text

    except Exception as e:

        answer = (
            "⚠️ Gemini API quota exceeded "
            "or temporary issue.\n\n"
            "Please try again later."
        )

        print(e)

    chat_history.append({
        "role": "user",
        "content":
            user_query
    })

    chat_history.append({
        "role": "assistant",
        "content":
            answer
    })

    if (
        len(chat_history)
        > 20
    ):
        chat_history = (
            chat_history[-20:]
        )

    return answer