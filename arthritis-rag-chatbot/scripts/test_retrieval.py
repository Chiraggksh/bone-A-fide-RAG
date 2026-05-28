import time
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# ==========================
# CONFIG
# ==========================

FAISS_PATH = "vector_store/faiss_index.bin"
CHUNKS_PATH = "vector_store/chunks.pkl"

TOP_K = 5

# similarity threshold
MAX_DISTANCE = 1.2

# ==========================
# LOAD MODEL
# ==========================

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ==========================
# LOAD VECTOR STORE
# ==========================

print("Loading vector database...")

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

print(
    f"Loaded {len(chunks)} chunks"
)

# ==========================
# RETRIEVAL
# ==========================

def retrieve(query):

    start_time = time.time()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    end_time = time.time()

    retrieval_time = (
        end_time - start_time
    )

    print("\n" + "=" * 80)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 80)

    relevant_found = False

    for rank, idx in enumerate(
        indices[0]
    ):

        distance = (
            distances[0][rank]
        )

        if distance > MAX_DISTANCE:
            continue

        relevant_found = True

        print(
            f"\nResult #{rank + 1}"
        )

        source = metadata[idx].get(
            "source_file",
            "Unknown"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Distance: "
            f"{distance:.4f}"
        )

        print("\nChunk Preview:\n")

        preview = (
            chunks[idx]
            .replace("\n", " ")
            [:800]
        )

        print(preview)

        print(
            "\n" + "-" * 80
        )

    if not relevant_found:
        print(
            "\nNo strong matches found."
        )

    print(
        f"\nRetrieval time: "
        f"{retrieval_time:.2f}s"
    )


# ==========================
# SEARCH LOOP
# ==========================

while True:

    query = input(
        "\nAsk question "
        "(or type exit): "
    )

    if query.lower() == "exit":
        break

    retrieve(query)