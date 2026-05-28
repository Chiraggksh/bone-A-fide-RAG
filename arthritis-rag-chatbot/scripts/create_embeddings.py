import os
import pickle
from tqdm import tqdm
import numpy as np
import faiss

from sentence_transformers import (
    SentenceTransformer
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# ==========================
# PATHS
# ==========================

TEXT_DIR = "data/processed_text"
VECTOR_DIR = "vector_store"

os.makedirs(
    VECTOR_DIR,
    exist_ok=True
)

FAISS_PATH = os.path.join(
    VECTOR_DIR,
    "faiss_index.bin"
)

CHUNKS_PATH = os.path.join(
    VECTOR_DIR,
    "chunks.pkl"
)

# ==========================
# CONFIG
# ==========================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBED_BATCH_SIZE = 64

# ==========================
# LOAD MODEL
# ==========================

print(
    "Loading embedding model..."
)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ==========================
# TEXT SPLITTER
# ==========================

splitter = (
    RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
)

# ==========================
# LOAD TEXT FILES
# ==========================

text_files = [
    file
    for file in os.listdir(
        TEXT_DIR
    )
    if file.endswith(".txt")
]

print(
    f"Found "
    f"{len(text_files)} "
    f"text files"
)

all_chunks = []
chunk_metadata = []

# ==========================
# CREATE CHUNKS
# ==========================

print(
    "\nCreating chunks..."
)

failed_files = 0

for file in tqdm(
    text_files
):

    file_path = os.path.join(
        TEXT_DIR,
        file
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        # Skip tiny text
        if len(text) < 500:
            continue

        chunks = splitter.split_text(
            text
        )

        for chunk in chunks:

            # Skip useless chunks
            if len(chunk.strip()) < 100:
                continue

            all_chunks.append(
                chunk
            )

            chunk_metadata.append(
                {
                    "source_file":
                        file
                }
            )

    except Exception as e:

        failed_files += 1

        print(
            f"\nError "
            f"in {file}: {e}"
        )

print(
    f"\nCreated "
    f"{len(all_chunks)} "
    f"chunks"
)

print(
    f"Failed files: "
    f"{failed_files}"
)

# ==========================
# CREATE EMBEDDINGS
# ==========================

print(
    "\nGenerating embeddings..."
)

all_embeddings = []

for i in tqdm(
    range(
        0,
        len(all_chunks),
        EMBED_BATCH_SIZE
    ),
    desc="Embedding batches"
):

    batch = all_chunks[
        i:
        i + EMBED_BATCH_SIZE
    ]

    batch_embeddings = (
        model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    )

    all_embeddings.append(
        batch_embeddings
    )

embeddings = np.vstack(
    all_embeddings
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

# ==========================
# CREATE FAISS INDEX
# ==========================

print(
    "\nCreating FAISS index..."
)

dimension = (
    embeddings.shape[1]
)

# Better for semantic search
index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

# ==========================
# SAVE VECTOR STORE
# ==========================

print(
    "\nSaving vector store..."
)

faiss.write_index(
    index,
    FAISS_PATH
)

with open(
    CHUNKS_PATH,
    "wb"
) as f:

    pickle.dump(
        {
            "chunks":
                all_chunks,

            "metadata":
                chunk_metadata
        },
        f
    )

# ==========================
# FINAL REPORT
# ==========================

print("\n" + "=" * 50)

print(
    "EMBEDDINGS COMPLETE"
)

print("=" * 50)

print(
    f"Total chunks: "
    f"{len(all_chunks)}"
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}"
)

print(
    "FAISS index saved"
)

print(
    f"Saved at: "
    f"{VECTOR_DIR}"
)