# 🩺 Bone-afide RAG – Medical Research Assistant

A Retrieval-Augmented Generation (RAG) based medical chatbot that answers research-backed queries related to **arthritis, osteoporosis, and rheumatoid arthritis** using scientific literature embeddings and LLM reasoning.

It combines **semantic search + reranking + LLM generation** to produce context-aware, evidence-grounded responses.

---

## Features

*  Semantic search using **FAISS vector database**
*  High-quality embeddings using **Sentence Transformers (BGE-small)**
*  Re-ranking with **Cross-Encoder (MS MARCO MiniLM)**
*  Answer generation using **Google Gemini API**
*  Conversational memory (chat history context handling)
*  Interactive UI built with **Streamlit**
*  Research-based responses from scientific chunks
*  Fast retrieval pipeline optimized for medical QA

---

## System Architecture

User Query
→ Query Embedding (SentenceTransformer)
→ FAISS Similarity Search
→ Top-K Retrieval
→ Cross-Encoder Re-ranking
→ Context Selection
→ Gemini LLM Generation
→ Final Answer

---

## Tech Stack

* Python
* FAISS (Vector Search)
* Sentence Transformers (Embeddings)
* Cross-Encoder (Re-ranking)
* Google Gemini API
* Streamlit (Frontend UI)
* NumPy, Pickle

---

## Project Structure

```
arthritis-rag-chatbot/
│
├── backend/
│   └── rag_engine.py
│
├── frontend/
│   └── app.py
│
├── scripts/
│   ├── create_embeddings.py
│   ├── preprocess.py
│   └── download_papers.py
│
├── data/ (ignored in git)
│   ├── raw_pdfs/
│   ├── processed_text/
│   └── metadata/
│
├── vector_store/ (ignored in git)
│   ├── faiss_index.bin
│   └── chunks.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone repository

```bash
git clone https://github.com/Chiraggksh/bone-A-fide-RAG.git
cd arthritis-rag-chatbot
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### 5. Run Streamlit app

```bash
streamlit run frontend/app.py
```

---

## How It Works

1. Medical papers are split into chunks
2. Each chunk is converted into embeddings
3. FAISS stores embeddings for fast retrieval
4. Query is embedded and matched with relevant chunks
5. Cross-encoder reranks results for accuracy
6. Gemini generates final answer using context

---

## Important Notes

* `vector_store/` is excluded due to size limits (>100MB files)
* Raw PDFs are not included in repository
* Embeddings can be regenerated using scripts
* Gemini API has rate limits (free tier)

---

## Example Queries

* What are biomarkers for osteoporosis?
* Symptoms of rheumatoid arthritis?
* Difference between osteoarthritis and rheumatoid arthritis?
* Latest treatments for bone degeneration?
* How does bone mineral density affect fracture risk?

---

## Future Improvements

* Add medical citation highlighting in UI
* Use hybrid search (BM25 + FAISS)
* Deploy on Streamlit Cloud / HuggingFace Spaces
* Add PDF upload for dynamic indexing
* Improve caching for embeddings and reranking

---

## Author

**Chirag Kaushik**
B.Tech (AI & Data Science)
