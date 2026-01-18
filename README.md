# 🔍 NLP Semantic Search Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-Web%20App-green) ![NLP](https://img.shields.io/badge/NLP-Word%20Embeddings-orange)

A powerful semantic search engine built with **Flask** that goes beyond simple keyword matching. This application uses advanced Natural Language Processing (NLP) techniques to understand the *meaning* behind your query and retrieves the most relevant documents using vector similarity.

## 🚀 Features

https://github.com/user-attachments/assets/101cd20d-a700-4f48-a563-3c10eb60a693


* **🧠 Multi-Model Intelligence:** Switch dynamically between four distinct embedding models to see how results differ:
    * **GloVe:** Global Vectors for Word Representation.
    * **Word2Vec (Skipgram):** Custom implementation of the Skipgram architecture.
    * **Word2Vec (Negative Sampling):** Optimized Skipgram with Negative Sampling.
    * **Gensim:** Standard Gensim library implementation.
* **⚡ Real-Time Similarity:** Calculates Cosine Similarity/Dot Product on the fly to rank documents.
* **🎨 Interactive UI:** A clean, responsive web interface to input queries and view scores instantly.
* **📂 Document Retrieval:** Returns the top 5 most semantically similar documents from the corpus.

---

## 🛠️ Installation & Setup

Follow these steps to get the search engine running on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/Santhosh01161/nlp-semantic-search.git](https://github.com/Santhosh01161/nlp-semantic-search-engine.git)
cd nlp-semantic-search


