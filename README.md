# ⚡ Regulatory AI Assistant for Indian Power Sector

> An Advanced Retrieval-Augmented Generation (RAG) System for Regulatory Intelligence, Tariff Petitions, ARR Filings, True-Up Petitions, Regulatory Orders, and Electricity Regulations.

---

## 📌 Overview

The Indian Power Sector generates large volumes of regulatory documents such as:

- Tariff Petitions
- Annual Revenue Requirement (ARR) Filings
- True-Up Petitions
- Annual Performance Review (APR) Filings
- Tariff Orders
- Regulatory Orders
- Electricity Regulations
- Commission Directives

These documents are typically:

- 100–1000+ pages long
- Rich in financial tables
- Filled with regulatory references
- Difficult to search manually
- Highly domain-specific

This project builds an AI-powered Regulatory Assistant that enables users to ask natural language questions and receive grounded, citation-backed answers directly from regulatory documents.

### Example Questions

```text
What ARR has JUSNL projected for FY 2025-26?

Why has JUSNL filed this petition?

What employee expenses has JUSNL projected?

What regulations has JUSNL relied upon?

What depreciation expenses has JUSNL claimed?

What capital expenditure schemes has JUSNL proposed?
```

---

# 🎯 Objectives

The primary objective of this project is to:

- Reduce manual analysis of regulatory documents
- Enable natural language search over petitions
- Improve regulatory intelligence workflows
- Extract financial and regulatory insights quickly
- Provide grounded responses with citations
- Build a scalable Regulatory Knowledge Assistant

---

# 🚀 Key Features

## 1. Semantic Search

Uses transformer-based embeddings to retrieve semantically relevant sections even when exact keywords are absent.

Example:

```text
User Query:
What ARR has JUSNL projected?

Retrieved Section:
5.11 ARR for FY 2025-26
```

---

## 2. Hybrid Search

Combines:

### Dense Retrieval

Vector similarity search using embeddings.

### Sparse Retrieval

Traditional keyword-based retrieval using MongoDB Full Text Search.

Benefits:

- Better Recall
- Better Precision
- Stronger performance on financial tables

---

## 3. Dynamic Metadata Filtering

LLM automatically extracts structured filters from user queries.

### Example

Question:

```text
What employee expenses has JUSNL projected?
```

Generated Filter:

```json
{
  "metadata.discom": "JUSNL",
  "metadata.cost_head": "employee_expense"
}
```

Question:

```text
What ARR has JUSNL projected for FY 2025-26?
```

Generated Filter:

```json
{
  "metadata.discom": "JUSNL",
  "metadata.filing_year": "FY 2025-26"
}
```

Benefits:

- Improved retrieval precision
- Reduced irrelevant chunks
- Better citation quality

---

# 🧠 Query Understanding Layer

Before retrieval, the system performs query understanding.

---

## Query Classification

Classifies user queries into:

### Numerical Queries

Examples:

```text
What ARR has JUSNL projected?

What depreciation expenses has JUSNL claimed?

What employee expenses has JUSNL projected?
```

### Conceptual Queries

Examples:

```text
What is JUSNL?

What regulations has JUSNL relied upon?
```

### Reasoning Queries

Examples:

```text
Why is JUSNL filing provisional true-up?

Why was a particular cost approved?
```

---

# 🔄 Query Rewriting Techniques

The system supports multiple query expansion techniques.

---

## 1. Multi Query Retrieval

Generates multiple paraphrases of the original query.

### Example

Original Query:

```text
What ARR has JUSNL projected?
```

Generated Variants:

```text
What Annual Revenue Requirement has JUSNL projected?

What revenue requirement has JUSNL forecasted?

What ARR has been proposed by JUSNL?

ARR filing for FY 2025-26
```

Benefits:

- Improves Recall
- Retrieves semantically diverse chunks
- Best-performing strategy for financial questions

---

## 2. HyDE (Hypothetical Document Embedding)

Generates a hypothetical answer first and retrieves documents based on that generated answer.

### Example

Question:

```text
What ARR has JUSNL projected?
```

Generated Hypothetical Answer:

```text
JUSNL has projected an ARR for FY 2025-26 as part of its tariff filing submitted before the Hon’ble Commission...
```

Benefits:

- Better for conceptual queries
- Strong semantic retrieval

---

## 3. Step-back Prompting

Generates a broader abstraction of the user query.

Example:

```text
Specific:
What ARR has JUSNL projected?

Step-back:
How is ARR determined for transmission utilities?
```

Benefits:

- Useful for reasoning questions
- Provides broader context

---

# 📚 Document Processing Pipeline

```text
PDF
 ↓
Parsing
 ↓
Cleaning
 ↓
Chunking
 ↓
Metadata Enrichment
 ↓
Embedding Generation
 ↓
MongoDB Storage
```

---

## Metadata Stored

Each chunk contains:

```json
{
  "text": "...",
  "embedding": [...],
  "metadata": {
    "discom": "JUSNL",
    "page_number": 52,
    "section_heading": "5.11 ARR for FY 2025-26",
    "document_type": "petition",
    "filing_year": "FY 2025-26"
  }
}
```

---

# 🗄️ Vector Database

MongoDB Atlas Vector Search is used as the vector database.

### Stores

- Chunk Text
- Embeddings
- Metadata

### Embedding Model

```text
BAAI/bge-large-en-v1.5
```

Vector Dimension:

```text
1024
```

---

# 🔍 Retrieval Pipeline

```text
User Query
     ↓
Metadata Extraction
     ↓
Query Rewriting
     ↓
Hybrid Search
     ↓
RRF Fusion
     ↓
MMR Diversification
     ↓
Cross Encoder Re-ranking
     ↓
Answer Generation
```

---

# Hybrid Retrieval

Combines:

```text
Vector Search
+
Full Text Search
```

Benefits:

- Semantic matching
- Exact keyword matching
- Better performance on regulatory tables

---

# Reciprocal Rank Fusion (RRF)

Combines results from multiple retrieval strategies.

Formula:

```text
RRF Score = Σ (1 / (k + rank))
```

Typical value:

```text
k = 60
```

Benefits:

- Robust ranking
- Query diversification
- Better Recall

---

# Maximum Marginal Relevance (MMR)

MMR removes redundant chunks.

Balances:

```text
Relevance
+
Diversity
```

Benefits:

- Better context coverage
- Reduced duplication
- Improved LLM context quality

---

# Cross Encoder Re-ranking

Final stage of retrieval.

Model:

```text
BAAI/bge-reranker-large
```

Pipeline:

```text
Retrieve Top 50 Chunks
      ↓
RRF
      ↓
MMR
      ↓
Cross Encoder
      ↓
Top 5 Chunks
```

Benefits:

- Better ranking quality
- Improved answer accuracy

---

# 🤖 Answer Generation

The final answer is generated using:

```text
Llama 3.3 70B
```

via:

```text
Groq API
```

The model is instructed to:

- Use only retrieved evidence
- Avoid hallucinations
- Cite sources
- Extract exact values from tables
- Preserve financial figures

---

# 🏗️ System Architecture

```text
User Query
      │
      ▼
Query Understanding
      │
      ▼
Metadata Extraction
      │
      ▼
Query Rewriting
(Multi Query / HyDE / Step-back)
      │
      ▼
Hybrid Search
(Vector + Text)
      │
      ▼
RRF Fusion
      │
      ▼
MMR Diversification
      │
      ▼
Cross Encoder Re-ranking
      │
      ▼
Evidence Extraction
      │
      ▼
Answer Generation
      │
      ▼
Final Response
```

---

# 🛠️ Tech Stack

## LLM

```text
Llama 3.3 70B
```

Provider:

```text
Groq
```

---

## Embedding Model

```text
BAAI/bge-large-en-v1.5
```

Dimension:

```text
1024
```

---

## Re-ranking Model

```text
BAAI/bge-reranker-large
```

---

## Vector Database

```text
MongoDB Atlas Vector Search
```

---

## Frameworks

- LangChain
- SentenceTransformers
- PyMongo
- HuggingFace
- LangSmith

---

# 📂 Project Structure

```text
project/
│
├── app.py
│
├── ingestion/
│   ├── parser.py
│   ├── chunking.py
│   ├── embedding.py
│
├── retrieval/
│   ├── hybrid_search.py
│   ├── rrf.py
│   ├── mmr.py
│   ├── reranker.py
│
├── query_understanding/
│   ├── metadata_filter.py
│   ├── query_rewriter.py
│
├── prompts/
│   ├── answer_prompt.py
│   ├── rewrite_prompt.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── metrics.py
│
└── utils/
```

---

# ⚙️ Installation

Clone repository:

```bash
git clone <repo-url>
cd regulatory-rag
```

Create Environment:

```bash
conda create -n regulatory-rag python=3.11
conda activate regulatory-rag
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file.

```env
MONGO_URI=

GROQ_API_KEY=

LANGCHAIN_API_KEY=

LANGCHAIN_PROJECT=
```

---

# ▶️ Running Application

```bash
python app.py
```

---

# 📊 Evaluation Metrics

The retrieval system is evaluated using:

### Hit@K

```text
Hit@1
Hit@3
Hit@5
```

### MRR

```text
Mean Reciprocal Rank
```

### Retrieval Recall

```text
Relevant Chunks Retrieved
--------------------------------
Total Relevant Chunks
```

### Answer Grounding Score

Measures whether generated answers are supported by retrieved evidence.

---

# 🔮 Future Roadmap

## Query Router

Automatically route queries:

```text
Numerical Query
      ↓
Multi Query

Conceptual Query
      ↓
HyDE

Reasoning Query
      ↓
Step-back
```

---

## Agentic RAG

Planned LangGraph Architecture:

```text
Query Understanding Agent
        ↓
Metadata Agent
        ↓
Retrieval Agent
        ↓
Reranking Agent
        ↓
Answer Generation Agent
```

---

## Citation Grounding

Future versions will provide:

```text
Section Number

Paragraph Number

Table Number

Page Number
```

for every answer.

---

## Regulatory Knowledge Graph

Create relationships between:

```text
Utilities

Petitions

ARR

APR

True-Up

Tariff Orders

Regulations
```

---

# 📈 Business Impact

This system significantly reduces manual effort involved in regulatory analysis and enables:

- Faster Petition Analysis
- ARR Validation
- Regulatory Compliance Review
- Tariff Intelligence
- Financial Projection Tracking
- Commission Order Analysis
- Regulatory Knowledge Discovery

---

# 👨‍💻 Author

**Bhavesh Kumar**

Data Scientist | Machine Learning Engineer | GenAI Engineer

### Focus Areas

- Retrieval-Augmented Generation (RAG)
- Agentic AI
- Regulatory Intelligence
- Time Series Forecasting
- Power Market Analytics
- Large Language Models

---

# 📜 License

MIT License