You are an expert AI Engineer. Build a complete Retrieval-Augmented Generation system from scratch.

Project goal:
Design and implement a RAG system capable of answering user questions using a dedicated domain-specific knowledge base. The system must produce grounded answers with citations/references to the retrieved sources.

Use case:
Build the system around a domain-specific AI topic. Prefer one of these datasets:
1. https://ai-2027.com/
2. A curated set of recent AI papers, blog posts, or reports, preferably published after the LLM training cutoff date.

Core requirements:
- Scrape or download the source content.
- Clean and preprocess the text.
- Chunk the documents.
- Create embeddings.
- Store chunks in a vector database.
- Build a retrieval pipeline.
- Build a generation pipeline that answers questions using retrieved context.
- Include citations in the final answer.
- Create an evaluation pipeline.
- Compare a base LLM answer versus the RAG-enhanced answer.
- Show that the RAG system outperforms the base LLM.

Preferred stack:
- Python
- FastAPI for API
- Streamlit for simple UI
- LangChain or LlamaIndex for RAG orchestration
- FAISS for vector store
- OpenAI, Anthropic, or local Ollama-compatible LLM
- sentence-transformers or OpenAI embeddings
- RAGAS or custom evaluation metrics
- pytest for tests

Expected repository structure:

rag-system/
  README.md
  requirements.txt
  .env.example
  src/
    config.py
    ingestion/
      scraper.py
      loader.py
      cleaner.py
      chunker.py
    indexing/
      embeddings.py
      vector_store.py
      build_index.py
    rag/
      retriever.py
      generator.py
      pipeline.py
      prompts.py
    evaluation/
      qa_dataset.py
      evaluate_ragas.py
      evaluate_retrieval.py
      compare_base_vs_rag.py
    api/
      main.py
    ui/
      app.py
  data/
    raw/
    processed/
    qa/
  tests/
    test_chunking.py
    test_retrieval.py
    test_rag_pipeline.py

Implementation details:

1. Data ingestion
- Implement a scraper/downloader for the selected public source.
- Store raw documents in data/raw/.
- Each document should preserve metadata:
  - source_url
  - title
  - published_date if available
  - author if available
  - section/page
  - ingestion_timestamp

2. Cleaning
- Remove boilerplate, navigation text, repeated headers/footers, empty lines, HTML artifacts.
- Normalize whitespace.
- Keep useful document structure such as headings and paragraphs.

3. Chunking
- Implement chunking with:
  - chunk_size around 700 to 1000 tokens
  - overlap around 100 to 200 tokens
- Preserve metadata for every chunk:
  - chunk_id
  - document_id
  - source_url
  - title
  - section
  - chunk_index

4. Embeddings and vector store
- Generate embeddings for all chunks.
- Store chunks and metadata in ChromaDB or FAISS.
- Make the vector store persistent.
- Add a script:
  python -m src.indexing.build_index

5. Retrieval
- Implement top-k retrieval.
- Default k = 5.
- Include optional hybrid retrieval if possible:
  - vector similarity
  - keyword/BM25 retrieval
  - reranking if simple to add
- Return retrieved chunks with similarity score and metadata.

6. Generation
- Implement a RAG prompt that forces grounded answers.
- The model must:
  - answer only from retrieved context
  - cite sources
  - say “I don’t know based on the provided knowledge base” when context is insufficient
  - avoid unsupported claims

Use this prompt template:

System:
You are a careful domain-specific AI assistant. Answer the user question using only the provided context. If the context is insufficient, say that the knowledge base does not contain enough information. Always cite the source title and URL for each important claim.

User:
Question:
{question}

Context:
{retrieved_context}

Answer format:
- Direct answer
- Key evidence
- Sources

7. API
Create a FastAPI app with endpoints:
- POST /ask
  Input: question, top_k
  Output: answer, retrieved_contexts, citations
- POST /retrieve
  Input: question, top_k
  Output: retrieved chunks
- GET /health

8. UI
Create a simple Streamlit app:
- text input for question
- slider for top_k
- display final answer
- display retrieved chunks and source URLs
- display confidence/retrieval scores if available

9. Evaluation dataset
Create a small labeled QA dataset in data/qa/qa_dataset.json.
Each item should contain:
- question
- expected_answer
- gold_source_url
- gold_context_snippet

Create at least 10 QA examples.

10. Evaluation metrics
Implement RAGAS or equivalent evaluation focused on:
- Faithfulness
- Context precision
- Context recall
- Answer relevance

Also implement retrieval metrics:
- hit_rate@k
- recall@k
- whether gold context/source appears in top-k retrieved chunks

Also implement comparative accuracy:
- Generate answers using the base LLM without retrieval.
- Generate answers using the RAG system.
- Compare both against expected answers.
- Produce a report showing where RAG performs better.

11. Output reports
Generate:
- evaluation_report.md
- evaluation_results.json
- retrieval_metrics.json
- base_vs_rag_comparison.md

12. README
Write a clear README with:
- project overview
- architecture diagram in Mermaid
- setup instructions
- how to configure API keys
- how to ingest data
- how to build index
- how to run API
- how to run UI
- how to run evaluation
- explanation of metrics
- example questions and answers
- known limitations
- next improvements

13. Testing
Add basic tests for:
- text cleaning
- chunk creation
- vector retrieval
- RAG answer generation with mocked LLM if needed

14. Quality requirements
- Use clean modular Python code.
- Use type hints.
- Use environment variables for secrets.
- Do not hardcode API keys.
- Add helpful logging.
- Handle errors gracefully.
- Make the project runnable end-to-end.

15. Deliver the final implementation
After building the files, summarize:
- what was implemented
- how to run it
- what commands to execute
- where the evaluation results are located
- what assumptions were made

Important:
The final project must satisfy these success criteria:
- It uses a domain-specific knowledge base.
- It returns grounded answers with citations.
- It evaluates the RAG system using RAGAS or equivalent metrics.
- It evaluates retrieval using hit rate or recall@k.
- It compares base LLM vs RAG.
- It demonstrates that the RAG system improves answer quality over the base LLM.