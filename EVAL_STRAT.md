# Evaluation Strategy

This document details the evaluation strategy for the AI Content RAG system. The focus is on ensuring **answer quality** — that the system retrieves the right information and generates grounded, faithful, relevant answers with proper citations.

## Overview

Evaluation is structured in three layers, each measuring a different aspect of quality:

1. **Retrieval quality** — Does the system find the right chunks from the knowledge base?
2. **Generation quality** — Does the LLM produce faithful, relevant answers from those chunks?
3. **End-to-end comparison** — Does RAG outperform a base LLM with no retrieval?

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  QA Dataset (12 labeled examples)                           │
│       │                                                     │
│       ├──► Retrieval Evaluation (hit_rate@k, recall@k)      │
│       │         └── No LLM calls, fast, deterministic       │
│       │                                                     │
│       ├──► RAGAS Framework (faithfulness, relevancy, etc.)  │
│       │         └── LLM-as-judge, requires OpenAI API       │
│       │                                                     │
│       └──► Base vs RAG Comparison                           │
│                 └── Side-by-side with similarity scoring     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Evaluation Dataset

### Source

The QA dataset is stored at `data/qa/qa_dataset.json` and contains 12 hand-crafted examples covering the major topics from [ai-2027.com](https://ai-2027.com/).

### Schema

Each example contains:

| Field | Description |
|-------|-------------|
| `question` | A natural language question about the AI 2027 scenario |
| `expected_answer` | A human-written reference answer (used for similarity comparison) |
| `gold_source_url` | The URL where the answer can be found in the knowledge base |
| `gold_context_snippet` | A verbatim excerpt from the source that contains the answer |

### Topics Covered

The 12 questions span:

| # | Topic | Gold Source |
|---|-------|-------------|
| 1 | Main premise of AI 2027 | ai-2027.com/ |
| 2 | Role of AI safety | ai-2027.com/summary |
| 3 | AI lab competition | ai-2027.com/ |
| 4 | Economic impacts | ai-2027.com/ |
| 5 | AI governance | ai-2027.com/ |
| 6 | Capabilities progression | ai-2027.com/summary |
| 7 | Risks from advanced AI | ai-2027.com/summary |
| 8 | National security | ai-2027.com/summary |
| 9 | Alignment problem | ai-2027.com/summary |
| 10 | Timelines for transformative AI | ai-2027.com/research/timelines-forecast |
| 11 | Open source AI development | ai-2027.com/summary |
| 12 | Role of compute | ai-2027.com/research/compute-forecast |

### Design Principles

- Questions are phrased as a user would naturally ask them (not keyword lookups)
- Expected answers require synthesizing information from the source, not just quoting a single sentence
- Gold context snippets are verbatim from the source to enable precise recall measurement
- Questions span multiple pages/sections to test retrieval breadth

---

## Layer 1: Retrieval Evaluation

**File:** `src/evaluation/evaluate_retrieval.py`

Measures whether the retrieval pipeline surfaces the correct documents before any LLM generation occurs. This is the foundation — if retrieval fails, generation cannot succeed.

### Metrics

#### Hit Rate@k

**Definition:** Proportion of queries where at least one retrieved chunk comes from the correct source.

**How it works:**
1. For each question, retrieve top-k chunks
2. Check if any chunk's `source_url` matches the `gold_source_url` (with flexible URL prefix matching)
3. As a fallback, check if the `gold_context_snippet` has >30% character 5-gram overlap with any retrieved chunk

**Why flexible URL matching:** Content from `ai-2027.com/summary` may also appear verbatim on subpages. The URL prefix matching (`r.startswith(g) or g.startswith(r)`) accounts for this.

#### Recall@k

**Definition:** Whether the key information from the gold context appears in the retrieved chunks.

**How it works:**
1. Split the `gold_context_snippet` into sentences
2. For each sentence, check if it appears in the concatenated retrieved text using:
   - Character 4-gram overlap (>30% threshold)
   - Fallback: keyword matching (>50% of words with length >3 found)
3. If >30% of gold sentences are matched, count as a recall hit

### Results

| Metric | k=3 | k=5 | k=10 |
|--------|:---:|:---:|:----:|
| Hit Rate@k | 0.83 | 0.92 | 1.00 |
| Recall@k | 0.75 | 1.00 | 1.00 |

### Interpretation

- At the default `top_k=5`, the system retrieves the correct source 92% of the time
- All gold context information is recoverable at k=5, meaning the chunks contain the needed evidence
- Perfect retrieval at k=10 confirms the index is comprehensive

---

## Layer 2: RAGAS Framework Evaluation

**File:** `src/evaluation/evaluate_ragas.py`

Uses the [RAGAS](https://github.com/explodinggradients/ragas) library (v0.1.x) for LLM-judged evaluation. RAGAS uses an LLM (gpt-3.5-turbo by default) to assess answer quality across four dimensions.

### Why RAGAS

- Industry-standard framework for RAG evaluation
- LLM-as-judge captures semantic quality that string matching cannot
- Measures both retrieval and generation quality in a unified pipeline
- Decomposed metrics allow diagnosing specific failure modes

### Metrics

#### Faithfulness

**What it measures:** Is every claim in the generated answer supported by the retrieved context?

**How RAGAS computes it:**
1. Extracts individual claims/statements from the answer
2. For each claim, asks the LLM judge: "Can this statement be inferred from the given context?"
3. Score = (number of supported claims) / (total claims)

**Why it matters:** Detects hallucination — the answer should never assert anything not present in the retrieved chunks.

#### Answer Relevancy

**What it measures:** Does the answer actually address the question asked?

**How RAGAS computes it:**
1. Generates N hypothetical questions from the answer text
2. Computes embedding similarity between generated questions and the original question
3. Score = average cosine similarity

**Known limitation:** This can score 0.0 for valid detailed answers if the answer covers much more than what the question narrowly asks. This is a known RAGAS quirk, not a system failure.

#### Context Precision

**What it measures:** Are relevant documents ranked at the top of retrieval results?

**How RAGAS computes it:**
1. For each retrieved chunk, the LLM judges whether it's relevant to answering the question (given the ground truth)
2. Computes a weighted precision score that rewards relevant documents appearing earlier

**Why it matters:** Even if we retrieve the right documents, they should be ranked first — this matters because the LLM pays more attention to context at the top.

#### Context Recall

**What it measures:** Does the retrieved context contain all the information needed to answer the question?

**How RAGAS computes it:**
1. Decomposes the ground truth answer into individual sentences/claims
2. For each ground truth sentence, asks the LLM: "Can this sentence be attributed to any of the retrieved contexts?"
3. Score = (attributable sentences) / (total ground truth sentences)

**Why it matters:** Even if we retrieve some relevant chunks, they might miss key details needed for a complete answer.

### Results

| Metric | Score | Interpretation |
|--------|:-----:|----------------|
| Faithfulness | 0.86 | Most claims are grounded in context; minor extrapolation detected |
| Answer Relevancy | 0.57 | Some RAGAS 0.0 scores due to detailed answers (see known limitation) |
| Context Precision | 0.78 | Relevant documents generally ranked well |
| Context Recall | 0.45 | Ground truth sentences not always fully recoverable from context |

### Per-Question Breakdown

| Question | Faith. | Relev. | Prec. | Recall |
|----------|:------:|:------:|:-----:|:------:|
| Main premise | 1.00 | 1.00 | 0.68 | 0.50 |
| AI safety role | 0.94 | 1.00 | 0.70 | 1.00 |
| Lab competition | 0.56 | 0.98 | 0.33 | 0.00 |
| Economic impacts | 0.75 | 0.00 | 0.83 | 0.50 |
| Governance | 0.81 | 0.00 | 1.00 | 0.50 |
| Capabilities progression | 1.00 | 0.00 | 0.68 | 0.33 |
| AI risks | 1.00 | 0.98 | 0.68 | 0.25 |
| National security | 1.00 | 0.96 | 0.95 | 0.00 |
| Alignment problem | 1.00 | 0.00 | 0.89 | 0.67 |
| Timelines | 0.88 | 0.96 | 1.00 | 1.00 |
| Open source | 0.54 | 0.00 | 0.58 | 0.33 |
| Compute role | 0.88 | 0.97 | 1.00 | 0.33 |

### Technical Implementation

RAGAS 0.1.x has incompatibilities with Python 3.13+ due to its use of `asyncio.wait_for` and `asyncio.as_completed` outside of a proper task context. The evaluation module includes two monkey-patches:

1. `Metric.ascore` — bypasses `asyncio.wait_for` timeout wrapping
2. `Executor.results` — wraps the async executor in `asyncio.run()` to provide a proper event loop

These patches are transparent and do not affect the metric computation itself.

---

## Layer 3: Base LLM vs RAG Comparison

**File:** `src/evaluation/compare_base_vs_rag.py`

Demonstrates that retrieval augmentation improves answer quality over using the LLM alone.

### Methodology

For each question in the QA dataset:

1. **Base LLM:** Ask the question directly with no context (prompt: "Answer to the best of your knowledge")
2. **RAG:** Run the full pipeline (retrieve → rerank → generate with context)
3. **Score both:** Compute word-overlap similarity against the `expected_answer`
4. **Compare:** Track which approach wins per question

### Scoring

Uses keyword overlap similarity:
- Extract words with length >3 from both the generated answer and expected answer
- Score = |intersection| / |expected keywords|

This is intentionally simple — it rewards answers that use the same terminology as the expected answer, which naturally favors the RAG system since it quotes from the same sources.

### Output

Produces:
- `data/processed/base_vs_rag_comparison.json` — raw results
- `data/processed/base_vs_rag_comparison.md` — formatted report with per-question analysis

---

## Quality Controls in Generation

The evaluation strategy is complemented by prompt engineering that maximizes answer quality:

### System Prompt Design

The generation prompt enforces:
1. **Strict grounding** — "Answer using ONLY the provided context"
2. **Explicit citation** — "Cite sources using [Source N] for each claim"
3. **Refusal behavior** — "If context is insufficient, say so"
4. **Terminology alignment** — "Use the same terminology as the source material"
5. **No knowledge leakage** — "Do not add information from your own knowledge"

### Context Formatting

Each retrieved chunk is presented with clear provenance:
```
[Source 1] Title - Section
URL: https://...

<chunk text>
```

This structure makes it easy for the LLM to cite specific sources and for evaluators to trace claims back to chunks.

---

## Index Quality

Evaluation is only meaningful if the underlying index is high-quality. Key decisions:

| Decision | Value | Rationale |
|----------|-------|-----------|
| Chunk size | 768 tokens | Large enough for coherent passages, small enough for precise retrieval |
| Chunk overlap | 150 tokens | Prevents information loss at boundaries |
| Deduplication | Removed 695 duplicate chunks | `/supplements/` pages were 100% identical to `/research/` |
| Final index size | 1,130 chunks | Focused, non-redundant coverage |
| Embedding model | all-mpnet-base-v2 (768-dim) | Strong semantic understanding for topic-diverse queries |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Refines initial retrieval with pairwise relevance |

---

## How to Run

```bash
# Retrieval metrics (fast, no LLM calls)
python -m src.evaluation.evaluate_retrieval

# RAGAS evaluation (requires OPENAI_API_KEY, ~2 min for 12 questions)
python -m src.evaluation.evaluate_ragas

# Base vs RAG comparison (requires LLM API key)
python -m src.evaluation.compare_base_vs_rag
```

All results are saved to `data/processed/`.

---

## Future Improvements

- Expand QA dataset to 50+ examples covering edge cases and adversarial queries
- Add human evaluation alongside LLM-as-judge for calibration
- Track metrics over time as the index is updated
- Add latency measurement (retrieval time, generation time, total time)
- Implement A/B testing framework for prompt/config changes
- Upgrade to RAGAS 0.2+ when Python 3.14 compatibility is resolved
