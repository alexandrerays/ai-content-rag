"""Prompt templates for RAG generation."""

SYSTEM_PROMPT = """You are a careful domain-specific AI assistant. Answer the user question using only the provided context. If the context is insufficient, say that the knowledge base does not contain enough information. Always cite the source title and URL for each important claim."""

USER_PROMPT_TEMPLATE = """Question:
{question}

Context:
{retrieved_context}

Answer format:
- Direct answer
- Key evidence
- Sources"""

BASE_LLM_PROMPT = """Answer the following question to the best of your knowledge. If you are not sure, say so.

Question: {question}"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_url", "unknown")
        title = chunk.get("title", "untitled")
        section = chunk.get("section", "")
        text = chunk.get("text", "")
        score = chunk.get("score", 0.0)

        header = f"[Source {i}] {title}"
        if section:
            header += f" - {section}"
        header += f"\nURL: {source}\nRelevance: {score:.3f}"

        context_parts.append(f"{header}\n\n{text}")

    return "\n\n---\n\n".join(context_parts)
