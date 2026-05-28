"""Prompt templates for RAG generation."""

SYSTEM_PROMPT = """You are a domain-specific AI assistant that answers questions using ONLY the provided context passages. Follow these rules strictly:

1. Base your answer entirely on information found in the context passages below.
2. Quote or closely paraphrase the source text when making claims.
3. Cite sources using the format [Source N] for each claim.
4. If the context does not contain enough information to answer the question, say "The knowledge base does not contain enough information to answer this question."
5. Do not add information from your own knowledge.
6. Be specific and use the same terminology as the source material."""

USER_PROMPT_TEMPLATE = """Question:
{question}

Context:
{retrieved_context}

Instructions: Answer the question using ONLY the context above. For each key claim, cite the source number in brackets (e.g., [Source 1]). Use direct quotes or close paraphrases from the context.

Answer:"""

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

        header = f"[Source {i}] {title}"
        if section:
            header += f" - {section}"
        header += f"\nURL: {source}"

        context_parts.append(f"{header}\n\n{text}")

    return "\n\n---\n\n".join(context_parts)
