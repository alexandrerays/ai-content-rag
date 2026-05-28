"""Build the vector index from raw documents."""

import logging

from src.indexing.embeddings import generate_embeddings
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_raw_documents

logger = logging.getLogger(__name__)


def build_index() -> VectorStore:
    """Full pipeline: load -> clean -> chunk -> embed -> index."""
    logger.info("Starting index build pipeline")

    documents = load_raw_documents()
    if not documents:
        raise RuntimeError("No documents found. Run the scraper first.")

    cleaned_docs = clean_documents(documents)
    chunks = chunk_documents(cleaned_docs)

    texts = [chunk.text for chunk in chunks]
    embeddings = generate_embeddings(texts)

    store = VectorStore(dimension=embeddings.shape[1])
    store.build_index(embeddings, chunks)
    store.save()

    logger.info("Index build complete")
    return store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    build_index()
