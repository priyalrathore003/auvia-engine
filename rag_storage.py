"""
rag_storage.py — ChromaDB RAG layer for audio issue transcripts.

Lazy-init for Cloud Run: embeddings load on first query, not at import.
Persist dir defaults to /tmp/chroma_db (Cloud Run ephemeral disk).
"""

import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")
COLLECTION = "audio_transcripts"

MOCK_DOCUMENTS = [
    Document(
        page_content=(
            "File_A has a severe low-frequency hum at 60Hz, likely caused by "
            "electrical interference from the recording environment."
        ),
        metadata={"source": "File_A", "issue_type": "low_frequency_noise"},
    ),
    Document(
        page_content=(
            "Vocal_B contains harsh sibilance in the upper frequencies (6kHz-10kHz), "
            "making 's' and 'sh' sounds overly sharp and fatiguing."
        ),
        metadata={"source": "Vocal_B", "issue_type": "sibilance"},
    ),
    Document(
        page_content=(
            "Track_C exhibits clipping distortion during the chorus sections, "
            "with peak levels exceeding 0dBFS by approximately 3dB."
        ),
        metadata={"source": "Track_C", "issue_type": "clipping_distortion"},
    ),
]

_vector_store: Chroma | None = None


def _get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    os.makedirs(PERSIST_DIR, exist_ok=True)
    logger.info("Initializing ChromaDB at %s", PERSIST_DIR)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    if store._collection.count() == 0:
        logger.info("Seeding RAG with %d mock documents", len(MOCK_DOCUMENTS))
        store.add_documents(MOCK_DOCUMENTS)

    _vector_store = store
    return _vector_store


def search_transcripts(query: str, k: int = 2) -> str:
    """Plain function for internal use (LangGraph nodes)."""
    results = _get_vector_store().similarity_search(query, k=k)
    if not results:
        return "No matching transcripts found."
    return "\n\n".join(doc.page_content for doc in results)


@tool
def query_audio_transcripts(query: str) -> str:
    """Query the audio transcripts vector store for relevant audio issue information."""
    return search_transcripts(query)
