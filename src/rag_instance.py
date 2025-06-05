# src/rag_instance.py
import logging
from src.rag_system import MTGRetrievalSystem

# Configure logging
logger = logging.getLogger("RAG_Instance")

# Initialize the RAG system
try:
    rag_system = MTGRetrievalSystem(embedding_model_name="all-MiniLM-L6-v2")
    logger.info("RAG system initialized with embedding model")
except Exception as e:
    logger.error(f"Error initializing RAG system: {e}")
    rag_system = MTGRetrievalSystem()
    logger.info("RAG system initialized without embedding model")