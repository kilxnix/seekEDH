import logging
from src.api_server import app
import src.update_api_server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGServer")

def main():
    """Run the API server with RAG capabilities"""
    logger.info("Starting API server with RAG capabilities")
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()