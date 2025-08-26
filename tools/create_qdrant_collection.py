import os
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv


load_dotenv()
pyansys_module = os.getenv("PYANSYS_MODULE")

# Qdrant Client
qdrant_client = QdrantClient(path="../langchain_qdrant")

# Create collection
qdrant_client.create_collection(
    collection_name=f"{pyansys_module}",
    vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE),
)