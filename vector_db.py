from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=1536):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim
        
        # Check if collection exists and has correct dimensions
        if self.client.collection_exists(self.collection):
            # Get collection info to check dimensions
            collection_info = self.client.get_collection(self.collection)
            existing_dim = collection_info.config.params.vectors.size
            
            if existing_dim != dim:
                print(f"Collection '{self.collection}' exists with dimension {existing_dim}, but expected {dim}")
                print(f"Recreating collection with correct dimensions...")
                self.client.delete_collection(self.collection)
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        ).points
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}