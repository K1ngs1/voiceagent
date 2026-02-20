"""
RAG (Retrieval-Augmented Generation) Service.

Loads the salon knowledge base, embeds documents into ChromaDB,
and provides semantic search for customer queries.
"""

import json
import os
import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

# Path to the knowledge base JSON
KB_PATH = Path(__file__).parent.parent.parent / "knowledge_base" / "salon_data.json"


class RAGService:
    """Retrieval-Augmented Generation service using ChromaDB."""

    def __init__(self):
        self._client = None
        self._embedding_fn = None
        self._collection = None
        self._raw_data = None

    def initialize(self):
        """Load knowledge base and build the vector index."""
        logger.info("Initializing RAG service...")

        # Lazy-initialize client and embeddings
        if self._client is None:
            self._client = chromadb.Client()
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

        # Load raw salon data
        with open(KB_PATH, "r") as f:
            self._raw_data = json.load(f)

        # Create or get the collection
        self._collection = self._client.get_or_create_collection(
            name="salon_knowledge",
            embedding_function=self._embedding_fn,
        )

        # If already populated, skip
        if self._collection.count() > 0:
            logger.info(f"RAG collection already has {self._collection.count()} documents.")
            return

        documents = []
        metadatas = []
        ids = []

        # ── Services ───────────────────────────────────
        for i, svc in enumerate(self._raw_data.get("services", [])):
            doc = (
                f"Service: {svc['name']}\n"
                f"Category: {svc['category']}\n"
                f"Description: {svc['description']}\n"
                f"Duration: {svc['duration_minutes']} minutes\n"
                f"Price: ${svc['price']}"
            )
            documents.append(doc)
            metadatas.append({"source": "services", "service_name": svc["name"]})
            ids.append(f"service_{i}")

        # ── Stylists ──────────────────────────────────
        for i, stylist in enumerate(self._raw_data.get("stylists", [])):
            doc = (
                f"Stylist: {stylist['name']}\n"
                f"Title: {stylist['title']}\n"
                f"Specialties: {', '.join(stylist['specialties'])}\n"
                f"Bio: {stylist['bio']}\n"
                f"Available: {', '.join(stylist['availability'])}"
            )
            documents.append(doc)
            metadatas.append({"source": "stylists", "stylist_name": stylist["name"]})
            ids.append(f"stylist_{i}")

        # ── Policies ──────────────────────────────────
        policies = self._raw_data.get("policies", {})
        for i, (key, value) in enumerate(policies.items()):
            doc = f"Policy – {key.replace('_', ' ').title()}: {value}"
            documents.append(doc)
            metadatas.append({"source": "policies", "policy_type": key})
            ids.append(f"policy_{i}")

        # ── FAQs ──────────────────────────────────────
        for i, faq in enumerate(self._raw_data.get("faqs", [])):
            doc = f"Q: {faq['question']}\nA: {faq['answer']}"
            documents.append(doc)
            metadatas.append({"source": "faqs"})
            ids.append(f"faq_{i}")

        # ── Locations ─────────────────────────────────
        for i, loc in enumerate(self._raw_data.get("locations", [])):
            hours_str = "\n".join(
                [f"  {day}: {hrs}" for day, hrs in loc.get("hours", {}).items()]
            )
            doc = (
                f"Location: {loc['name']}\n"
                f"Address: {loc['address']}\n"
                f"Phone: {loc['phone']}\n"
                f"Hours:\n{hours_str}\n"
                f"Parking: {loc.get('parking', 'N/A')}"
            )
            documents.append(doc)
            metadatas.append({"source": "locations", "location_name": loc["name"]})
            ids.append(f"location_{i}")

        # ── Salon Info ────────────────────────────────
        salon = self._raw_data.get("salon", {})
        doc = (
            f"Salon: {salon.get('name', '')}\n"
            f"Tagline: {salon.get('tagline', '')}\n"
            f"Phone: {salon.get('phone', '')}\n"
            f"Email: {salon.get('email', '')}\n"
            f"Website: {salon.get('website', '')}"
        )
        documents.append(doc)
        metadatas.append({"source": "salon_info"})
        ids.append("salon_info")

        # Add all documents to the collection
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"RAG service initialized with {len(documents)} documents.")

    def query(self, question: str, top_k: int = 3) -> list[dict]:
        """
        Search the knowledge base for relevant information.

        Args:
            question: The user's question or query.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'content', 'source', and 'relevance_score'.
        """
        if self._collection is None or self._collection.count() == 0:
            logger.warning("RAG collection is empty, initializing...")
            self.initialize()

        results = self._collection.query(
            query_texts=[question],
            n_results=min(top_k, self._collection.count()),
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append(
                {
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "relevance_score": round(
                        1 - results["distances"][0][i], 4  # Convert distance to similarity
                    ),
                }
            )
        return output

    def get_service_by_name(self, service_name: str) -> dict | None:
        """Look up a specific service by name from raw data."""
        if self._raw_data is None:
            self.initialize()
        for svc in self._raw_data.get("services", []):
            if svc["name"].lower() == service_name.lower():
                return svc
        return None

    def get_stylist_by_name(self, stylist_name: str) -> dict | None:
        """Look up a specific stylist by name from raw data."""
        if self._raw_data is None:
            self.initialize()
        for stylist in self._raw_data.get("stylists", []):
            if stylist["name"].lower() == stylist_name.lower():
                return stylist
        return None

    def get_all_stylists(self) -> list[dict]:
        """Get all stylists."""
        if self._raw_data is None:
            self.initialize()
        return self._raw_data.get("stylists", [])

    def get_all_services(self) -> list[dict]:
        """Get all services."""
        if self._raw_data is None:
            self.initialize()
        return self._raw_data.get("services", [])


# Singleton instance
rag_service = RAGService()
