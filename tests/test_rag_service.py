"""
Tests for the RAG Service.

Tests the knowledge base loading, embedding, and retrieval
without requiring any external API calls.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.rag_service import RAGService


@pytest.fixture(scope="module")
def rag():
    """Create and initialize a RAG service for testing."""
    service = RAGService()
    service.initialize()
    return service


class TestRAGInitialization:
    """Test knowledge base loading and indexing."""

    def test_initialization(self, rag):
        """RAG service should initialize and populate the collection."""
        assert rag._collection is not None
        assert rag._collection.count() > 0
        assert rag._raw_data is not None

    def test_has_services(self, rag):
        """Should load salon services from the knowledge base."""
        services = rag.get_all_services()
        assert len(services) > 0
        assert any("Haircut" in s["name"] for s in services)

    def test_has_stylists(self, rag):
        """Should load stylists from the knowledge base."""
        stylists = rag.get_all_stylists()
        assert len(stylists) > 0
        assert all("name" in s for s in stylists)


class TestRAGRetrieval:
    """Test semantic search retrieval."""

    def test_haircut_query(self, rag):
        """Should retrieve relevant results for a haircut query."""
        results = rag.query("How much does a haircut cost?")
        assert len(results) > 0
        # Should find something about haircuts
        all_content = " ".join(r["content"].lower() for r in results)
        assert "haircut" in all_content or "cut" in all_content

    def test_color_query(self, rag):
        """Should retrieve color service information."""
        results = rag.query("Do you offer hair coloring?")
        assert len(results) > 0
        all_content = " ".join(r["content"].lower() for r in results)
        assert "color" in all_content or "highlight" in all_content or "balayage" in all_content

    def test_stylist_query(self, rag):
        """Should retrieve stylist information."""
        results = rag.query("Who specializes in balayage?")
        assert len(results) > 0

    def test_policy_query(self, rag):
        """Should retrieve cancellation policy."""
        results = rag.query("What is your cancellation policy?")
        assert len(results) > 0
        all_content = " ".join(r["content"].lower() for r in results)
        assert "cancel" in all_content or "policy" in all_content

    def test_hours_query(self, rag):
        """Should retrieve business hours."""
        results = rag.query("What are your business hours?")
        assert len(results) > 0

    def test_parking_query(self, rag):
        """Should retrieve parking information."""
        results = rag.query("Is there free parking?")
        assert len(results) > 0

    def test_results_have_scores(self, rag):
        """Results should include relevance scores."""
        results = rag.query("haircut price")
        assert len(results) > 0
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "relevance_score" in r

    def test_top_k_parameter(self, rag):
        """Should respect the top_k parameter."""
        results_1 = rag.query("salon services", top_k=1)
        results_3 = rag.query("salon services", top_k=3)
        assert len(results_1) == 1
        assert len(results_3) == 3


class TestRAGLookup:
    """Test direct data lookups."""

    def test_get_service_by_name(self, rag):
        """Should find a service by exact name."""
        svc = rag.get_service_by_name("Women's Haircut & Style")
        assert svc is not None
        assert svc["price"] == 85
        assert svc["duration_minutes"] == 60

    def test_get_service_not_found(self, rag):
        """Should return None for a non-existent service."""
        svc = rag.get_service_by_name("Invisible Ink Tattoo")
        assert svc is None

    def test_get_stylist_by_name(self, rag):
        """Should find a stylist by name."""
        stylist = rag.get_stylist_by_name("Sophia Martinez")
        assert stylist is not None
        assert "Balayage" in stylist["specialties"]

    def test_get_stylist_not_found(self, rag):
        """Should return None for a non-existent stylist."""
        stylist = rag.get_stylist_by_name("John Nobody")
        assert stylist is None
