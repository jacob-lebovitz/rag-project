"""Mistral-backed generation + embeddings."""
from .llm import MistralClient, get_client

__all__ = ["MistralClient", "get_client"]
