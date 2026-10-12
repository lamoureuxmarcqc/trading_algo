"""Fixtures pytest pour services/knowledge.

Force le mode stub pour les embeddings -> pas de telechargement
du modele HuggingFace pendant les tests.
"""
import os

# Force le mode stub AVANT tout import de app.embeddings
os.environ["KNOW_EMBEDDING_MODE"] = "stub"