
"""Test configured chat and embedding model initialization."""

from core.config import get_settings
from core.model_factory import (
    create_chat_model,
    create_embedding_model,
)

settings = get_settings()

# Initialize the selected chat provider.
chat_model = create_chat_model(settings=settings)
print(f"Chat model initialized: {type(chat_model).__name__}")

# Initialize the selected embedding provider.
embedding_model = create_embedding_model(settings=settings)
print(
    f"Embedding model initialized: "
    f"{type(embedding_model).__name__}"
)

# Test an actual chat request.
response = chat_model.invoke(
    "Explain what a Python function is in one sentence."
)
print("\nChat response:")
print(response.content)

# Test actual embedding generation.
vector = embedding_model.embed_query(
    "Python function for password verification"
)
print(f"\nEmbedding dimensions: {len(vector)}")