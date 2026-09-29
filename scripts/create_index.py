"""Create the Pinecone serverless index if it doesn't exist, then print its size.

python scripts/create_index.py
"""

from __future__ import annotations

from app.bootstrap import open_vector_store
from app.core.config import Settings
from scripts._cli import run


def main(settings: Settings) -> int:
    with open_vector_store(settings) as store:
        store.ensure_index()
        print(
            f"Index '{settings.pinecone_index_name}' is ready "
            f"(dimension {settings.embedding_dimension}, metric cosine). "
            f"Namespace '{settings.pinecone_namespace}' holds {store.count()} vectors."
        )
    return 0


if __name__ == "__main__":
    run(main)
