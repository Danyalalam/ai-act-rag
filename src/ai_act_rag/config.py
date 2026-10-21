import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()  # read .env from the project root into environment variables

COLLECTION_NAME = "ai_act"  # the Qdrant collection holding the chunks

# Where Ollama runs. Natively on the Mac it's localhost; from inside a Docker container
# it's http://host.docker.internal:11434 (Mac) or http://ollama:11434 (Ollama container).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def required_env(name: str) -> str:
    """Return an environment variable, or stop with a clear message if it's missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def qdrant_client() -> QdrantClient:
    """Connect to the Qdrant Cloud cluster configured in .env."""
    return QdrantClient(
        url=required_env("QDRANT_URL"),
        api_key=required_env("QDRANT_API_KEY"),
        timeout=60,  # seconds; the default (5 s) is too short for uploading batches
    )


def main() -> None:
    client = qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    print(f"Connected to Qdrant. Collections: {collections}")


if __name__ == "__main__":
    main()
