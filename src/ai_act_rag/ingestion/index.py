import time
import uuid

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import PayloadSchemaType

from ai_act_rag.config import COLLECTION_NAME, qdrant_client
from ai_act_rag.ingestion.chunk import Chunk, chunk_units
from ai_act_rag.ingestion.parse import PDF_PATH, extract_lines, split_units
from ai_act_rag.models import configure_settings

# Metadata the embedding sees: the citation and title help match questions like
# "what are the obligations of deployers?". Page numbers etc. would only add noise.
EMBED_METADATA = {"citation", "title"}
# Metadata the LLM sees next to each chunk, so it can cite unit and page.
LLM_METADATA = {"citation", "title", "pages"}
# Fields we may filter searches on; Qdrant needs an index on them.
FILTER_FIELDS = ("kind", "number", "chapter")


def chunk_id(chunk: Chunk) -> str:
    """A stable ID from the citation, so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ai-act/{chunk.citation}"))


def chunk_to_node(chunk: Chunk) -> TextNode:
    """Turn a chunk into a LlamaIndex TextNode with its legal metadata."""
    unit = chunk.unit
    pages = (
        str(chunk.first_page)
        if chunk.first_page == chunk.last_page
        else f"{chunk.first_page}-{chunk.last_page}"
    )
    metadata = {
        "citation": chunk.citation,
        "kind": unit.kind,
        "number": unit.number,
        "paragraph": chunk.paragraph,
        "point": chunk.point,
        "title": unit.title,
        "chapter": unit.chapter,
        "section": unit.section,
        "pages": pages,
        "first_page": chunk.first_page,
        "last_page": chunk.last_page,
    }
    return TextNode(
        id_=chunk_id(chunk),
        text=chunk.text,
        metadata=metadata,
        excluded_embed_metadata_keys=[k for k in metadata if k not in EMBED_METADATA],
        excluded_llm_metadata_keys=[k for k in metadata if k not in LLM_METADATA],
    )


def build_nodes() -> list[TextNode]:
    """Parse the PDF and return one node per chunk."""
    chunks = chunk_units(split_units(extract_lines(PDF_PATH)))
    return [chunk_to_node(chunk) for chunk in chunks]


def main() -> None:
    configure_settings()
    nodes = build_nodes()
    print(f"Nodes: {len(nodes)}")
    print("--- What the embedding model sees for the first article chunk ---")
    first_article = next(n for n in nodes if n.metadata["kind"] == "article")
    print(first_article.get_content(metadata_mode="embed")[:300])

    client = qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)  # rebuild from scratch each run
        print(f"Deleted old collection '{COLLECTION_NAME}'")

    vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        client=client,
        batch_size=32,  # points per upload request; smaller requests time out less
    )
    storage = StorageContext.from_defaults(vector_store=vector_store)

    start = time.time()
    VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    print(f"Embedded and uploaded in {time.time() - start:.0f} s")

    for field in FILTER_FIELDS:
        client.create_payload_index(COLLECTION_NAME, field, PayloadSchemaType.KEYWORD)

    count = client.count(COLLECTION_NAME).count
    print(f"Collection '{COLLECTION_NAME}' now holds {count} points")


if __name__ == "__main__":
    main()
