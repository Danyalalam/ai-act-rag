from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3.5:9b"

# Qwen3-embedding works best when queries (not documents) start with a task
# instruction in this format: "Instruct: <task>\nQuery: <question>".
QUERY_INSTRUCTION = (
    "Instruct: Given a question about the EU AI Act, "
    "retrieve the passages of the regulation that answer it\nQuery:"
)


def embed_model() -> OllamaEmbedding:
    """The local Ollama embedding model, used for both chunks and questions."""
    return OllamaEmbedding(
        model_name=EMBED_MODEL,
        query_instruction=QUERY_INSTRUCTION,
        embed_batch_size=32,  # texts sent to Ollama per request
        # Context in tokens: the longest chunk is ~1,000 tokens. Ollama's default (4,096)
        # reserves ~2.4 GB of RAM; 2,048 roughly halves that.
        ollama_additional_kwargs={"num_ctx": 2048},
        keep_alive="2m",  # unload from RAM after 2 idle minutes (Ollama's default is 5)
    )


def llm() -> Ollama:
    """The local Ollama LLM that writes the answers."""
    return Ollama(
        model=LLM_MODEL,
        # Tokens: question + 6 chunks + answer is ~2,000-3,000. 8,192 made the model take
        # 9.9 GB and pushed a 16 GB Mac into swap; 4,096 is enough and much lighter.
        context_window=4096,
        temperature=0.1,  # low = stick closely to the retrieved text
        thinking=False,  # skip Qwen's long "thinking" phase; much faster on a laptop
        request_timeout=300,  # seconds; the first call also loads the model into memory
        keep_alive="2m",  # unload from RAM after 2 idle minutes
    )


def configure_settings() -> None:
    """Set LlamaIndex's global models so nothing silently falls back to OpenAI."""
    Settings.embed_model = embed_model()
    Settings.llm = llm()  # created lazily: the model only loads on the first LLM call
