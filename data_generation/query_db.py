from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config.app_config import load_app_config
from data_vectorize import VectorDb

app_config = load_app_config()

db = VectorDb()

embeddings = OpenAIEmbeddings(
    model=app_config.embeddings.model,
    api_key=app_config.embeddings.api_key,
    dimensions=None if app_config.embeddings.model == "text-embedding-ada-002" else app_config.embeddings.dimensions,
)

# Resolve namespace using the same logic as Ingester:
# bioc_download.object_prefix takes precedence over pinecone.namespace.
_prefix = app_config.bioc_download.object_prefix or ""
_namespace = _prefix or app_config.pinecone.namespace

vector_store = PineconeVectorStore(
    index=db.index,
    embedding=embeddings,
    namespace=_namespace,
)
