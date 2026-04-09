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

vector_store = PineconeVectorStore(
    index=db.index,
    embedding=embeddings,
    namespace=app_config.pinecone.namespace,
)
