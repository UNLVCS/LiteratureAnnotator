from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from vector_db import VectorDb
from dotenv import load_dotenv
import openai

db = VectorDb()


def generate_embeddings(embed_text):
    load_dotenv()
    embeddings_obj =  openai.embeddings.create(
        model = "text-embedding-ada-002",
        input = embed_text,
        encoding_format = "float"
    )
    return embeddings_obj

vector_store = PineconeVectorStore(index=db.__get_index__(), embedding=embeddings, namespace="article_upload_test_1")


