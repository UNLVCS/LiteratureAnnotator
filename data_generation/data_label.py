import sys
from pathlib import Path

# Add parent directory to path to import utilities module
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from chnker import Chunker
from config.app_config import load_app_config
from utilities.vector_db import VectorDb
import json
import openai
from dotenv import load_dotenv
from minio import Minio

app_config = load_app_config()
openai_client = openai.OpenAI(api_key=app_config.embeddings.api_key)

client = Minio(
    app_config.minio.url,
    access_key=app_config.minio.access_key,
    secret_key=app_config.minio.secret_key,
    secure=app_config.minio.secure,
)
bucket_name = app_config.minio.raw_articles_bucket
pinecone_namespace = app_config.pinecone.namespace


def _object_stem(object_name: str) -> str:
    """Last path segment without extension, e.g. papers/99999.json -> 99999."""
    return Path(object_name).stem


def normalize_raw_article_json(data: dict, object_name: str) -> dict:
    """
    Chunker expects {"PMID": {"Title": ..., "Abstract": ..., ...}}.

    Also accepts a flat article body {"Title": ..., ...} and wraps it using the
    object filename stem as PMID (fixes TypeError when the first key is "Title").
    """
    if not isinstance(data, dict):
        raise TypeError(f"{object_name}: root JSON must be an object")

    if len(data) == 1:
        sole_key, sole_val = next(iter(data.items()))
        if isinstance(sole_val, dict) and "Title" in sole_val:
            return data

    if "Title" in data and isinstance(data["Title"], (str, list)):
        pmid = _object_stem(object_name)
        return {pmid: data}

    raise ValueError(
        f"{object_name}: expected {{'PMID': {{'Title': ...}}}} or a flat object "
        f"with 'Title'; got top-level keys: {list(data.keys())}"
    )


def generate_embeddings(embed_text):
    load_dotenv()
    embeddings_obj = openai_client.embeddings.create(
        model=app_config.embeddings.model,
        input=embed_text,
        encoding_format="float",
        dimensions=app_config.embeddings.dimensions,
    )
    return embeddings_obj


def load_articles():
    """
    Load all JSON articles from the MinIO bucket.
    Returns a dictionary containing all articles from all files in the bucket.
    """
    all_articles = []
    
    # List all objects in the bucket
    objects = client.list_objects(bucket_name)
    
    for obj in objects:
        try:
            # Get the object from MinIO
            response = client.get_object(bucket_name=bucket_name, object_name=obj.object_name)
            
            # Read and decode the response data
            article_data = json.loads(response.read().decode("utf-8"))
            all_articles.append(
                normalize_raw_article_json(article_data, obj.object_name)
            )
            print(f"Loaded article: {obj.object_name}")
        except Exception as e:
            print(f"Error loading {obj.object_name}: {e}")
            continue
            
    return all_articles



if __name__ == "__main__":

    raw_articles = load_articles()
    vdb = VectorDb()

    for article in raw_articles:
        # i = 10
        
        chnkr = Chunker()
        # chnkr.set_chunk(article_dict={article.keys(): article.values()})
        k, v = next(iter(article.items()))
        chnkr.set_chunk({k: v})

        chnkd_article = chnkr.get_chunked_article()
        
        for i, chunk in enumerate(chnkd_article['chunks']):

            response = generate_embeddings(chunk)
            response = json.loads(response.model_dump_json())
            embedding = response['data'][0]['embedding']
                
            record = {
                "id" : f"{chnkd_article['id']}-chunk{i}",
                "values" : embedding,
                "metadata" : {
                    "text" : chunk,
                    "doc" : chnkd_article['id'],
                    "title" : chnkd_article['title'],
                    "chunk" : i
                 }
            }

            vdb.upsert(pinecone_namespace, [record])
            
            # chnkr = Chunker()

            # documents = []
            # if k == 'Title':
            #     continue
            # chnkr.set_chunk(article_dict={k: v})
            # chnkd_article = chnkr.get_chunked_article()

            # print(f"Doing article: {k}")

            
            # if i <= 0:
            #     break

    print("Done")