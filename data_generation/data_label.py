import sys
from pathlib import Path

# Add parent directory to path to import utilities module
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from chnker import Chunker
from utilities.vector_db import VectorDb
import json
import openai
from dotenv import load_dotenv
from minio import Minio

client = Minio(
    "localhost:5000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
bucket_name = "raw-pubmed-articles"

def generate_embeddings(embed_text):
    load_dotenv()
    embeddings_obj =  openai.embeddings.create(
        model = "text-embedding-ada-002",
        input = embed_text,
        encoding_format = "float"
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
            article_data = json.loads(response.read().decode('utf-8'))
            article_id = obj.object_name.split('/')[-1].split('.')[0]
            all_articles.append({article_id: article_data})
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

            vdb.upsert("V3_raw_pubmed_articles", [record])       
            
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