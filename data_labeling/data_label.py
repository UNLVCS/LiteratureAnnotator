from chnker import Chunker
from vector_db import VectorDb
import os
import json
import openai
from dotenv import load_dotenv
from langchain_core.documents import Document


def generate_embeddings(embed_text):
    load_dotenv()
    embeddings_obj =  openai.embeddings.create(
        model = "text-embedding-ada-002",
        input = embed_text,
        encoding_format = "float"
    )
    return embeddings_obj


def load_articles(file_name):
    current_directory = os.getcwd()
    # current_directory = "/home/tadesa1/research/ADBM/data_labeling"
    
    file_path = os.path.join(current_directory, file_name)
    
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    return json_data



if __name__ == "__main__":

    raw_articles = load_articles('processed_output.json')
    vdb = VectorDb()

    i = 10
    for k, v in raw_articles.items():
        
        chnkr = Chunker()

        documents = []

        chnkr.set_chunk(article_dict={k: v})
        chnkd_article = chnkr.get_chunked_article()

        print(f"Doing article: {k}")
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
                    "title" : v['Title'],
                    "chunk" : i
                }
            }

            vdb.upsert("article_upload_test_2", [record])
        
        if i <= 0:
            break

    print("Done")