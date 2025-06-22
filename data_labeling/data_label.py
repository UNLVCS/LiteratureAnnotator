import openai
import json
import os
from dotenv import load_dotenv


def generate_embeddings(embed_text):
    load_dotenv()
    embeddings_obj =  openai.embeddings.create(
        model = "text-embedding-ada-002",
        input = embed_text,
        encoding_format = "float"
    )
    return embeddings_obj



def load_json_with_path(file_name):
    current_directory = os.getcwd()
    
    file_path = os.path.join(current_directory, file_name)
    
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    return json_data

## Takes in a dictionary named texts and creates a list of overlapping chunks
    ## Chunking strategy follows the logical separation of the text, i.e. Title, Abstract, ...Methods, ...
    ## However, some kind of overlap is achieved between this chunks
def overlap_window_chunker(text, overlap_size=50):

    chunks = []

    # i = 0
    for i in range(1, len(text)):

        text_first = "".join(text[i-1])
        text_later = "".join(text[i])
        
        chunk_first = ""
        chunk_latter = ""
        
        if len(text_first) < overlap_size:
            chunk_latter = text_first + ". " + text_later
        else:
            chunk_latter = text_first[overlap_size:] + ". " +  text_later


        if len(text_later) < overlap_size:
            chunk_first = text_first + ". " +  text_later
        else:
            chunk_first = text_first + ". " +  text_later[:overlap_size]

        chunks.append(chunk_first)
        chunks.append(chunk_latter)
        


        
    return chunks
        
from collections import defaultdict

raw_articles = load_json_with_path('processed_output.json')

articles = {}

for k, v in raw_articles.items():
    articles[k] = {}
    for section_type, section_value in v.items():
        articles[k][section_type] = section_value


article_instance = next(iter(articles))

# chunk = 
article_as_list = list(articles[article_instance].values())
chunks = overlap_window_chunker(article_as_list)



from vector_db import vector_db

db = vector_db()
# records = []
for i, chunk in enumerate(chunks):

    response = generate_embeddings(chunk)
    response = json.loads(response.model_dump_json())
    embedding = response['data'][0]['embedding']

    record = {
        "id": f"doc1-chunk{i}", 
        "values" : embedding,
        "metadata" : {
            "doc" : "doc1",
            "chunk" : i
        }
    }

    db.upsert("123456", [record])

# db.upsert('test_namespace', )



response = generate_embeddings("What is alzheimer's disease?")
response = json.loads(response.model_dump_json())
embedding = response['data'][0]['embedding']



from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub



from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("rlm/rag-prompt")

from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=db.__get_index__(), embedding=embeddings)


qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )


question = "How is ai being used in Alzheimer's disease, cite me a specifc title of a paper exploring this"
result = qa_chain({"query": question })
print(result)