from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from dotenv import load_dotenv, find_dotenv
import os

class VectorDb():
    def __init__(self, index_name: str = "adbm"):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        try:
            self.pc.create_index(
                name=index_name,
                dimension=1536 or os.getenv("EMBEDDER_DIM"),
                metric='cosine' or os.getenv('EMBEDDER_METRIC'),
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        except PineconeApiException as e:
            if e.status == 409:
                print("Index already exists")

        self.host = self.pc.describe_index(name = index_name).host
        self.index = self.pc.Index(index_name)


    def upsert(self, user_id: str,data: list):
        """
            During an upsertion a namespace is created automatically if it does not exist.
                This namespace is essentially a logical separation between the different users we have. 
                This allows for separation of data within the vector db 
        """
        self.index.upsert(
            namespace = user_id,        # This is the namespace
            vectors = data            # This is the data to be upserted
        )

    def __get_index__(self):
        return self.index