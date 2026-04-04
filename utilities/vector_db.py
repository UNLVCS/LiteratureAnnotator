from typing import Optional, TYPE_CHECKING

from pinecone import Pinecone, ServerlessSpec, PineconeApiException

if TYPE_CHECKING:
    from config.app_config import PineconeConfig


class VectorDb:
    """Pinecone vector database wrapper."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        pinecone_config: Optional["PineconeConfig"] = None,
    ):
        """
        Initialize VectorDb.
        
        Args:
            api_key: Pinecone API key. If not provided, loads from AppConfig.
            index_name: Index name. If not provided, uses config default.
            pinecone_config: Optional PineconeConfig object.
        """
        app_config = None
        if pinecone_config is None and api_key is None:
            from config.app_config import load_app_config
            app_config = load_app_config()
            pinecone_config = app_config.pinecone
            embedding_dimensions = embedding_dimensions or app_config.embeddings.dimensions
        elif embedding_dimensions is None:
            from config.app_config import load_app_config
            app_config = load_app_config()
            embedding_dimensions = app_config.embeddings.dimensions
        
        if pinecone_config is not None:
            api_key = pinecone_config.api_key
            index_name = index_name or pinecone_config.index_name
        
        if index_name is None:
            index_name = "adbm"
        if embedding_dimensions is None:
            embedding_dimensions = 1536
        
        self.pc = Pinecone(api_key=api_key)

        try:
            self.pc.create_index(
                name=index_name,
                dimension=embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except PineconeApiException as e:
            if e.status == 409:
                print("Index already exists")
            else:
                raise

        self.host = self.pc.describe_index(name=index_name).host
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