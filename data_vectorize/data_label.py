"""CLI entry point: ingest all articles from MinIO into the vector DB."""

from data_vectorize.ingester import Ingester

if __name__ == "__main__":
    Ingester().run()
