# ai/temporal_db_client.py
"""
Manages the connection and lifecycle of the temporal index (Elasticsearch)
for storing and retrieving time-stamped event data.
"""
import os
import logging
from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
from datetime import date

class TemporalDBClient:
    """
    A client to manage interactions with the Elasticsearch temporal index.
    """
    def __init__(self, index_name: str = "phoenix-temporal-index"):
        """
        Initializes the connection to Elasticsearch and ensures the index exists.
        """
        self.logger = logging.getLogger("PhoenixProject.TemporalDBClient")
        self.index_name = index_name
        self.es: Optional[Elasticsearch] = None
        es_url = os.getenv("ELASTICSEARCH_URL")

        if not es_url:
            self.logger.error("ELASTICSEARCH_URL environment variable not set. TemporalDBClient will be non-operational.")
            return
        
        try:
            self.es = Elasticsearch(es_url)
            if not self.es.ping():
                raise ConnectionError("Elasticsearch ping failed.")
            self.logger.info("Successfully connected to Elasticsearch.")
            self._setup_index()
        except Exception as e:
            self.logger.error(f"Failed to connect or setup Elasticsearch: {e}")
            self.es = None

    def _setup_index(self):
        """Creates the index with the correct mapping if it doesn't exist."""
        if not self.es or self.es.indices.exists(index=self.index_name):
            return

        mapping = {
            "properties": {
                "timestamp": {"type": "date"},
                "entities": {"type": "keyword"},
                "keywords": {"type": "text"},
                "source_id": {"type": "keyword"}
            }
        }
        try:
            self.es.indices.create(index=self.index_name, mappings=mapping)
            self.logger.info(f"Successfully created Elasticsearch index '{self.index_name}' with correct mapping.")
        except Exception as e:
            self.logger.error(f"Failed to create Elasticsearch index: {e}")

    def insert_events(self, events: List[Dict[str, Any]]):
        """
        Bulk inserts a list of event documents into the index.

        Args:
            events: A list of event dictionaries.
        """
        if not self.es:
            self.logger.error("No Elasticsearch connection. Cannot insert events.")
            return
        
        actions = [
            {"_index": self.index_name, "_id": event["source_id"], "_source": event}
            for event in events
        ]
        try:
            from elasticsearch.helpers import bulk
            bulk(self.es, actions)
            self.logger.info(f"Successfully indexed {len(events)} events.")
        except Exception as e:
            self.logger.error(f"Failed to bulk insert events: {e}")

    def query_by_time_and_entities(self, start_date: date, end_date: date, entities: List[str], size: int = 25) -> List[Dict]:
        """
        Queries for documents containing specific entities within a date range.
        """
        if not self.es: return []
        
        query = {
            "bool": {
                "filter": [
                    {"range": {"timestamp": {"gte": start_date, "lte": end_date}}},
                    {"terms": {"entities": entities}}
                ]
            }
        }
        try:
            response = self.es.search(index=self.index_name, query=query, size=size)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            self.logger.error(f"Failed to query Elasticsearch: {e}")
            return []

    def is_healthy(self) -> bool:
        """Checks if the Elasticsearch connection is active."""
        return self.es is not None and self.es.ping()
