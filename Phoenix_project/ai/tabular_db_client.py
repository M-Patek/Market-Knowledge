# ai/tabular_db_client.py
"""
Manages the connection and lifecycle of the tabular database (PostgreSQL)
for storing and retrieving structured financial data.
"""
import os
import logging
import psycopg2
from psycopg2.extras import DictCursor, Json
from typing import List, Dict, Any, Optional

class TabularDBClient:
    """
    A client to manage interactions with the PostgreSQL tabular database.
    """
    def __init__(self):
        """
        Initializes the connection to PostgreSQL and ensures the schema exists.
        """
        self.logger = logging.getLogger("PhoenixProject.TabularDBClient")
        self.conn = None
        db_url = os.getenv("POSTGRES_DB_URL")
        if not db_url:
            self.logger.error("POSTGRES_DB_URL environment variable not set. TabularDBClient will be non-operational.")
            return

        try:
            self.conn = psycopg2.connect(db_url)
            self.logger.info("Successfully connected to PostgreSQL database.")
            self.setup_schema()
        except psycopg2.OperationalError as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.conn = None

    def setup_schema(self):
        """
        Creates the 'financial_data' table if it doesn't already exist.
        """
        if not self.conn: return
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS financial_data (
            id SERIAL PRIMARY KEY,
            source_id VARCHAR(255) NOT NULL,
            ticker VARCHAR(20) NOT NULL,
            report_date DATE NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value NUMERIC(20, 4),
            metadata_jsonb JSONB,
            observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(ticker, report_date, metric_name)
        );
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
            self.conn.commit()
            self.logger.info("Table 'financial_data' schema is up to date.")
        except Exception as e:
            self.logger.error(f"Failed to create or verify table schema: {e}")
            self.conn.rollback()

    def insert_table_data(self, data: List[Dict[str, Any]]):
        """
        Inserts a list of structured data records into the database.

        Args:
            data: A list of dictionaries, each representing a row to insert.
        """
        if not self.conn:
            self.logger.error("No database connection. Cannot insert data.")
            return

        insert_query = """
        INSERT INTO financial_data (source_id, ticker, report_date, metric_name, metric_value, metadata_jsonb)
        VALUES (%(source_id)s, %(ticker)s, %(report_date)s, %(metric_name)s, %(metric_value)s, %(metadata_jsonb)s)
        ON CONFLICT (ticker, report_date, metric_name) DO UPDATE SET
            metric_value = EXCLUDED.metric_value,
            metadata_jsonb = EXCLUDED.metadata_jsonb,
            observed_at = NOW();
        """
        try:
            with self.conn.cursor() as cur:
                # The execute_batch helper is efficient for bulk inserts
                psycopg2.extras.execute_batch(cur, insert_query, [
                    {**d, 'metadata_jsonb': Json(d.get('metadata_jsonb', {}))} for d in data
                ])
            self.conn.commit()
            self.logger.info(f"Successfully inserted or updated {len(data)} records.")
        except Exception as e:
            self.logger.error(f"Failed to insert data: {e}")
            self.conn.rollback()

    def query_by_metric(self, ticker: str, metric_name: str, limit: int = 5) -> Optional[List[Dict]]:
        """
        Queries for the most recent values of a specific metric for a ticker.
        """
        if not self.conn: return None
        
        query = """
        SELECT report_date, metric_value, metadata_jsonb
        FROM financial_data
        WHERE ticker = %s AND metric_name = %s
        ORDER BY report_date DESC
        LIMIT %s;
        """
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, (ticker, metric_name, limit))
                results = [dict(row) for row in cur.fetchall()]
            return results
        except Exception as e:
            self.logger.error(f"Failed to query data for {ticker}/{metric_name}: {e}")
            return None

    def is_healthy(self) -> bool:
        """Checks if the database connection is active."""
        return self.conn is not None and not self.conn.closed

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("PostgreSQL connection closed.")
