# ai/tabular_db_client.py
from sqlalchemy import create_engine
from alembic.config import Config
from alembic import command
import os

class TabularDBClient:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.connection = None
        # Other initialization...

    def connect(self):
        self._run_migrations()
        self.connection = self.engine.connect()
        print("Successfully connected to the database.")
        return self.connection

    def _run_migrations(self):
        """
        Ensures the database is upgraded to the latest version using Alembic.
        """
        print("Running database migrations...")
        try:
            # Assumes alembic.ini is in the root project directory
            alembic_cfg = Config("alembic.ini")
            # Tell Alembic to use our app's database connection
            alembic_cfg.set_main_option('sqlalchemy.url', self.db_uri)
            command.upgrade(alembic_cfg, "head")
            print("Database migrations complete.")
        except Exception as e:
            print(f"An error occurred during database migration: {e}")
            # Depending on the policy, you might want to re-raise the exception
            # to prevent the application from starting with a mismatched schema.
            raise

    def get_data(self, query):
        if not self.connection:
            self.connect()
        return self.connection.execute(query).fetchall()

    def close(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
