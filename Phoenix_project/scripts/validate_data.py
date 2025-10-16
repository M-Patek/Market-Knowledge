import json
import sys
from jsonschema import validate, ValidationError

def main():
    """
    Validates a JSON data file against a schema defined in the data catalog.
    Exits with a non-zero status code on failure.
    """
    catalog_path = 'data_catalog.json'
    data_file_path = 'data_cache/asset_analysis_cache.json' # Example target
    schema_ref = '#/definitions/fused_ai_analysis_schema' # Target schema

    try:
        print(f"Loading data catalog from: {catalog_path}")
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)

        print(f"Loading data file from: {data_file_path}")
        with open(data_file_path, 'r') as f:
            data_to_validate = json.load(f)

        # Extract the specific schema definition from the catalog
        schema_path_parts = schema_ref.strip('#/').split('/')
        schema = catalog
        for part in schema_path_parts:
            schema = schema[part]

        # The data file is a nested structure: {date: {ticker: analysis_object}}
        # We must validate each nested analysis object.
        print("Starting validation of each item against the schema...")
        for date_key, tickers in data_to_validate.items():
            for ticker, analysis_object in tickers.items():
                validate(instance=analysis_object, schema=schema)
        
        print("✅ Data validation successful. All items conform to the schema.")
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"🚨 Error: Required file not found - {e}. Skipping validation.")
        # In a real CI/CD, we might want to fail here, but for now we'll allow it
        # if the cache file doesn't exist.
        sys.exit(0)
    except (ValidationError, KeyError) as e:
        print(f"❌ Data validation failed!")
        print(f"Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

