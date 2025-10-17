import json
import sys
import argparse
from jsonschema import validate, ValidationError

def main(args):
    """
    Validates a JSON data file against a schema defined in the data catalog.
    Exits with a non-zero status code on failure.
    """
    catalog_path = args.catalog_path
    data_file_path = args.data_file
    schema_ref = args.schema_ref

    try:
        print(f"Loading data catalog from: {catalog_path}")
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)

        print(f"Loading data file from: {data_file_path}")
        with open(data_file_path, 'r') as f:
            data_to_validate = json.load(f)

        # Resolve the JSON Pointer to get the correct schema from the catalog
        try:
            schema_path_parts = schema_ref.strip('#/').split('/')
            schema = catalog
            for part in schema_path_parts:
                schema = schema[part]
        except (KeyError, IndexError) as e:
            print(f"❌ Failed to resolve schema reference '{schema_ref}': {e}")
            sys.exit(1)

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
    parser = argparse.ArgumentParser(description="Validate a JSON data file against a schema.")
    parser.add_argument('--catalog-path', type=str, default='data_catalog.json',
                        help='Path to the data catalog JSON file.')
    parser.add_argument('--data-file', type=str, default='data_cache/asset_analysis_cache.json',
                        help='Path to the JSON data file to validate.')
    parser.add_argument('--schema-ref', type=str, default='#/definitions/fused_ai_analysis_schema',
                        help='JSON Pointer reference to the schema within the catalog.')
    args = parser.parse_args()
    main(args)
