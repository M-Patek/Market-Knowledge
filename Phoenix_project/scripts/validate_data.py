import json
import sys
import os
import argparse
import pydantic # Required for schema validation

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# FIX: Import the entire data_schema module
import core.schemas.data_schema as data_schemas
from monitor.logging import get_logger

logger = get_logger('DataValidator')

# Define the mapping from schema type argument to the class name in data_schema.py
SCHEMA_NAME_MAP = {
    'market_event': 'MarketEvent',
    'economic_indicator': 'EconomicIndicator',
    'corporate_action': 'CorporateAction',
    'analyst_rating': 'AnalystRating',
    'social_media': 'SocialMediaPost',
    'news_article': 'NewsArticle',
    'market_data': 'MarketData',
    'fused_analysis': 'FusedAnalysis',
    'portfolio_decision': 'PortfolioDecision'
}

def load_data(file_path: str):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

# FIX: Updated function to use Pydantic model for validation
def validate_with_schema(data: dict, schema: pydantic.BaseModel) -> bool:
    """
    Validates a single data item against the provided Pydantic schema class.
    """
    if not isinstance(data, dict):
        logger.warning(f"Data item is not a dictionary, skipping: {type(data)}")
        return False
        
    try:
        # Attempt to instantiate the model. This triggers validation.
        schema(**data)
        return True
    except pydantic.ValidationError as e:
        item_id = data.get('id', data.get('article_id', 'N/A'))
        logger.warning(f"Validation Error for item [ID: {item_id}]:\n{e}")
        return False
    except Exception as e:
        item_id = data.get('id', data.get('article_id', 'N/A'))
        logger.error(f"An unexpected error occurred during validation for item [ID: {item_id}]: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate data files against core schemas.")
    parser.add_argument("file_path", type=str, help="Path to the JSON data file to validate.")
    parser.add_argument(
        "--schema-type", 
        type=str, 
        default="market_event",
        choices=SCHEMA_NAME_MAP.keys(),
        help=f"The type of schema to validate against. Defaults to 'market_event'. Choices: {list(SCHEMA_NAME_MAP.keys())}"
    )
    
    args = parser.parse_args()
    
    file_path = args.file_path
    schema_type = args.schema_type

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    # FIX: Dynamically get the schema class from the imported module
    schema_name = SCHEMA_NAME_MAP.get(schema_type)
    if not schema_name:
        # This should be caught by argparse choices, but as a safeguard:
        logger.error(f"Error: Unknown schema type '{schema_type}'")
        sys.exit(1)
        
    schema_class = getattr(data_schemas, schema_name, None)
    if not schema_class:
        logger.error(f"Error: Schema class '{schema_name}' not found in core.schemas.data_schema.py")
        sys.exit(1)
        
    logger.info(f"Validating '{file_path}' against schema: {schema_name}")

    data = load_data(file_path)
    if data is None:
        logger.error("Failed to load data. Exiting.")
        sys.exit(1)

    total_items = 0
    valid_items = 0
    
    # Data can be a single object or a list of objects
    if isinstance(data, list):
        total_items = len(data)
        if total_items == 0:
            logger.warning("Data file is an empty list.")
            sys.exit(0)
        
        # Validate each item in the list
        results = [validate_with_schema(item, schema_class) for item in data]
        valid_items = sum(results)
        
    elif isinstance(data, dict):
        total_items = 1
        # Validate the single object
        if validate_with_schema(data, schema_class):
            valid_items = 1
    else:
        logger.error(f"Data file does not contain a JSON object or list of objects. Found: {type(data)}")
        sys.exit(1)

    # --- Report Results ---
    if total_items > 0:
        valid_percentage = (valid_items / total_items) * 100
        logger.info("--- Validation Summary ---")
        logger.info(f"Total items checked: {total_items}")
        logger.info(f"Valid items: {valid_items}")
        logger.info(f"Invalid items: {total_items - valid_items}")
        logger.info(f"Pass Rate: {valid_percentage:.2f}%")
        
        if valid_items == total_items:
            logger.info("Validation SUCCESS: All items conform to the schema.")
            sys.exit(0) # Exit with success code
        else:
            logger.error("Validation FAILED: One or more items do not conform to the schema.")
            sys.exit(1) # Exit with failure code
    else:
        logger.warning("No items were validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
