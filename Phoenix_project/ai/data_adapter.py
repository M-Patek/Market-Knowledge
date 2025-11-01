from typing import List, Dict, Any
import pandas as pd
import json
from ..core.schemas.data_schema import MarketData, NewsData, AlternativeData

logger = logging.getLogger(__name__)

class DataAdapter:
    """
    (L1) Converts structured data objects into a text format suitable for LLM consumption.
    """

    @staticmethod
    def to_llm_text(data_points: List[DataSchema]) -> str:
        """
        Transforms a list of DataSchema objects into a structured text block.

        Args:
            data_points: A list of DataSchema objects.

        Returns:
            A single string formatted for an LLM prompt.
        """
        if not data_points:
            return "No data points available."

        text_blocks = []
        for dp in data_points:
            # Basic formatting, can be made more sophisticated
            block = (
                f"On {dp.timestamp.strftime('%Y-%m-%d')}, "
                f"source '{dp.source}' reported for symbol '{dp.symbol}': "
                f"value = {dp.value}."
            )
            if dp.metadata:
                meta_str = ', '.join([f"{k}: {v}" for k, v in dp.metadata.items()])
                block += f" (Metadata: {meta_str})"
            text_blocks.append(block)
        
        logger.info(f"Adapted {len(data_points)} data points to LLM text format.")
        return "\n".join(text_blocks)
