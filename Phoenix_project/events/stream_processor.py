from typing import Dict, Any, List
from core.schemas.data_schema import MarketData, NewsData
from core.pipeline_state import PipelineState
from events.event_distributor import EventDistributor
from monitor.logging import get_logger

logger = get_logger(__name__)

class StreamProcessor:
    """
    Handles the initial ingestion of raw data streams (live or backtest).
    It validates data against schemas, publishes raw events, and updates
    the PipelineState buffers.
    """

    def __init__(
        self,
        pipeline_state: PipelineState,
        event_distributor: EventDistributor
    ):
        self.pipeline_state = pipeline_state
        self.event_distributor = event_distributor
        logger.info("StreamProcessor initialized.")

    async def process_data_batch(self, data_batch: Dict[str, List[Any]]):
        """
        Processes a batch of data from the DataManager.
        
        Args:
            data_batch (Dict[str, List[Any]]): A dict containing lists of
                                              Pydantic data models.
                                              e.g., {"market_data": [...], "news_data": [...]}
        """
        
        # 1. Process Market Data
        market_data_list = data_batch.get("market_data", [])
        validated_market_data = []
        for data in market_data_list:
            try:
                # Data should already be validated by DataManager/API client
                # but we can re-validate if needed.
                if isinstance(data, MarketData):
                    validated_market_data.append(data)
                    # Publish the raw event for other listeners (e.g., RiskFilter)
                    await self.event_distributor.publish(
                        "market_data_raw", data=data
                    )
                else:
                    logger.warning(f"Skipping invalid market data item: {type(data)}")
            except Exception as e:
                logger.error(f"Failed to validate market data: {e}", exc_info=True)
                
        # 2. Process News Data
        news_data_list = data_batch.get("news_data", [])
        validated_news_data = []
        for data in news_data_list:
            try:
                if isinstance(data, NewsData):
                    validated_news_data.append(data)
                    await self.event_distributor.publish(
                        "news_data_raw", data=data
                    )
                else:
                    logger.warning(f"Skipping invalid news data item: {type(data)}")
            except Exception as e:
                logger.error(f"Failed to validate news data: {e}", exc_info=True)
                
        # ... Process other data types (economic, etc.)
        
        # 3. Update PipelineState buffers
        try:
            update_dict = {}
            if validated_market_data:
                update_dict["market_data"] = validated_market_data
            if validated_news_data:
                update_dict["news_data"] = validated_news_data
            
            if update_dict:
                await self.pipeline_state.update_state(update_dict)
                logger.info(
                    f"StreamProcessor updated state with "
                    f"{len(validated_market_data)} market, "
                    f"{len(validated_news_data)} news items."
                )
                
            # 4. Publish a "processed" event to trigger the main cycle
            await self.event_distributor.publish(
                "data_batch_processed",
                batch_size=len(market_data_list) + len(news_data_list)
            )
            
        except Exception as e:
            logger.error(f"Failed to update PipelineState with data batch: {e}", exc_info=True)
