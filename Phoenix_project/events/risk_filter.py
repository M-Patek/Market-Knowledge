from typing import Dict, Any, Optional
from core.pipeline_state import PipelineState
from core.schemas.data_schema import MarketData, NewsData
from monitor.logging import get_logger

logger = get_logger(__name__)

class RiskFilter:
    """
    A specialized event subscriber that listens to raw data events
    and flags high-risk data (e.g., flash crashes, suspicious news).
    
    This acts as an early warning system before the CognitiveEngine.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("risk_filter", {})
        
        # Example threshold: 5% drop in one tick
        self.price_drop_threshold = self.config.get("price_drop_threshold", 0.05)
        self.volume_spike_factor = self.config.get("volume_spike_factor", 10.0) # 10x avg
        
        # Keywords for high-risk news
        self.risk_keywords = set(self.config.get("risk_keywords", [
            "fraud", "halt", "investigation", "crash", "plunge", "SEC"
        ]))
        
        logger.info("RiskFilter initialized.")

    async def on_market_data(self, pipeline_state: PipelineState, data: MarketData):
        """
        Callback for the 'market_data_raw' event.
        Checks for flash crashes or extreme volume.
        """
        
        # 1. Check for price drop
        price_change_pct = (data.close - data.open) / data.open
        if abs(price_change_pct) > self.price_drop_threshold:
            reason = (
                f"Potential flash crash/spike detected for {data.symbol}: "
                f"{price_change_pct:+.2%} change in one tick."
            )
            logger.warning(reason)
            await pipeline_state.event_distributor.publish(
                "HIGH_RISK_DATA",
                data_type="MarketData",
                reason=reason,
                data=data
            )
            
        # 2. Check for volume spike
        # This requires historical avg volume, which should be in PipelineState
        avg_volume = pipeline_state.get_value(f"metrics.{data.symbol}.avg_volume", 0)
        if avg_volume > 0 and data.volume > (avg_volume * self.volume_spike_factor):
            reason = (
                f"Extreme volume spike detected for {data.symbol}: "
                f"{data.volume} vs avg {avg_volume}."
            )
            logger.warning(reason)
            await pipeline_state.event_distributor.publish(
                "HIGH_RISK_DATA",
                data_type="MarketData",
                reason=reason,
                data=data
            )

    async def on_news_data(self, pipeline_state: PipelineState, data: NewsData):
        """
        Callback for the 'news_data_raw' event.
        Checks for high-risk keywords.
        """
        
        text_to_check = (data.headline + " " + (data.summary or "")).lower()
        found_keywords = {kw for kw in self.risk_keywords if kw.lower() in text_to_check}
        
        if found_keywords:
            reason = (
                f"High-risk keywords {found_keywords} detected in news "
                f"from {data.source}: {data.headline}"
            )
            logger.warning(reason)
            await pipeline_state.event_distributor.publish(
                "HIGH_RISK_DATA",
                data_type="NewsData",
                reason=reason,
                data=data
            )

    async def subscribe_to_events(self, event_distributor):
        """Helper to subscribe to all relevant raw data events."""
        await event_distributor.subscribe("market_data_raw", self.on_market_data)
        await event_distributor.subscribe("news_data_raw", self.on_news_data)
        logger.info("RiskFilter subscribed to raw data events.")
