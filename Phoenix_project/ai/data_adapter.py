from typing import List, Dict, Any, Union
import pandas as pd
from core.schemas.data_schema import MarketData, NewsData, EconomicIndicator

class DataAdapter:
    """
    Transforms and formats diverse data types (market, news, etc.)
    into a unified string representation suitable for LLM prompts.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Max tokens per data type to avoid overly long prompts
        self.max_market_tokens = self.config.get("max_market_tokens", 1024)
        self.max_news_tokens = self.config.get("max_news_tokens", 1024)
        self.max_economic_tokens = self.config.get("max_economic_tokens", 512)

    def format_market_data(self, market_data: List[MarketData]) -> str:
        """Converts a list of MarketData objects into a compact string."""
        if not market_data:
            return "No recent market data available."

        # Create a DataFrame for easy manipulation
        try:
            df = pd.DataFrame([md.model_dump() for md in market_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by="timestamp", ascending=False)
            
            # Summarize: Get latest, and maybe some aggregates
            latest = df.iloc[0]
            summary = (
                f"Latest Market Data ({latest['symbol']} at {latest['timestamp']}):\n"
                f"Open: {latest['open']:.2f}, High: {latest['high']:.2f}, "
                f"Low: {latest['low']:.2f}, Close: {latest['close']:.2f}, "
                f"Volume: {latest['volume']}\n"
            )
            
            # Add simple trend info
            if len(df) > 1:
                prev_close = df.iloc[1]['close']
                change = latest['close'] - prev_close
                change_pct = (change / prev_close) * 100
                summary += f"Change: {change:+.2f} ({change_pct:+.2f}%)\n"
            
            # Simple technical indicator (e.g., 5-period SMA)
            if len(df) >= 5:
                sma_5 = df['close'].head(5).mean()
                summary += f"5-Period SMA: {sma_5:.2f}\n"

            # Truncate if necessary (simplified token counting)
            if len(summary) > self.max_market_tokens:
                summary = summary[:self.max_market_tokens] + "... [Truncated]"
                
            return summary
        except Exception as e:
            return f"Error formatting market data: {e}"

    def format_news_data(self, news_data: List[NewsData]) -> str:
        """Converts a list of NewsData objects into a compact string."""
        if not news_data:
            return "No recent news available."

        # Sort by timestamp, most recent first
        sorted_news = sorted(news_data, key=lambda x: x.timestamp, reverse=True)
        
        output = "Recent News:\n"
        total_len = len(output)
        
        for item in sorted_news:
            # Format: [Timestamp] (Source) Headline: Summary
            headline = item.headline or "No Headline"
            summary = item.summary or "No Summary"
            source = item.source or "Unknown Source"
            
            entry = (
                f"- [{item.timestamp.isoformat()}] ({source}) {headline}: {summary}\n"
            )
            
            if total_len + len(entry) > self.max_news_tokens:
                output += "... [Truncated]"
                break
                
            output += entry
            total_len += len(entry)
            
        return output

    def format_economic_data(self, economic_data: List[EconomicIndicator]) -> str:
        """Converts a list of EconomicIndicator objects into a compact string."""
        if not economic_data:
            return "No recent economic indicators available."
            
        # Sort by timestamp, most recent first
        sorted_data = sorted(economic_data, key=lambda x: x.timestamp, reverse=True)

        output = "Recent Economic Indicators:\n"
        total_len = len(output)

        for item in sorted_data:
            # Format: [Timestamp] Indicator: Value (Previous: X, Consensus: Y)
            entry = f"- [{item.timestamp.isoformat()}] {item.name}: {item.value}"
            details = []
            if item.previous:
                details.append(f"Previous: {item.previous}")
            if item.consensus:
                details.append(f"Consensus: {item.consensus}")
            
            if details:
                entry += f" ({'; '.join(details)})\n"
            else:
                entry += "\n"

            if total_len + len(entry) > self.max_economic_tokens:
                output += "... [Truncated]"
                break
                
            output += entry
            total_len += len(entry)
            
        return output

    def format_context(self, context: Dict[str, List[Any]]) -> str:
        """
        Takes a dictionary of context data (from PipelineState)
        and formats it all into a single string.
        """
        formatted_strings = []
        
        if "market_data" in context:
            formatted_strings.append(
                self.format_market_data(context["market_data"])
            )
        if "news_data" in context:
            formatted_strings.append(
                self.format_news_data(context["news_data"])
            )
        if "economic_data" in context:
            formatted_strings.append(
                self.format_economic_data(context["economic_data"])
            )
        
        # Add other data types as needed
        # ...

        return "\n\n".join(formatted_strings)
