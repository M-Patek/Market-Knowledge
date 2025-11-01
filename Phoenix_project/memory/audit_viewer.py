import pandas as pd
import json
from typing import Dict, Any, List

from ..monitor.logging import get_logger

logger = get_logger(__name__)

class AuditViewer:
    """
    A utility to read, parse, and analyze the audit log file
    (e.g., 'logs/phoenix_audit.jsonl') created by AuditLogger.
    
    This is used for debugging, analysis, and generating reports.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AuditViewer.
        
        Args:
            config: Main system configuration.
        """
        audit_config = config.get('audit_logger', {})
        self.log_path = audit_config.get('log_path', 'logs/phoenix_audit.jsonl')
        self.full_log: List[Dict] = []
        logger.info(f"AuditViewer initialized for log file: {self.log_path}")

    def load_log(self) -> bool:
        """
        Loads the entire .jsonl audit log into memory.
        
        Returns:
            bool: True on success, False on failure.
        """
        try:
            self.full_log = []
            with open(self.log_path, 'r') as f:
                for line in f:
                    try:
                        self.full_log.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed audit log line: {line[:100]}...")
            
            # Sort by timestamp just in case
            self.full_log.sort(key=lambda x: x.get('timestamp', ''))
            
            logger.info(f"Successfully loaded {len(self.full_log)} audit log entries.")
            return True
            
        except FileNotFoundError:
            logger.error(f"Audit log file not found: {self.log_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load audit log: {e}", exc_info=True)
            return False

    def get_logs_by_type(self, log_type: str) -> List[Dict]:
        """
        Filters the loaded log by 'log_type'.
        
        Args:
            log_type (str): e.g., "TRADE", "FUSION_RESULT", "PIPELINE_IO"
            
        Returns:
            List[Dict]: A list of matching log entries.
        """
        if not self.full_log:
            logger.warning("Log not loaded. Call load_log() first.")
            return []
            
        return [entry for entry in self.full_log if entry.get('log_type') == log_type]

    def get_pipeline_run(self, event_id: str) -> Dict[str, Any]:
        """
        Reconstructs the full pipeline run for a single event_id.
        
        Args:
            event_id (str): The event_id to search for.
            
        Returns:
            Dict[str, Any]: A consolidated dictionary for that event.
        """
        if not self.full_log:
            logger.warning("Log not loaded. Call load_log() first.")
            return {}
            
        run_data = {"event_id": event_id}
        
        for entry in self.full_log:
            log_type = entry.get('log_type')
            data = entry.get('data', {})
            
            if log_type == "EVENT_IN" and data.get('event', {}).get('event_id') == event_id:
                run_data['event_in'] = data['event']
            
            elif log_type == "PIPELINE_IO" and data.get('event_id') == event_id:
                run_data['pipeline_io'] = data['io_bundle']
                
            elif log_type == "FUSION_RESULT" and data.get('result', {}).get('event_id') == event_id:
                run_data['fusion_result'] = data['result']
                
            elif log_type == "STRATEGY_SIGNAL" and data.get('signal', {}).get('metadata', {}).get('event_id') == event_id:
                run_data['signal'] = data['signal']
                
            elif log_type == "TRADE":
                # Trades aren't directly linked to event_id, but to the signal's timestamp
                # This requires more complex time-based correlation
                pass
                
        return run_data
        
    def get_trades_as_df(self) -> pd.DataFrame:
        """Returns all 'TRADE' logs as a pandas DataFrame."""
        trade_logs = self.get_logs_by_type("TRADE")
        if not trade_logs:
            return pd.DataFrame()
            
        # Extract the 'fill' data from each log
        fill_data = [entry['data']['fill'] for entry in trade_logs]
        
        df = pd.DataFrame(fill_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
        return df
        
    def get_pnl_as_df(self) -> pd.DataFrame:
        """Returns all 'PORTFOLIO_SNAPSHOT' logs as a pandas DataFrame."""
        pnl_logs = self.get_logs_by_type("PORTFOLIO_SNAPSHOT")
        if not pnl_logs:
            return pd.DataFrame()
            
        # Extract the 'pnl' data
        pnl_data = [entry['data']['pnl'] for entry in pnl_logs]
        
        df = pd.DataFrame(pnl_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
        return df
