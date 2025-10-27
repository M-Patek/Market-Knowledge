import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import redis
import time
from enum import Enum
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# 设置 OpenTelemetry
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

def get_logger(name: str):
    """获取一个标准化的日志记录器。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class CanaryMonitor:
    """
    负责监控金丝雀部署的性能，并在出现问题时触发回滚。
    """
    def __init__(self, pipeline_orchestrator, prediction_server, config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.CanaryMonitor")
        self.pipeline_orchestrator = pipeline_orchestrator
        self.prediction_server = prediction_server
        self.config = config.get('canary_monitor', {})
        # 冠军模型的基线指标
        self.champion_baseline = {"avg_variance": 0.05, "std_dev_variance": 0.015}
        self.is_monitoring = False
        self.challenger_metrics: List[Dict[str, float]] = []
        # --- [Sub-Task 1.1.2: Canary Timer Logic] ---
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.challenger_model_id: Optional[str] = None
        self.challenger_deployment_time: Optional[datetime] = None
        self.monitoring_thread: Optional[threading.Thread] = None

    def start_challenger_observation(self, model_id: str):
        """
        Starts the monitoring process for a newly deployed challenger model.
        This method is intended to be called by the PredictionServer.
        """
        if self.is_monitoring:
            self.logger.warning("Already monitoring a challenger. Ignoring new request.")
            return

        self.challenger_model_id = model_id
        self.challenger_deployment_time = datetime.utcnow()
        self.is_monitoring = True
        self.challenger_metrics = [] # Reset metrics for the new challenger
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Canary monitoring started for challenger '{model_id}'. 72-hour observation period has begun.")

    def _monitor_loop(self):
        """The internal monitoring loop that runs in a background thread."""
        while self.is_monitoring:
            time.sleep(self.config.get('monitoring_interval_seconds', 60))
            self._check_for_rollback()
            self._check_for_promotion()

    def stop_monitoring(self):
        """停止监控循环。"""
        self.is_monitoring = False
        self.challenger_model_id = None
        self.challenger_deployment_time = None
        self.logger.info("金丝雀监控器已停止。")

    def _check_for_rollback(self):
        """检查挑战者的指标是否突破了安全阈值。"""
        if not self.prediction_server.challenger_model: return

        # This is a simulation; we would typically aggregate metrics over the interval
        dummy_features = {}
        challenger_pred = self.prediction_server.predict(dummy_features)
        self.challenger_metrics.append(challenger_pred)
        current_avg_variance = sum(m['variance'] for m in self.challenger_metrics) / len(self.challenger_metrics)
        variance_threshold = self.champion_baseline['avg_variance'] + (self.config.get('variance_std_dev_threshold', 2.0) * self.champion_baseline['std_dev_variance'])

        if current_avg_variance > variance_threshold:
            self.logger.critical(f"触发回滚: 挑战者方差 ({current_avg_variance:.4f}) 超出阈值 ({variance_threshold:.4f})。")
            self.pipeline_orchestrator.trigger_rollback()
            self.stop_monitoring()

    def _check_for_promotion(self):
        """Checks if the challenger has survived the observation period."""
        if not self.challenger_deployment_time or not self.challenger_model_id:
            return

        elapsed = datetime.utcnow() - self.challenger_deployment_time
        if elapsed >= timedelta(hours=72):
            self.logger.info(f"Challenger '{self.challenger_model_id}' survived the 72-hour canary period.")
            self.redis_client.lpush('AWAITING_SHADOW_DEPLOYMENT', self.challenger_model_id)
            self.logger.info(f"Model '{self.challenger_model_id}' pushed to Redis queue 'AWAITING_SHADOW_DEPLOYMENT'.")
            self.stop_monitoring() # The observation period is over.


class ShadowMonitor:
    """
    [Sub-Task 1.1.3] Compares the performance of the shadow model against the champion
    by analyzing the decision logs from the AuditManager.
    """
    def __init__(self, audit_manager):
        self.logger = logging.getLogger("PhoenixProject.ShadowMonitor")
        self.audit_manager = audit_manager # A-coupler
        self.logger.info("ShadowMonitor initialized.")

    def start_monitoring(self):
        """Starts the process of comparing shadow and champion decisions."""
        self.logger.info("Shadow monitoring process started.")
        # TODO: Implement logic to subscribe to or periodically query AuditManager logs.
        # TODO: Calculate and compare cumulative simulated P&L for both models.
        # TODO: Generate a comparison report for the final promotion approval stage.

    def get_promotion_candidates_report(self) -> Dict[str, Any]:
        """
        [Sub-Task 1.1.4] Generates the comparison report for models awaiting final promotion.
        """
        self.logger.info("Generating promotion candidates report...")
        # TODO: This should be driven by a real queue of models that have passed shadow mode.
        # For now, we return a mock report for a single candidate.
        mock_model_id = "mlflow_run_abc123"
        report = {
            "model_id": mock_model_id,
            "comparison_report": {
                "period_days": 30,
                "champion_simulated_pnl": 50234.56,
                "shadow_simulated_pnl": 55102.10,
                "decision_concordance_rate": 0.88 # Percentage of time decisions were the same
            }
        }
        return {"promotion_candidates": [report]}


class PerformanceMonitor:
    """
    [Sub-Task 1.2.1] Monitors the live performance of the champion model to detect
    degradation and trigger an adaptive retraining cycle.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.PerformanceMonitor")
        self.config = config.get('performance_monitor', {})
        # In a real system, this would be a rolling window of daily portfolio returns
        self.portfolio_returns_history: List[float] = []
        self.cognitive_uncertainty_history: List[float] = []
        self.logger.info("PerformanceMonitor initialized.")

    def start_monitoring(self):
        """Starts the main monitoring loop."""
        self.logger.info("Performance monitoring has started.")
        # TODO: This would be an event-driven loop that subscribes to daily/weekly portfolio updates.
        # For now, we will just have a placeholder method.

    def _check_performance_degradation(self):
        """Calculates rolling metrics and checks if they breach retraining thresholds."""
        # TODO: Calculate 30-day rolling Sharpe Ratio from self.portfolio_returns_history.
        # TODO: Calculate current drawdown from self.portfolio_returns_history.
        # TODO: Calculate average cognitive uncertainty from self.cognitive_uncertainty_history.
        # TODO: If any metric breaches the thresholds defined in the config,
        #       insert a record into the RetrainingRecommendations database table.
        self.logger.info("Checking for performance degradation (placeholder).")


class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    [Sub-Task 2.3.1] A reusable Circuit Breaker to protect against repeated failures of an external service.
    """
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.recovery_time = 0

    def call(self, func, *args, **kwargs):
        """Executes the function if the circuit is closed or half-open."""
        if self.state == CircuitBreakerState.OPEN:
            if time.monotonic() > self.recovery_time:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise ConnectionError("Circuit breaker is open.")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e

    def record_failure(self):
        """Records a failure. If the threshold is met, opens the circuit."""
        self.failures += 1
        if self.state == CircuitBreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.recovery_time = time.monotonic() + self.recovery_timeout
            logging.warning(f"Circuit breaker has been opened for {self.recovery_timeout} seconds.")

    def record_success(self):
        """Records a success, closing the circuit and resetting failure count."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            logging.info("Circuit breaker has been reset to closed after successful call.")
        self.state = CircuitBreakerState.CLOSED
        self.failures = 0


class ApiCostTracker:
    """
    [Sub-Task 2.3.2] Tracks API call counts within a 1-hour rolling window to
    prevent cost overruns.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config.get('cost_breaker', {}) if config else {}
        # Using db=2 to keep cost counters separate from other Redis data
        self.redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=True)

    def _get_key_for_source(self, api_source: str) -> str:
        """Generates a consistent Redis key for a given API source."""
        return f"cost_tracker:{api_source}"

    def log_call(self, api_source: str):
        """Logs a single API call, creating a 1-hour rolling window in Redis on the first call."""
        key = self._get_key_for_source(api_source)
        count = self.redis_client.incr(key)
        if count == 1:
            # This is a new key, so we set its expiration to 1 hour (3600 seconds)
            self.redis_client.expire(key, 3600)

    def is_cost_limit_exceeded(self, api_source: str) -> bool:
        """Checks if the call count for an API source exceeds its configured threshold."""
        key = self._get_key_for_source(api_source)
        threshold = self.config.get('hourly_limit', {}).get(api_source, 1000) # Default to 1000 calls/hr
        current_count = int(self.redis_client.get(key) or 0)
        return current_count > threshold
