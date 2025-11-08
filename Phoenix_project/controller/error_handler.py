import asyncio
import httpx  # [✅ 修复] 阶段 5 添加
import os     # [✅ 修复] 阶段 5 添加
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class ErrorHandler:
    """
    Centralized error handling component.
    Responds to critical errors, manages retries, and can trigger
    system-wide safety mechanisms (like circuit breakers).
    """

    def __init__(self, config: dict):
        self.config = config.get("error_handler", {})
        self.max_retries = self.config.get("max_retries", 3)
        # [✅ 修复] 键名与 system.yaml (Source 31) 中的 "retry_delay_seconds" 保持一致
        self.retry_delay_base = self.config.get("retry_delay_seconds", 5) 
        
        # Track failures for specific components
        self.failure_counts = {}
        
        # --- [✅ 修复] 阶段 5 添加 ---
        self.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if not self.slack_webhook_url:
            logger.warning("ErrorHandler: SLACK_WEBHOOK_URL 未设置。将跳过 Slack 警报。")
        # --- [修复结束] ---
        
        logger.info("ErrorHandler initialized.")

    async def handle_error(
        self,
        error: Exception,
        component: str,
        context: dict,
        # We need a way to trigger actions, e.g., via the orchestrator
        # or event distributor, passed during init.
        # For simplicity, we'll just log and suggest actions.
    ):
        """
        Main error handling entry point.
        
        Args:
            error (Exception): The exception that occurred.
            component (str): Name of the component that failed (e.g., "CognitiveEngine").
            context (dict): Context about what was happening (e.g., "decision_id").
        """
        
        decision_id = context.get("decision_id", "N/A")
        logger.error(
            f"Critical error in component '{component}' during cycle '{decision_id}': {error}",
            exc_info=True
        )
        
        # Update failure count
        self.failure_counts[component] = self.failure_counts.get(component, 0) + 1
        
        # --- Decision Logic ---
        
        # 1. Check for retries (if applicable to the error type)
        # This is complex; the *caller* usually manages its own retries.
        # This handler is more for *unrecoverable* errors.
        
        # 2. Check for circuit breaker
        if self.failure_counts[component] > self.max_retries:
            logger.critical(
                f"Component '{component}' has failed {self.failure_counts[component]} consecutive times. "
                "This may require a circuit breaker!"
            )
            # In a real system:
            # await self.risk_manager.trip_circuit_breaker(
            #     f"Component '{component}' failed repeatedly."
            # )
            
        # 3. Send notification (e.g., to Sentry, PagerDuty)
        await self.send_alert(error, component, context)
        
        # 4. Determine recovery strategy
        # For now, we just log. A real handler might try to
        # restart a component or switch to a fallback.
        
    async def send_alert(self, error: Exception, component: str, context: dict):
        """[✅ 修复] 阶段 5 修复：发送警报到 Slack (替换占位符)。"""
        alert_message = (
            f"🔥 Phoenix Project 严重警报 🔥\n"
            f"Component: {component}\n"
            f"Error: {str(error)}\n"
            f"Context: {str(context)}\n" # [修复] 确保 context 被序列化为 str
        )

        # 仍然在本地日志中记录
        logger.info(f"--- ALERT (Sending) ---\n{alert_message}")

        if not self.slack_webhook_url:
            # (已经在 __init__ 中警告过了, 这里可以安静跳过)
            return

        payload = {"text": alert_message}
        try:
            # 使用 httpx (已在 requirements.txt 中) 异步发送
            async with httpx.AsyncClient() as client:
                response = await client.post(self.slack_webhook_url, json=payload)
                response.raise_for_status() # 如果是 4xx/5xx 则抛出异常
            logger.info("警报已成功发送至 Slack。")
        except Exception as e:
            # 即使 Slack 发送失败，也不应让 ErrorHandler 崩溃
            logger.error(f"发送 Slack 警报失败: {e}", exc_info=True)

        # [✅ 修复] 移除旧的占位符 sleep
        # await asyncio.sleep(0.01)

    def reset_failure_count(self, component: str):
        """Resets the failure count for a component upon success."""
        if component in self.failure_counts:
            logger.info(f"Component '{component}' recovered. Resetting failure count.")
            self.failure_counts[component] = 0
