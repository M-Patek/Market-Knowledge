import pandas as pd
from typing import List, Dict, Any, Optional

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.execution.signal_protocol import Signal
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.sizing.base import IPositionSizer
# [任务 B.1] 导入 Sizer 实现
from Phoenix_project.sizing.fixed_fraction import FixedFractionSizer
from Phoenix_project.sizing.volatility_parity import VolatilityParitySizer

logger = get_logger(__name__)

class PortfolioConstructor:
    """
    Generates desired target positions based on the L2 fusion signal
    and the risk manager's constraints.
    """
    
    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager):
        """
        Initializes the PortfolioConstructor.
        
        Args:
            config: The strategy configuration.
            risk_manager: The system's risk manager.
        """
        self.config = {}
        self.risk_manager = risk_manager
        
        # [任务 B.1] Sizer 注册表
        self.sizer_registry: Dict[str, IPositionSizer] = {
            "FixedFraction": FixedFractionSizer,
            "VolatilityParity": VolatilityParitySizer,
        }
        self.position_sizer: Optional[IPositionSizer] = None
        self.set_config(config) # Apply initial config
        
        logger.info("PortfolioConstructor initialized.")

    def set_config(self, config: Dict[str, Any]):
        """
        Dynamically updates the component's configuration.
        """
        self.config = config.get('portfolio_constructor', {})
        logger.info(f"PortfolioConstructor config set: {self.config}")
        
        # [任务 B.1] 根据配置实例化仓位管理器
        try:
            sizer_type = self.config.get('position_sizer', 'FixedFraction')
            SizerClass = self.sizer_registry.get(sizer_type)
            
            if SizerClass:
                # 传递 sizer 特定的配置
                sizer_config = self.config.get('sizer_config', {})
                self.position_sizer = SizerClass(**sizer_config)
                logger.info(f"PositionSizer '{sizer_type}' initialized.")
            else:
                logger.error(f"Unknown position sizer type: {sizer_type}. No sizer will be used.")
                self.position_sizer = None
        except Exception as e:
            logger.error(f"Failed to initialize PositionSizer: {e}", exc_info=True)
            self.position_sizer = None

    def generate_orders(
        self, 
        fusion_result: FusionResult, 
        pipeline_state: PipelineState
    ) -> List[Signal]:
        """
        Generates trade signals based on the fusion result and risk constraints.
        
        [任务 B.1] MOCK logic has been replaced.
        """
        
        symbol = fusion_result.symbol
        
        # 1. 获取当前状态
        current_state = pipeline_state.get_current_state()
        market_data = current_state.get_market_data(symbol)
        current_price = market_data.get('close') if market_data else None
        
        if current_price is None or current_price <= 0:
            logger.warning(f"No current price for {symbol} in state. Cannot calculate position size.")
            return []

        current_holdings = current_state.get_holdings(symbol)
        current_balance = current_state.get_balance()
        
        # 计算当前总资产 (一个简化的计算，更复杂的 sizer 可能需要所有资产的价值)
        total_portfolio_value = current_balance + (current_holdings * current_price)
        
        # 2. 检查 Sizer 是否存在
        if not self.position_sizer:
            logger.error("PositionSizer is not initialized. Cannot calculate order size.")
            return []

        # 3. [任务 B.1] 调用 IPositionSizer 接口替换 MOCK 逻辑
        logger.info(f"Calling PositionSizer '{self.position_sizer.__class__.__name__}' for {symbol}...")
        try:
            target_quantity = self.position_sizer.calculate_target_position(
                signal=fusion_result,
                current_price=current_price,
                current_holdings=current_holdings,
                total_portfolio_value=total_portfolio_value
            )
        except Exception as e:
            logger.error(f"PositionSizer failed to calculate position: {e}", exc_info=True)
            return []

        # 4. 计算交易量
        trade_qty = target_quantity - current_holdings
        
        logger.info(f"Sizing for {symbol}: Target Qty={target_quantity}, Current Qty={current_holdings}, Trade Qty={trade_qty}")

        # 修复：TBD 风险管理逻辑
        # TBD: RiskManager can veto or modify the trade_qty here.
        # trade_qty = self.risk_manager.apply_trade_limits(symbol, trade_qty)
        
        # 6. 生成信号
        if abs(trade_qty) < 1e-6: # 仅在有意义的交易时生成
            logger.info(f"Trade quantity for {symbol} is negligible. No signal generated.")
            return []
            
        # 修复：在检查数量之后，但在返回之前，创建 Signal 对象
        signal_obj = Signal(
            symbol=symbol,
            quantity=trade_qty,
            price_target=current_price, # (可改进为使用 L1/L2 的目标价)
            signal_type="TARGET_WEIGHT", # (表示这是基于仓位目标的)
            source_event_id=fusion_result.source_event_id,
            confidence=fusion_result.confidence_score, # 修复：使用 confidence_score
            timestamp=pd.Timestamp.now(tz='UTC')
        )
        
        # 5. (修复) 在此处应用风控覆盖
        logger.debug(f"Applying RiskManager evaluation to signal for {symbol}...")
        adjusted_signal = self.risk_manager.evaluate_and_adjust(signal_obj, pipeline_state)

        # 6. 检查调整后的信号
        if adjusted_signal.quantity == 0.0 and signal_obj.quantity != 0.0:
            logger.warning(f"RiskManager VETOED trade for {symbol}. Reason: {adjusted_signal.metadata.get('risk_veto', 'Unknown')}")
            return []
        
        if adjusted_signal.quantity != signal_obj.quantity:
             logger.warning(f"RiskManager ADJUSTED trade for {symbol}. Original Qty: {signal_obj.quantity}, New Qty: {adjusted_signal.quantity}")
        
        logger.info(f"Generated Signal for {symbol}: Qty={adjusted_signal.quantity}")
        return [adjusted_signal]
