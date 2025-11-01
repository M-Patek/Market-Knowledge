from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import yaml
import re
import asyncio

# 假设 PipelineState 和 GeminiPoolManager, PromptManager 会被正确导入
# 实际项目中可能需要像这样：
# from ..core.pipeline_state import PipelineState
# from ..api.gemini_pool_manager import GeminiPoolManager
# from ..ai.prompt_manager import PromptManager


# --- 新增：补全缺失的 BaseReasoner 抽象基类 ---
class BaseReasoner(ABC, BaseModel):
    """
    推理器抽象基类，供 ReasoningEnsemble 使用。
    """
    reasoner_id: str = Field(..., description="推理器的唯一标识符")

    @abstractmethod
    async def reason(self, state: "PipelineState", context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行推理逻辑。
        
        Args:
            state (PipelineState): 当前的流水线状态。
            context (Dict[str, Any]): 相关的上下文信息 (e.g., 检索到的数据)。

        Returns:
            Dict[str, Any]: 推理结果。
        """
        pass

    class Config:
        arbitrary_types_allowed = True


# --- 现有代码：更新 SymbolicReasoner 以继承 BaseReasoner ---
class SymbolicReasoner(BaseReasoner):
    """
    Applies symbolic rules (e.g., YAML-defined logic) to the current state.
    """
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    reasoner_id: str = "symbolic_reasoner" # 提供默认ID

    def __init__(self, rules_path: str, **data):
        """
        Load rules from a YAML file.
        """
        # We call super().__init__ first using the **data
        # This populates fields defined in BaseModel and BaseReasoner
        super().__init__(**data) 
        
        try:
            with open(rules_path, 'r') as f:
                self.rules = yaml.safe_load(f).get('rules', [])
        except Exception as e:
            print(f"Error loading symbolic rules from {rules_path}: {e}") # 替换为 logger
            self.rules = []

    def _evaluate_condition(self, condition: str, state: "PipelineState", context: Dict[str, Any]) -> bool:
        """
        Evaluates a single condition string against the state and context.
        This is a simplified evaluator.
        """
        # Example condition: "state.market_condition == 'volatile'"
        # WARNING: Using eval() is unsafe. A real implementation MUST use a safe expression parser.
        # This is just a placeholder.
        try:
            # 模拟状态访问
            if "state.market_condition" in condition:
                # 假设 state 有 market_condition 属性
                if hasattr(state, 'market_condition'):
                    # 简化演示，实际应使用 AST 解析器
                    val = condition.split("==")[-1].strip().strip("'\"")
                    return state.market_condition == val
            
            # 模拟上下文访问
            # Example: "context.news_sentiment > 0.5"
            if "context.news_sentiment" in condition:
                 if 'news_sentiment' in context:
                    # 简化演示
                    val = float(condition.split(">")[-1].strip())
                    return context['news_sentiment'] > val

        except Exception as e:
            print(f"Error evaluating condition '{condition}': {e}") # 替换为 logger
        return False

    async def reason(self, state: "PipelineState", context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies loaded symbolic rules.
        """
        results = []
        for rule in self.rules:
            try:
                conditions_met = all(
                    self._evaluate_condition(cond, state, context) 
                    for cond in rule.get('conditions', [])
                )
                if conditions_met:
                    results.append(rule['action'])
            except Exception as e:
                print(f"Error applying rule '{rule.get('name', 'Unnamed')}': {e}") # 替换为 logger
        
        return {"symbolic_output": results, "confidence": 1.0, "reasoner_id": self.reasoner_id}


# --- 现有代码：更新 LLMReasoner 以继承 BaseReasoner ---
class LLMReasoner(BaseReasoner):
    """
    Uses an LLM (via GeminiPoolManager) to reason about the state and context.
    """
    gemini_pool: "GeminiPoolManager"
    prompt_manager: "PromptManager"
    prompt_name: str = Field(..., description="Name of the prompt template to use")
    reasoner_id: str = "llm_reasoner" # 提供默认ID

    def __init__(self, gemini_pool: "GeminiPoolManager", prompt_manager: "PromptManager", prompt_name: str, **data):
        super().__init__(gemini_pool=gemini_pool, prompt_manager=prompt_manager, prompt_name=prompt_name, **data)

    async def reason(self, state: "PipelineState", context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates reasoning using an LLM.
        """
        try:
            # 1. 渲染提示
            prompt_data = {
                "state": state.model_dump_json(), # 序列化 Pydantic 模型
                "context": context,
                # ... (添加其他所需数据)
            }
            rendered_prompt = self.prompt_manager.render(self.prompt_name, prompt_data)
            
            # 2. 调用 LLM
            # (假设 gemini_pool 有一个 'generate' 方法)
            response_text = await self.gemini_pool.generate_text(
                prompt=rendered_prompt,
                model_name="default" # 或者从配置中获取
            )
            
            # 3. (可选) 解析LLM的结构化输出
            
            return {"llm_output": response_text, "confidence": 0.8, "reasoner_id": self.reasoner_id} # 示例置信度
        except Exception as e:
            print(f"Error during LLM reasoning: {e}") # 替换为 logger
            return {"llm_output": None, "confidence": 0.0, "error": str(e), "reasoner_id": self.reasoner_id}


class ReasoningEnsemble:
    """
    Manages and executes multiple heterogeneous reasoners (symbolic, LLM, etc.).
    """
    
    reasoners: List[BaseReasoner] = Field(default_factory=list)

    def __init__(self, config: Dict[str, Any], gemini_pool: "GeminiPoolManager", prompt_manager: "PromptManager"):
        """
        Initializes the ensemble based on configuration.
        """
        self.reasoners = []
        
        # 示例：从配置动态加载推理器
        for reasoner_config in config.get('reasoners', []):
            try:
                if reasoner_config['type'] == 'symbolic':
                    self.add_reasoner(
                        SymbolicReasoner(
                            rules_path=reasoner_config['rules_path'],
                            reasoner_id=reasoner_config.get('id', 'symbolic_default')
                        )
                    )
                elif reasoner_config['type'] == 'llm':
                    self.add_reasoner(
                        LLMReasoner(
                            gemini_pool=gemini_pool,
                            prompt_manager=prompt_manager,
                            prompt_name=reasoner_config['prompt_name'],
                            reasoner_id=reasoner_config.get('id', 'llm_default')
                        )
                    )
            except KeyError as e:
                 print(f"Missing config key for reasoner: {e}") # 替换为 logger
            except Exception as e:
                 print(f"Error initializing reasoner: {e}") # 替换为 logger


    async def run(self, state: "PipelineState", context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Runs all reasoners in parallel and collects their outputs.
        """
        tasks = [
            reasoner.reason(state, context) 
            for reasoner in self.reasoners
        ]
        results = await asyncio.gather(*tasks)
        return results

    def add_reasoner(self, reasoner: BaseReasoner):
        """
        Adds a pre-initialized reasoner to the ensemble.
        """
        if isinstance(reasoner, BaseReasoner):
            self.reasoners.append(reasoner)
        else:
            raise TypeError("Reasoner must be an instance of BaseReasoner")

