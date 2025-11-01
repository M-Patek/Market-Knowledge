import pandas as pd
from typing import Dict, Any, List, Optional
import json
import logging

# 修正: 'CoTDatabase' 在 'cot_database.py' 中不存在。
# 正确的类名是 'CoTDatabaseConnection'。
from memory.cot_database import CoTDatabaseConnection

logger = logging.getLogger(__name__)

class AuditViewer:
    """
    一个用于查询和格式化来自 CoT (思维链) 数据库的
    人类可读审计追踪的工具。
    (Task 16 - 审计追踪查看器)
    """
    
    def __init__(self, db_path: str):
        """
        初始化审计查看器。
        
        Args:
            db_path (str): CoT (SQLite) 数据库文件的路径。
        """
        # 修正: 实例化 CoTDatabaseConnection
        self.db = CoTDatabaseConnection(db_path)
        logger.info(f"AuditViewer initialized with database: {db_path}")

    def get_full_trace_by_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        检索与单个决策 ID 相关联的完整思维链。
        """
        logger.debug(f"Retrieving full trace for decision_id: {decision_id}")
        try:
            # 修正: 调用 CoTDatabaseConnection 上的方法
            records = self.db.get_trace_by_decision_id(decision_id)
            if not records:
                logger.warning(f"No audit trace found for decision_id: {decision_id}")
                return None
            
            # 格式化输出
            return self._format_trace(records)
        
        except Exception as e:
            logger.error(f"Error retrieving trace for {decision_id}: {e}", exc_info=True)
            return None

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        检索最近的 N 个决策及其摘要。
        """
        logger.debug(f"Retrieving last {limit} decisions")
        try:
            # 修正: 调用 CoTDatabaseConnection 上的方法
            records = self.db.get_recent_decisions(limit)
            
            # 聚合每个 decision_id 的记录
            decisions = {}
            for rec in records:
                did = rec['decision_id']
                if did not in decisions:
                    decisions[did] = {
                        "decision_id": did,
                        "timestamp": rec['timestamp'],
                        "steps": []
                    }
                decisions[did]['steps'].append(rec)
            
            # 格式化每个决策
            formatted_decisions = []
            for did, data in decisions.items():
                formatted = self._format_summary(data['steps'])
                formatted['decision_id'] = did
                formatted['timestamp'] = data['timestamp']
                formatted_decisions.append(formatted)
                
            return formatted_decisions
            
        except Exception as e:
            logger.error(f"Error retrieving recent decisions: {e}", exc_info=True)
            return []

    def _format_trace(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将原始数据库记录格式化为结构化的 JSON 追踪。
        """
        if not records:
            return {}
            
        # 按时间戳排序
        records.sort(key=lambda r: r['timestamp'])
        
        trace = {
            "decision_id": records[0]['decision_id'],
            "start_time": records[0]['timestamp'],
            "end_time": records[-1]['timestamp'],
            "steps": []
        }
        
        for rec in records:
            step = {
                "step_name": rec['step_name'],
                "timestamp": rec['timestamp'],
                "status": rec['status'],
                "context": self._safe_json_load(rec['context']),
                "output": self._safe_json_load(rec['output']),
                "error": rec['error']
            }
            trace['steps'].append(step)
            
        return trace

    def _format_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从完整的追踪记录中提取一个高级摘要。
        """
        summary = {
            "overall_status": "FAILURE" if any(r['status'] == 'FAILURE' for r in records) else "SUCCESS",
            "total_steps": len(records),
            "final_decision": None,
            "error_step": None
        }
        
        # 按时间戳排序
        records.sort(key=lambda r: r['timestamp'])

        for rec in records:
            if rec['status'] == 'FAILURE':
                summary['error_step'] = rec['step_name']
            
            # 假设最后一步或 'PortfolioConstruction' 包含最终决策
            if rec['step_name'] == 'PortfolioConstruction' and rec['status'] == 'SUCCESS':
                output = self._safe_json_load(rec['output'])
                summary['final_decision'] = output.get('target_weights', 'N/A')

        # 如果未找到特定步骤，则使用最后一步
        if summary['final_decision'] is None and records:
             output = self._safe_json_load(records[-1]['output'])
             summary['final_decision'] = output if isinstance(output, dict) else str(output)

        return summary

    def _safe_json_load(self, text_data: Optional[str]) -> Any:
        """安全地将 JSON 字符串解析为 Python 对象。"""
        if text_data is None:
            return None
        try:
            return json.loads(text_data)
        except json.JSONDecodeError:
            return text_data # 如果不是 JSON，则按原样返回文本
