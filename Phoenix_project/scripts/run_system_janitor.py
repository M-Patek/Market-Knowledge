"""
System Janitor (全能保洁机器人)

一个计划内的后台任务，用于清理所有过时的数据，防止存储溢出。
包括：Pinecone 向量, Elasticsearch CoT 记录, 审计数据库, 和本地快照文件。
"""
import os
import glob
import time
from datetime import datetime, timedelta

# [新] 导入所有依赖
try:
    from Phoenix_project.config.loader import ConfigLoader
    from Phoenix_project.monitor.logging import get_logger
    from Phoenix_project.memory.vector_store import PineconeVectorStore
    from Phoenix_project.memory.cot_database import ElasticsearchCoTDatabase
    from Phoenix_project.audit.logger import AuditLogger
    from Phoenix_project.snapshot_manager import SnapshotManager
except ImportError:
    print("错误：无法导入 Phoenix Project 模块。请确保从项目根目录运行。")
    exit(1)


# --- 全局初始化 ---
# (这假设 Janitor 是从项目根目录运行的)
logger = get_logger(__name__)
config_loader = ConfigLoader(config_path="config")
system_config = config_loader.load_config('system.yaml')
if not system_config:
    logger.critical("Janitor 无法加载 system.yaml，任务中止。")
    exit(1)

pipeline_config = system_config.get('pipeline', {})

def get_retention_days(config_key: str, default: int) -> int:
    """从配置中安全地获取保留天数"""
    days = pipeline_config.get(config_key, default)
    logger.info(f"配置 {config_key} 保留天数: {days} 天")
    return days

# --- 清理函数 ---

def clean_pinecone_vectors(vector_store: PineconeVectorStore):
    """
    清理 1: [已修复] 删除 Pinecone 中的过时命名空间
    """
    try:
        # 我们需要一个比向量保留期更早的时间来删除
        # 例如，保留 90 天，我们在 120 天时删除 (确保数据已不再需要)
        retention_days = get_retention_days('vector_retention_days', 90)
        # 我们删除 90 天前的那个月份的命名空间
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # 确定要删除的命名空间 (例如 "ns-2025-08")
        ns_to_delete = cutoff_date.strftime("ns-%Y-%m")
        
        # [安全检查] 永远不要删除当月的命名空间！
        current_ns = datetime.utcnow().strftime("ns-%Y-%m")
        if ns_to_delete == current_ns:
            logger.warning(f"Pinecone 清理跳过：{ns_to_delete} 是当前月份。")
            return

        logger.info(f"正在清理 {retention_days} 天前的 Pinecone 命名空间: {ns_to_delete}...")
        
        # [已修复] 调用我们在 vector_store.py 中添加的新方法
        vector_store.delete_by_namespace(namespace=ns_to_delete)
        
        logger.info(f"Pinecone 命名空间 {ns_to_delete} 清理完成。")

    except Exception as e:
        logger.error(f"Pinecone 清理失败: {e}", exc_info=True)

def clean_elasticsearch_cot(cot_db: ElasticsearchCoTDatabase):
    """
    清理 2: 删除 Elasticsearch 中的过时 CoT 日志 (无占位符)
    """
    try:
        retention_days = get_retention_days('cot_retention_days', 30)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()
        
        logger.info(f"正在清理 {retention_days} 天前 (即 {cutoff_iso}) 的 ES CoT 日志...")
        
        query = {
            "query": {
                "range": {
                    "@timestamp": { # 假设 CoTDatabase 使用 @timestamp
                        "lt": cutoff_iso
                    }
                }
            }
        }
        
        result = cot_db.delete_by_query(body=query) # CoTDatabase 已有此方法
        logger.info(f"ES CoT 清理完成: {result}")
        
    except Exception as e:
        logger.error(f"ES CoT 清理失败: {e}", exc_info=True)

def clean_audit_logs(audit_db: AuditLogger):
    """
    清理 3: [已修复] 删除审计数据库中的过时日志
    """
    try:
        # 读取 system.yaml 中已有的配置
        retention_days = get_retention_days('audit_log_retention_days', 30)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        logger.info(f"正在清理 {retention_days} 天前 (即 {cutoff_date}) 的审计日志...")
        
        # [已修复] 调用我们在 audit/logger.py 中添加的新方法
        result = audit_db.delete_older_than(cutoff_date)
        logger.info(f"审计日志清理完成: {result}")

    except Exception as e:
        logger.error(f"审计日志清理失败: {e}", exc_info=True)

def clean_local_snapshots(snapshot_mgr: SnapshotManager):
    """
    清理 4: 删除本地的过时快照 (.pkl) 文件 (无占位符)
    """
    try:
        retention_days = get_retention_days('snapshot_retention_days', 7)
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        snapshot_dir = snapshot_mgr.snapshot_dir
        
        logger.info(f"正在清理 {retention_days} 天前 (即 {datetime.fromtimestamp(cutoff_time)}) 的本地快照...")
        
        deleted_count = 0
        for filepath in glob.glob(os.path.join(snapshot_dir, "*.pkl")):
            try:
                # [安全检查] 确保我们只删除 .pkl 文件
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    logger.debug(f"已删除过时快照: {filepath}")
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"删除快照 {filepath} 失败: {e}")
                
        logger.info(f"本地快照清理完成。共删除 {deleted_count} 个文件。")

    except Exception as e:
        logger.error(f"本地快照清理失败: {e}", exc_info=True)


def run_all_cleanup_tasks():
    """
    [新] 主函数，供 Celery Worker [m-patek/market-knowledge/Market-Knowledge-main/Phoenix_project/worker.py] 导入和调用。
    """
    logger.info("--- 系统保洁机器人 (System Janitor) 任务启动 ---")
    
    # --- [新] 初始化所有依赖的客户端 ---
    # （这确保了 Janitor 拥有自己的、独立的连接）
    try:
        vector_store = PineconeVectorStore(config=system_config.get('ai', {}).get('vector_database', {}))
        cot_db = ElasticsearchCoTDatabase(config=system_config.get('ai', {}).get('cot_database', {}))
        
        # [修复] 审计日志也需要配置
        # (假设它使用与 CoT 相同的 ES，但索引不同)
        audit_config = system_config.get('audit_db', system_config.get('ai', {}).get('cot_database', {}))
        # (假设 audit/logger.py 也使用 host/port/index 配置)
        audit_config['index'] = "phoenix-audit-logs" 
        audit_db = AuditLogger(config=audit_config)
        
        snapshot_mgr = SnapshotManager()
    except Exception as e:
        logger.critical(f"Janitor 未能初始化所有客户端，任务中止: {e}", exc_info=True)
        return # 优雅地失败
    # --- 初始化结束 ---

    clean_pinecone_vectors(vector_store)
    clean_elasticsearch_cot(cot_db)
    clean_audit_logs(audit_db)
    clean_local_snapshots(snapshot_mgr)
    
    logger.info("--- 系统保洁机器人 (System Janitor) 任务完成 ---")

if __name__ == "__main__":
    # 允许脚本被手动执行 (例如 `python scripts/run_system_janitor.py`)
    run_all_cleanup_tasks()
