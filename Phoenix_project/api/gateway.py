from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from monitor.logging import logger
from controller.orchestrator import Orchestrator
from core.schemas.task_schema import Task
from ai.embedding_client import EmbeddingClient
from ai.retriever import Retriever

# --- Pydantic 模型 ---

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    task_id: Optional[str] = None

class QueryResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10
    tickers: Optional[List[str]] = None
    data_types: Optional[List[str]] = None

class RetrieveResponse(BaseModel):
    documents: List[Dict[str, Any]] # 序列化的 Document

# --- API 网关 ---

class APIGateway:
    """
    提供一个 FastAPI 接口，用于与 Phoenix 系统的不同部分进行交互。
    """
    def __init__(
        self,
        orchestrator: Orchestrator,
        embedding_client: EmbeddingClient,
        retriever: Retriever
    ):
        self.app = FastAPI(
            title="Phoenix Cognitive Architecture API",
            description="API gateway for interacting with the Phoenix system.",
            version="0.1.0"
        )
        self.orchestrator = orchestrator
        self.embedding_client = embedding_client
        self.retriever = retriever
        self._register_routes()
        logger.info("APIGateway initialized and routes registered.")

    def _register_routes(self):
        """
        在 FastAPI 应用上注册所有 API 路由。
        """
        
        @self.app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            """
            处理一个用户查询，触发完整的分析流程。
            """
            try:
                task = Task(
                    description=request.query,
                    user_id=request.user_id,
                    task_id=request.task_id
                )
                logger.info(f"Received query for task: {task.task_id}")
                
                # 异步处理任务
                result = await self.orchestrator.process_task(task)
                
                if "error" in result:
                    logger.warning(f"Task {task.task_id} failed: {result['error']}")
                    raise HTTPException(status_code=500, detail=result["error"])

                logger.info(f"Task {task.task_id} completed successfully.")
                return QueryResponse(
                    task_id=task.task_id,
                    status="completed",
                    result=result
                )
            except Exception as e:
                logger.error(f"Error processing query for task {request.task_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

        @self.app.post("/embed", response_model=EmbeddingResponse)
        async def serve_embedding(payload: EmbeddingRequest):
            """
            按需提供文本嵌入服务。
            """
            try:
                # 注意：EmbeddingClient 在其内部的CPU密集型任务
                # (使用 asyncio.to_thread) 中运行，以避免阻塞API事件循环。
                embeddings = await self.embedding_client.embed(payload.texts)
                return EmbeddingResponse(embeddings=embeddings)
            except Exception as e:
                logger.error(f"Error processing embedding request: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/retrieve", response_model=RetrieveResponse)
        async def serve_retrieval(payload: RetrieveRequest):
            """
            从多个数据源检索和重排文档。
            """
            try:
                documents = await self.retriever.retrieve(
                    query=payload.query,
                    top_k=payload.top_k,
                    tickers=payload.tickers,
                    data_types=payload.data_types
                )
                # 序列化 Document 对象
                serialized_docs = [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ]
                return RetrieveResponse(documents=serialized_docs)
            except Exception as e:
                logger.error(f"Error processing retrieval request: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.get("/health")
        def health_check():
            """
            简单的健康检查端点。
            """
            logger.debug("Health check endpoint hit.")
            return {"status": "ok"}

    def get_app(self) -> FastAPI:
        """
        返回 FastAPI 应用实例。
        """
        return self.app
