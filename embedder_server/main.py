"""
FastAPI Server for Text Embeddings using Infinity
This server provides an OpenAI-compatible embedding API using the Infinity library.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import AsyncGenerator
import uvicorn
import os
import logging

from infinity_emb import AsyncEmbeddingEngine
from schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingObject

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage the application's lifecycle events
    
    :param app: FastAPI application instance
    :yield: None during application runtime
    """
    try:
        # Load configuration from environment
        model_name = os.getenv("EMBEDDING_MODEL", "deepvk/USER-bge-m3")
        device = os.getenv("DEVICE", "cuda")
        engine_type = os.getenv("ENGINE", "torch")
        
        logger.info(f"Initializing embedding engine with model: {model_name}")
        logger.info(f"Using device: {device}, engine: {engine_type}")
        
        # Initialize the embedding engine
        app.state.engine = AsyncEmbeddingEngine(
            model_name_or_path=model_name,
            engine=engine_type,
            device=device
        )
        await app.state.engine.astart()
        logger.info("Embedding engine started successfully")
        
        yield
        
    finally:
        # Cleanup resources on shutdown
        logger.info("Stopping embedding engine")
        if hasattr(app.state, 'engine'):
            await app.state.engine.astop()
            logger.info("Embedding engine stopped")

# --- Initialize FastAPI Application ---
app = FastAPI(
    title="Infinity Embeddings Server",
    description="OpenAI-compatible embedding API using Infinity library",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)

# --- API Endpoints ---
@app.post("/embeddings", 
          response_model=EmbeddingResponse,
          summary="Generate text embeddings",
          description="Generate embeddings for input texts using the configured model")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for input texts
    
    :param request: EmbeddingRequest containing input texts and model name
    :return: EmbeddingResponse with generated embeddings
    :raises HTTPException 503: If embedding engine is not ready
    :raises HTTPException 500: For internal processing errors
    """
    if not hasattr(app.state, 'engine') or app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding engine not initialized"
        )
    
    try:
        embeddings, usage = await app.state.engine.embed(request.input)
        
        data = [
            EmbeddingObject(embedding=embedding.tolist(), index=i)
            for i, embedding in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": usage,
                "total_tokens": 0
            }
        )
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding processing error: {str(e)}"
        ) from e


@app.get("/health", 
         summary="Server health check",
         description="Check server status and model information")
async def health_check() -> dict:
    """
    Health check endpoint
    
    :return: Dictionary with server status and model information
    """
    status = {
        "status": "ok",
        "model": "unknown",
        "device": "unknown",
        "ready": False
    }
    
    if hasattr(app.state, 'engine') and app.state.engine is not None:
        status.update({
            "model": app.state.engine.model,
            "device": app.state.engine.device,
            "ready": True
        })
    
    return status


if __name__ == "__main__":
    host = '0.0.0.0' if os.getenv('IS_DOCKER_RUN') else '127.0.0.1'
    uvicorn.run(app, host=host, port=8888)