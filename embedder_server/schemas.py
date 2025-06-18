from pydantic import BaseModel
from typing import List, Optional

# --- Request/Response Models ---
class EmbeddingRequest(BaseModel):
    """
    Request model for embedding generation
    
    :param input: List of text strings to embed
    :param model: Model name (for OpenAI compatibility)
    :param user: Optional user identifier
    """
    input: List[str]
    model: str
    user: Optional[str] = None

class EmbeddingObject(BaseModel):
    """
    Single embedding result object
    
    :param object: Always "embedding"
    :param embedding: List of floating-point numbers representing the embedding
    :param index: Position of the input in the request
    """
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    """
    Response model for embedding generation
    
    :param object: Always "list"
    :param data: List of embedding objects
    :param model: Model name used (from request)
    :param usage: Token usage information (simulated for compatibility)
    """
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: dict
