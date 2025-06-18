from pydantic import BaseModel, Field
from typing import List, Optional

# --- Input Schemas ---

class InputText(BaseModel):
    """Schema for text-based requests, allowing for batch processing."""
    question: List[str] = Field(
        ...,
        min_length=1,
        example=["What is the founding year of the University of Southampton?"]
    )

# --- Response Schemas ---

class TranscriptionResponse(BaseModel):
    """
    Schema for the transcription endpoint response.
    ASR models often return a list of transcribed segments.
    """
    transcription: str = Field(
        ..., 
        example="The employer of Neville A. Stanton is University of Southampton."
    )

class AnswerResponse(BaseModel):
    """
    Schema for the structured answer from the RAG pipeline.
    Fields are optional to handle cases where the LLM might not return
    a complete or parsable structure.
    """
    full_answer: Optional[List[str]] = Field(
        None,
        example="The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862."
    )
    short_answer: Optional[List[str]] = Field(
        None, 
        example="1862."
    )
    docs: Optional[List[List[str]]] = Field(
        None, 
        example=[["source_doc1.txt", "source_doc2.txt"]]
    )
