import os
import shutil
import tempfile
import logging

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

import gigaam
from src.hipporag.HippoRAG import HippoRAG
from schemas import InputText, TranscriptionResponse, AnswerResponse
from pydub import AudioSegment

# --------------------------------------------------------------------------- #
# === Application Setup ===================================================== #
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Minerva API",
    version="1.1.0",
    description="API for audio transcription and question answering using RAG.",
)

# --------------------------------------------------------------------------- #
# === Model Initialization ================================================== #
# --------------------------------------------------------------------------- #


class AudioProcessor:
    def convert_to_wav(self, input_file: str, output_file: str):
        try:
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_file, format="wav")
        except Exception as e:
            print(f"Error during conversion: {e}")
            raise


logger.info("Initializing models...")
try:
    asr_model = gigaam.load_model("v2_rnnt")
    audio_processor = AudioProcessor()

    hipporag = HippoRAG(
        save_dir=os.getenv('RAG_SAVE_DIR', 'outputs_tinkoff_short'),
        llm_name=os.getenv('LLM_MODEL_NAME', 'deepseek-chat'),
        llm_base_url=os.getenv('LLM_BASE_URL', 'https://api.deepseek.com'),
        llm_api_key=os.getenv('LLM_API_KEY'),
        embedding_name=os.getenv('EMBEDDING_MODEL_NAME', 'deepvk/USER-bge-m3'),
        embedding_base_url=os.getenv('EMBEDDING_BASE_URL', 'http://0.0.0.0:8888'),
        embedding_api_key=os.getenv('EMBEDDING_API_KEY')
    )
    logger.info("Models initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize models: {e}", exc_info=True)

# --------------------------------------------------------------------------- #
# === API Endpoints ========================================================= #
# --------------------------------------------------------------------------- #
@app.post(
    "/transcribe/",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file"
)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file, converts it to WAV format, transcribes it,
    and returns the transcription text.

    - **file**: The audio file to be transcribed (e.g., .ogg, .mp3, .wav).

    Returns the transcription as a list of strings.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or ".tmp")[-1]) as temp_input:
            shutil.copyfileobj(file.file, temp_input)
            temp_input.seek(0)

            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_output:

                audio_processor.convert_to_wav(temp_input.name, temp_output.name)
                transcription = asr_model.transcribe(temp_output.name)
        
        logger.info(f"Successfully transcribed file: {file.filename}")
        return TranscriptionResponse(transcription=transcription)

    except Exception as e:
        logger.error(f"Error processing audio file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

@app.post(
    "/request_processing/",
    response_model=AnswerResponse,
    summary="Process a question via RAG"
)
async def send_response(request_data: InputText):
    """
    Receives a question (or a list of questions), processes it using the
    HippoRAG pipeline, and returns the generated answer(s).

    - **request_data**: A JSON object containing the 'question' field.

    Returns a list of answers.
    """
    try:
        logger.info(f"Processing RAG request for {len(request_data.question)} question(s).")
        rag_results = await hipporag.rag_qa(queries=request_data.question)
        short_answers, full_answers, docs = [], [], []
        if not rag_results or not rag_results[0]:
            short_answers, full_answers, docs = [], [], []
        else:
            for el in rag_results[0]:
                full_answers.append(el.full_answer)
                short_answers.append(el.short_answer)
                docs.append([doc.split('#@$')[0] for doc in el.docs]) 

        logger.info(f"Generated {len(full_answers)} answer(s).")
        return AnswerResponse(full_answer=full_answers, short_answer=short_answers, docs=docs)
    
    except Exception as e:
        logger.error(f"Error during RAG processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process request: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9875)
