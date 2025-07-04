import os
import zipfile
from pathlib import Path
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
        save_dir=os.getenv('RAG_SAVE_DIR'),
        llm_name=os.getenv('LLM_MODEL_NAME'),
        llm_base_url=os.getenv('LLM_BASE_URL'),
        llm_api_key=os.getenv('LLM_API_KEY'),
        embedding_name=os.getenv('EMBEDDING_MODEL_NAME'),
        embedding_base_url=os.getenv('EMBEDDING_BASE_URL'),
        embedding_api_key=os.getenv('EMBEDDING_API_KEY', 'sk-custom_key'),
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

@app.get(
    "/get_database_name/",
    summary="Get database_name of RAG system"
)
async def get_database_name():
    """
    Return HippoRAG working directory
    
    :return work_dir: String, working directory of HippoRAG 
    """
    return {"message": hipporag.global_config.save_dir}

@app.post(
    "/index/",
    summary="Index files for RAG system"
)
async def index_files(file: UploadFile = File(...)):
    """
    Accepts a zip file, decompresses it, and indexes the contents using HippoRAG.
    
    :param file: 
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_zip_path = temp_file.name
        
        extract_dir = Path(file.filename).stem.split('/')[-1]
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        os.unlink(temp_zip_path)
        
        global hipporag
        if extract_dir != hipporag.global_config.save_dir:
            hipporag = HippoRAG(
                save_dir=extract_dir,
                llm_name=os.getenv('LLM_MODEL_NAME'),
                llm_base_url=os.getenv('LLM_BASE_URL'),
                llm_api_key=os.getenv('LLM_API_KEY'),
                embedding_name=os.getenv('EMBEDDING_MODEL_NAME'),
                embedding_base_url=os.getenv('EMBEDDING_BASE_URL'),
                embedding_api_key=os.getenv('EMBEDDING_API_KEY', 'sk-custom_key'),
            )
        
        await hipporag.index([os.path.join(extract_dir, extract_dir)])
        
        return {"message": "Файлы успешно проиндексированы"}
    except Exception as e:
        logger.error(f"Error indexing files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error indexing files: {e}")

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
    host = '0.0.0.0'
    uvicorn.run(app, host=host, port=9875)
