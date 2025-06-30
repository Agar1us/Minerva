import os
import io
import asyncio
import traceback
import tempfile
from typing import Any, List, Optional, Dict

import httpx
import fireducks.pandas as pd

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder

import fire
from database import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Minerva:
    """
    The main class for the Minerva assistant bot.
    It encapsulates all the logic for handling messages, interacting with external APIs,
    and managing the bot's lifecycle.
    """

    # --------------------------------------------------------------------------- #
    # === Constants for messages and callbacks ================================== #
    # --------------------------------------------------------------------------- #
    DEFAULT_ANSWER_MODE = "full"
    SUPPORTED_MODES = {"full", "short"}
    CSV_REQUIRED_COLUMN = "question"

    GREETING_MESSAGE = "Привет! Меня зовут Minerva, как тебе помочь?"
    ABOUT_MESSAGE = "MINERVA - интеллектуальный ассистент службы поддержки"
    PROCESSING_MESSAGE = "💬"
    GENERAL_ERROR_MESSAGE = "Что-то пошло не так. Попробуйте еще раз позже."
    UNSUPPORTED_TYPE_MESSAGE = "Ошибка! Такой тип сообщений пока не поддерживается!"
    API_RESPONSE_ERROR_MESSAGE = "Не удалось получить корректный ответ от сервера."
    
    FEEDBACK_CALLBACK_PREFIX = "feedback:"
    LIKE_CALLBACK_DATA = f"{FEEDBACK_CALLBACK_PREFIX}like"
    DISLIKE_CALLBACK_DATA = f"{FEEDBACK_CALLBACK_PREFIX}dislike"
    ANSWER_MODE_CALLBACK_PREFIX = "ansmode:"
    FULL_MODE_CALLBACK = f"{ANSWER_MODE_CALLBACK_PREFIX}full"
    SHORT_MODE_CALLBACK = f"{ANSWER_MODE_CALLBACK_PREFIX}short"
    

    def __init__(self, bot_token: str, db_path: str, rag_url: str, transcription_url: str):
        """
        Initializes the bot, database, and dispatcher.

        :param bot_token: The token for the Telegram Bot API.
        :param db_path: The path to the SQLite database file.
        :param api_url: The URL for the query processing API.
        :param transcription_url: The URL for the query transcription API. 
        """
        logger.info("Инициализация бота")
        self.db = Database(db_path)
        self.rag_url = rag_url
        self.transcription_url = transcription_url
        
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()
        
        self._likes_kb = self._build_feedback_keyboard()
        self._answer_mode_kb = self._build_answer_mode_keyboard()
        self._register_handlers()

        self.user_answer_mod: Dict[int, str] = {}
        self.user_states: Dict[int, str] = {}
        logger.info("Бот успешно инициализирован")

    def _build_feedback_keyboard(self) -> InlineKeyboardBuilder:
        """Builds the inline keyboard for feedback."""
        builder = InlineKeyboardBuilder()
        builder.add(InlineKeyboardButton(text="👍", callback_data=self.LIKE_CALLBACK_DATA))
        builder.add(InlineKeyboardButton(text="👎", callback_data=self.DISLIKE_CALLBACK_DATA))
        return builder
    
    def _build_answer_mode_keyboard(self) -> InlineKeyboardBuilder:
        """Builds the inline keyboard for answer mode."""
        builder = InlineKeyboardBuilder()
        builder.add(
            InlineKeyboardButton(text="📜 Полный ответ",  callback_data=self.FULL_MODE_CALLBACK),
            InlineKeyboardButton(text="✂️ Краткий ответ", callback_data=self.SHORT_MODE_CALLBACK),
        )
        return builder

    def _register_handlers(self):
        """Registers all message and callback handlers for the dispatcher."""
        self.dp.message.register(self.start_command, Command("start"))
        self.dp.message.register(self.answer_mode_command, Command("answer_mode"))
        self.dp.message.register(self.indexing_command, Command("indexing"))
        
        self.dp.message.register(self._handle_text, F.text)
        self.dp.message.register(self._handle_voice, F.voice)
        self.dp.message.register(self._handle_document, F.document)
        
        self.dp.callback_query.register(self._handle_feedback_callback, F.data.startswith(self.FEEDBACK_CALLBACK_PREFIX))
        self.dp.callback_query.register(self._handle_answer_mode_callback, F.data.startswith(self.ANSWER_MODE_CALLBACK_PREFIX))

    async def start_polling(self):
        """Starts the bot's polling loop."""
        await self.dp.start_polling(self.bot)

    async def start_command(self, message: Message):
        """
        Handles the /start command. Creates a new conversation for the user.
        
        :param message: The incoming message object from aiogram.
        """
        self.db.create_conv_id(message.chat.id)
        await message.reply(self.GREETING_MESSAGE)
    
    async def answer_mode_command(self, message: Message):
        """Show the user keyboard to select answer mode."""
        markup = self._answer_mode_kb.as_markup()
        await message.answer("Выберите режим ответа:", reply_markup=markup)

    async def indexing_command(self, message: Message):
        """
        Processes the /indexing command - prompts the user to send 
        a ZIP file for indexing and informs them about the current database.
        
        :param message: Incoming message.
        """
        user_id = message.from_user.id if message.from_user else None
        if not user_id:
            return
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self.rag_url + 'get_database_name/')
                response.raise_for_status()
                data = response.json()
                db_name = data.get("message", "Неизвестно")
        except Exception as e:
            logger.error(f"Ошибка при получении имени базы данных: {e}")
            return "Не удалось получить имя базы данных"
        
        self.user_states[user_id] = "waiting_for_zip"

        await message.reply(
            text=f"Текущая база данных: {db_name}\n\n"
            "Пожалуйста, отправьте ZIP-файл с документами для индексации.\n"
            "Если имя файла совпадает с текущим именем базы данных, она будет обновлена.\n"
            "В противном случае будет создана новая база данных.",
            parse_mode=None
        )
        
    def _get_user_full_name(self, message: Message) -> Optional[str]:
        """
        Safely retrieves the user's full name or username.

        :param message: The message object.
        :return: User's full name, username, or None.
        """
        if not message.from_user:
            return None
        return message.from_user.full_name or message.from_user.username

    async def _handle_text(self, message: Message):
        """
        Handles an incoming text message by querying the API.
        
        :param message: The incoming message object.
        """
        if not message.text:
            return
        await self._process_content(message, message.text)
    
    async def _handle_voice(self, message: Message):
        """
        Handles an incoming voice message by transcribing it and then querying the API.

        :param message: The incoming message object.
        """
        user_id = message.from_user.id if message.from_user else None

        placeholder = await message.answer(f"🎤 Распознаю голосовое сообщение...")
        try:
            transcribed_text = await self._transcribe_voice(message)
            await placeholder.delete()
            await self._process_content(message, transcribed_text)
        except Exception as e:
            logger.error(f"Ошибка при обработке голосового сообщения для пользователя {user_id}: {e}")
            traceback.print_exc()
            await placeholder.edit_text(f"Не удалось распознать речь: {e}")

    async def _handle_document(self, message: Message):
        """
        Handles an incoming document (CSV, Excel, ZIP) for processing.

        :param message: The incoming message object.
        """
        user_id = message.from_user.id if message.from_user else None

        if self.user_states.get(user_id) == "waiting_for_zip":
            await self._handle_zip_for_indexing(message)
            self.user_states.pop(user_id, None)
            return

        content = await self._parse_document(message)
        if content is None:
            await message.answer(self.UNSUPPORTED_TYPE_MESSAGE)
            return

        placeholder = await message.answer(f"Обрабатываю файл...")
        try:
            logger.debug(f"Отправка запроса к API для пакетной обработки от пользователя {user_id}")
            full_ans, short_ans, docs = await self._query_api(content)
            df = pd.DataFrame({'Question': content, 'Short Answer': short_ans, 'Full Answer': full_ans, 'Documents': docs})
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', encoding='utf-8') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                await self.bot.send_document(
                    document=FSInputFile(tmp_file.name, filename="results.csv"), 
                    chat_id=message.chat.id
                )
            os.remove(tmp_file.name)
            await placeholder.delete()

        except Exception:
            logger.error(f"Ошибка при пакетной обработке документа для пользователя {user_id}: {e}")
            traceback.print_exc()
            await placeholder.edit_text(self.GENERAL_ERROR_MESSAGE)

    async def _process_content(self, message: Message, content: Any):
        """
        A generic function to process content, save it, query the API, and send a response.
        
        :param message: The original message to reply to.
        :param content: The content to be processed (text or list of texts).
        """
        user_id = message.from_user.id
        user_name = self._get_user_full_name(message)
        conv_id = self.db.get_current_conv_id(user_id)

        self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)
        placeholder = await message.answer(self.PROCESSING_MESSAGE)

        try:
            full_ans, short_ans, docs = await self._query_api([content])

            mode = self.user_answer_mod.get(user_id, self.DEFAULT_ANSWER_MODE)
            answer_text = full_ans[0] if mode == "full" else short_ans[0]
            docs_text = ", ".join(docs[0])
            final_answer = (
                f"{answer_text}\n"
                f"===========================\n"
                f"Документы:\n{docs_text}"
            )

            markup = self._likes_kb.as_markup()
            new_message = await placeholder.edit_text(final_answer, parse_mode=None, reply_markup=markup)

            self.db.save_assistant_message(
                content=final_answer,
                conv_id=conv_id,
                message_id=new_message.message_id,
            )
        except httpx.RequestError as exc:
            logger.error(f"Ошибка сети при запросе к API для пользователя {user_id}: {exc}")
            await placeholder.edit_text(f"Ошибка сети. Не могу связаться с сервером. {exc}")
        except Exception:
            logger.error(f"Общая ошибка при обработке контента для пользователя {user_id}")
            traceback.print_exc()
            await placeholder.edit_text(self.GENERAL_ERROR_MESSAGE)

    async def _handle_feedback_callback(self, callback: CallbackQuery):
        """
        Handles a feedback callback (like/dislike).

        :param callback: The callback query object.
        """
        if not callback.data: return
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.replace(self.FEEDBACK_CALLBACK_PREFIX, "")
        is_correct = 1 if feedback == 'dislike' else 0

        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id, is_correct=is_correct)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )
        await callback.answer(f"Спасибо за ваш отзыв: {feedback}!")

    async def _handle_answer_mode_callback(self, callback: CallbackQuery):
        """
        Handles a answer mode callback (full/short).
        
        :param callback: The callback query object.
        """
        if not callback.data:
            return

        mode = callback.data.replace(self.ANSWER_MODE_CALLBACK_PREFIX, "")

        if mode not in self.SUPPORTED_MODES:
            await callback.answer("Неизвестный режим.")
            return

        user_id = callback.from_user.id
        self.user_answer_mod[user_id] = mode

        text = "Полный" if mode=='full' else 'Краткий'
        await callback.message.edit_text(
            text='Успешно сменили режим на: ' + text,
            parse_mode=None,
            reply_markup=None
            )
        await callback.answer(f"✅ Режим ответа установлен на «{mode}»")

    async def _query_api(self, user_content: List[str]) -> List[str]:
        """
        Sends a request to the processing API.

        :param user_content: The user's message content (string or list of strings).
        :return: A list of answers from the model.
        :raises httpx.RequestError: If a network-related error occurs.
        :raises httpx.HTTPStatusError: If the server returns a 4xx or 5xx response.
        :raises KeyError: If the 'answer' key is not in the response JSON.
        """
        question = {'question': user_content}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.rag_url + 'request_processing/', json=question)
            response.raise_for_status()
            data = response.json()
            
            for k in ("full_answer", "short_answer", "docs"):
                if k not in data:
                    raise KeyError(self.API_RESPONSE_ERROR_MESSAGE)

            return data["full_answer"], data["short_answer"], data["docs"]

    async def _transcribe_voice(self, message: Message) -> str:
        """
        Downloads a voice message, converts it, and transcribes it.

        :param message: The message with the voice file.
        :return: The transcribed text.
        """
        if not message.voice:
            return ""

        user_id = message.from_user.id if message.from_user else None

        try:
            voice_file = await self.bot.get_file(message.voice.file_id)

            voice_ogg_buffer = io.BytesIO()
            await self.bot.download_file(voice_file.file_path, destination=voice_ogg_buffer)
            voice_ogg_buffer.seek(0)

            files = {
                'file': ('voice.ogg', voice_ogg_buffer, 'audio/ogg')
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.transcription_url, files=files)
                response.raise_for_status()
                data = response.json()
                return data.get("transcription", "")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка при транскрибации для пользователя {user_id}: {e.response.status_code} - {e.response.text}")
            return "Произошла ошибка при обработке вашего сообщения."
        except httpx.RequestError as e:
            logger.error(f"Ошибка сети при транскрибации для пользователя {user_id}: {e}")
            return "Сервис транскрибации временно недоступен."
        except Exception as e:
            return "Что-то пошло не так."

    async def _handle_zip_for_indexing(self, message: Message):
        """
        Processes the ZIP file for indexing by sending it to the server.
        
        :param message: A message with an attached ZIP file.
        """
            
        file_name = message.document.file_name
        if not file_name or not file_name.lower().endswith('.zip'):
            await message.reply("Пожалуйста, отправьте файл с расширением .zip")
            return
            
        placeholder = await message.reply("⏳ Отправка файла на сервер для индексации...")
        
        try:
            file_info = await self.bot.get_file(message.document.file_id)
            file_data = await self.bot.download_file(file_info.file_path)
            
            files = {'file': (file_name, file_data, 'application/zip')}
            
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(self.rag_url + 'index/', files=files)
                response.raise_for_status()
                result = response.json()
                await placeholder.edit_text(f"✅ {result.get('message', 'Индексация завершена успешно.')}")
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Ошибка сервера: {e.response.status_code}"
            try:
                error_detail = e.response.json().get('detail', '')
                if error_detail:
                    error_msg += f" - {error_detail}"
            except:
                pass
            logger.error(f"HTTP ошибка при индексации: {error_msg}")
            await placeholder.edit_text(f"❌ {error_msg}")
        except httpx.RequestError as e:
            logger.error(f"Ошибка сети при индексации: {e}")
            await placeholder.edit_text(f"❌ Ошибка сети: {e}")
        except Exception as e:
            logger.error(f"Ошибка при индексации: {e}")
            traceback.print_exc()
            await placeholder.edit_text(f"❌ Ошибка: {e}")

    async def _parse_document(self, message: Message) -> Optional[List[str]]:
        """
        Parses a document (CSV/XLSX) to extract a list of questions.

        :param message: The message with the document.
        :return: A list of strings or None if the format is unsupported.
        """
        if not message.document: return None

        user_id = message.from_user.id if message.from_user else None
        file_name = message.document.file_name or ""
        file_extension = file_name.split('.')[-1].lower()
        
        logger.debug(f"Парсинг документа для пользователя {user_id}: {file_name} (расширение: {file_extension})")

        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}") as tmp_file:
            file_info = await self.bot.get_file(message.document.file_id)
            await self.bot.download_file(file_info.file_path, tmp_file.name)
            
            try:
                if file_extension == "csv":
                    df = pd.read_csv(tmp_file.name)
                    if self.CSV_REQUIRED_COLUMN not in [el.lower() for el in df.columns]:
                        await message.reply(f"Ошибка: в CSV файле отсутствует обязательная колонка '{self.CSV_REQUIRED_COLUMN}'.")
                        return None
                    return df[self.CSV_REQUIRED_COLUMN].dropna().astype(str).to_list()
                
                elif file_extension in ["xls", "xlsx"]:
                    df = pd.read_excel(tmp_file.name)
                    return df.iloc[:, 0].dropna().astype(str).to_list()

            except Exception as e:
                logger.error(f"Ошибка при парсинге документа для пользователя {user_id}: {e}")
                await message.reply(f"Не удалось обработать файл: {e}")
                return None
        return None

def main():
    """
    Main entry point for the bot.
    Initializes and starts the Minerva bot.
    Configuration is loaded from environment variables.
    
    """
    bot_token = os.getenv("BOT_TOKEN")
    rag_url = os.getenv("RAG_URL")
    transcription_url = os.getenv("TRANSCRIPTION_URL")
    db_path = os.getenv('DB_PATH', 'minerva.db')
    if not bot_token:
        raise ValueError("BOT_TOKEN environment variable not set!")

    bot = Minerva(bot_token=bot_token, db_path=db_path, rag_url=rag_url, transcription_url=transcription_url)
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)