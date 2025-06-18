# --- Database Access Layer ---import secrets
import secrets
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict, Any, Generator, Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session


# --- SQLAlchemy Setup ---

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass

# --- Model Definitions ---

class Message(Base):
    """Represents a single message within a conversation."""
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    user_id = Column(Integer, nullable=True)
    user_name = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    conv_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(Integer, nullable=False)
    message_id = Column(Integer, nullable=True)

class Conversation(Base):
    """Represents a conversation session for a user."""
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    conv_id = Column(String(32), nullable=False, unique=True)
    timestamp = Column(Integer, nullable=False)

class Like(Base):
    """Represents feedback (like/dislike) on a message."""
    __tablename__ = 'likes'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    message_id = Column(Integer, nullable=False, index=True)
    feedback = Column(String(255), nullable=False)
    is_correct = Column(Boolean, nullable=False, default=True)


# --- Database Access Layer ---

class Database:
    """Handles all database operations for the application."""

    def __init__(self, db_path: str):
        """
        Initializes the database engine and creates tables if they don't exist.
        
        Note: For production environments, consider using a migration tool like Alembic
        instead of `create_all`.
        
        :param db_path: Filesystem path to the SQLite database file.
        """
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine)

    @contextmanager
    def _session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _get_current_ts() -> int:
        """Returns the current UTC timestamp as an integer."""
        return int(datetime.now(timezone.utc).timestamp())

    def create_conv_id(self, user_id: int) -> str:
        """
        Creates a new conversation ID for a user and saves it to the database.

        :param user_id: The ID of the user.
        :return: A new unique conversation ID.
        """
        conv_id = secrets.token_hex(nbytes=16)
        with self._session_scope() as session:
            new_conv = Conversation(
                user_id=user_id, 
                conv_id=conv_id, 
                timestamp=self._get_current_ts()
            )
            session.add(new_conv)
        return conv_id

    def get_current_conv_id(self, user_id: int) -> str:
        """
        Retrieves the most recent conversation ID for a user.
        If no conversation exists, a new one is created.

        :param user_id: The ID of the user.
        :return: The current conversation ID.
        """
        with self._session_scope() as session:
            latest_conv = session.query(Conversation.conv_id)\
                                 .filter(Conversation.user_id == user_id)\
                                 .order_by(Conversation.timestamp.desc())\
                                 .first()
            if latest_conv is None:
                return self.create_conv_id(user_id)
            return latest_conv[0]

    def _format_message_for_output(self, msg: Message, include_meta: bool) -> Dict[str, Any]:
        """Helper to format a message object for API response."""
        message_data = {
            "role": msg.role,
            "text": self._parse_content(msg.content)
        }
        if include_meta:
            message_data["timestamp"] = msg.timestamp
        return message_data

    def fetch_conversation(self, conv_id: str, include_meta: bool = False) -> List[Dict[str, Any]]:
        """
        Fetches all messages for a given conversation ID, ordered by timestamp.

        :param conv_id: The conversation ID.
        :param include_meta: If True, includes metadata like timestamp in the output.
        :return: A list of message dictionaries.
        """
        with self._session_scope() as session:
            messages = session.query(Message)\
                              .filter(Message.conv_id == conv_id)\
                              .order_by(Message.timestamp)\
                              .all()
            
            return [self._format_message_for_output(m, include_meta) for m in messages]

    def save_user_message(self, content: Any, conv_id: str, user_id: int, user_name: Optional[str] = None) -> None:
        """
        Saves a message from a user to the database.

        :param content: The message content (string or JSON-serializable object).
        :param conv_id: The conversation ID.
        :param user_id: The user's ID.
        :param user_name: The user's name (optional).
        """
        with self._session_scope() as session:
            new_message = Message(
                role="user",
                content=self._serialize_content(content),
                conv_id=conv_id,
                user_id=user_id,
                user_name=user_name,
                timestamp=self._get_current_ts()
            )
            session.add(new_message)

    def save_assistant_message(self, content: str, conv_id: str, message_id: int) -> None:
        """
        Saves a message from the assistant to the database.

        :param content: The message content.
        :param conv_id: The conversation ID.
        :param message_id: The external message ID.
        """
        with self._session_scope() as session:
            new_message = Message(
                role="assistant",
                content=content,
                conv_id=conv_id,
                timestamp=self._get_current_ts(),
                message_id=message_id,
            )
            session.add(new_message)

    def save_feedback(self, feedback: str, user_id: int, message_id: int, is_correct: bool = True) -> None:
        """
        Saves user feedback for a specific message.
        
        Note: The `is_correct` parameter defaults to True, matching original behavior.
        This could be extended to handle different types of feedback.

        :param feedback: The feedback text (e.g., 'like', 'dislike').
        :param user_id: The ID of the user giving feedback.
        :param message_id: The ID of the message receiving feedback.
        :param is_correct: A boolean indicating the feedback type.
        """
        with self._session_scope() as session:
            new_feedback = Like(
                user_id=user_id,
                message_id=message_id,
                feedback=feedback,
                is_correct=is_correct
            )
            session.add(new_feedback)

    def get_all_conv_ids(self) -> List[str]:
        """
        Retrieves all unique conversation IDs from the database.

        :return: A list of all conversation ID strings.
        """
        with self._session_scope() as session:
            stmt = select(Conversation.conv_id)
            return list(session.scalars(stmt).all())

    def _serialize_content(self, content: Any) -> str:
        """
        Serializes content to a string.
        
        :param content: The message content.
        :return: A serialized message content."""
        if isinstance(content, str):
            return content
        return json.dumps(content)

    def _parse_content(self, content: str) -> Any:
        """
        Parses content that might be a JSON string.
        If it's not valid JSON, returns the original string.
        
        :param content: The message content.
        :return: Parsed content 
        """
        try:
            parsed_content = json.loads(content)
            if isinstance(parsed_content, list) and all(isinstance(m, dict) for m in parsed_content):
                return parsed_content
            return content
        except json.JSONDecodeError:
            return content