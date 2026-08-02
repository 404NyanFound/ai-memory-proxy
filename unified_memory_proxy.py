"""Unified Memory Proxy Server.

Executes owner-aware memory tools server-side and proxies OpenAI-compatible
chat-completion requests to a local llama.cpp-compatible endpoint.

Storage model:
- personal-{username}: User-private memories (database)
- ai-drive: Assistant's own workspace (D:\\AI Drive, files on disk)
- shared: Globally readable memories (database)
- imatrix: Calibration samples (D:\\AI Drive\\imatrix\\, files on disk, metadata in database)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import psycopg2
import psycopg2.pool
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from psycopg2.extras import Json, register_uuid

register_uuid()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("unified-memory-proxy")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = json.load(file)

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

# Storage paths
AI_DRIVE_PATH = Path(config.get("memory", {}).get("ai_drive_path", r"D:\AI Drive"))
IMATRIX_PATH = AI_DRIVE_PATH / "imatrix"
ARCHIVE_PATH = Path(config.get("memory", {}).get("archive_path", r"D:\AI Archives"))

ALLOWED_STORAGE_SPACES = {"personal", "ai-drive", "shared", "imatrix", "development"}
ALLOWED_SCOPES = {"personal", "shared", "both", "owned_shared", "development"}
ALLOWED_PREFERENCE_KEYS = {
    "name",
    "pronouns",
    "preferred_language",
    "communication_style",
    "technical_level",
    "gaming_preferences",
    "work_style",
}
ALLOWED_CATEGORIES = {
    "user_preferences",
    "programming_language",
    "project_context",
    "coding_style",
    "technical_skill",
    "work_habit",
    "communication_preference",
    "eureka_moment",
    "correction",
    "future_interest",
    "banter",
    "gaming",
    "hardware_specs",
    "software_stack",
    "tool_preferences",
    "file_organization",
    "running_topics",
    "cultural_reference",
    "creative_idea",
    "imatrix_sample",
    "development_note",
    "miscellaneous",
    # Technical domains
    "moe",
    "quantization",
    "cuda",
    "gpu",
    "training",
    "inference",
    "optimization",
    "debugging",
    "algorithms",
    "data_structures",
    "architecture",
    "performance",
    "security",
    "networking",
    "databases",
    "devops",
    "machine_learning",
    "deep_learning",
    "nlp",
    "computer_vision",
    "math",
    "science",
    "engineering",
    "technical",
    "code",
    "system",
    "model",
    "data",
    "research",
}

MEMORY_TOOL_NAMES = {
    "write_imatrix_sample",
    "save_to_memory",
    "recall_memory",
    "recall_conversations",
    "edit_memory",
    "share_memory",
    "unshare_memory",
    "set_preference",
    "get_preferences",
    "list_categories",
    "create_category",
    "list_archives",
    "restore_archive",
    "get_memory_history",
    "consolidate_memories",
    "disk_stats",
}

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,254}$")
WORD_PATTERN = re.compile(r"[\w@.+#-]+", re.UNICODE)

db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def init_db() -> None:
    global db_pool
    db_cfg = config["database"]
    db_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=db_cfg.get("host", "localhost"),
        port=db_cfg.get("port", 5432),
        dbname=db_cfg.get("name", "ai_memory"),
        user=db_cfg.get("user", "memory_user"),
        password=db_cfg.get("password", "memory_pass_123"),
    )
    logger.info("PostgreSQL connection pool initialised")
    prune_expired_turns()
    # Load categories from database
    _load_categories_from_db()


def _load_categories_from_db() -> None:
    """Load categories from database into ALLOWED_CATEGORIES set."""
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT name FROM categories")
            for (name,) in cursor.fetchall():
                ALLOWED_CATEGORIES.add(name)
        logger.info(f"Loaded {len(ALLOWED_CATEGORIES)} categories from database")
    finally:
        release_db(conn)


def get_db():
    if db_pool is None:
        raise RuntimeError("Database pool is not initialised")
    return db_pool.getconn()


def release_db(conn) -> None:
    if db_pool is not None:
        db_pool.putconn(conn)


def prune_expired_turns() -> None:
    """Delete expired session_turns records at startup."""
    try:
        conn = get_db()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM session_turns WHERE expires IS NOT NULL AND expires < NOW()"
                )
                deleted = cursor.rowcount
            conn.commit()
            if deleted:
                logger.info("Pruned %d expired session_turns", deleted)
        finally:
            release_db(conn)
    except psycopg2.Error:
        logger.exception("Failed to prune expired session_turns")


def validate_user_id(raw_user_id: str) -> str:
    user_id = raw_user_id.strip()
    if not USER_ID_PATTERN.fullmatch(user_id):
        raise HTTPException(status_code=400, detail="Invalid user identifier")
    return user_id


def resolve_user_id(request: Request, username: Optional[str] = None) -> str:
    header_user = request.headers.get("X-User-Id")
    if header_user and username and header_user != username:
        raise HTTPException(
            status_code=400,
            detail="URL username and X-User-Id do not match",
        )
    return validate_user_id(header_user or username or "anonymous")


def require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def validate_categories(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("categories must contain at least one category")
    categories = list(dict.fromkeys(value))
    invalid = [item for item in categories if item not in ALLOWED_CATEGORIES]
    if invalid:
        raise ValueError(f"Invalid categories: {', '.join(map(str, invalid))}")
    return categories


def validate_storage_space(value: Any, user_id: str) -> str:
    if not isinstance(value, str) or value not in ALLOWED_STORAGE_SPACES:
        raise ValueError(f"Invalid storage_space. Must be one of: {', '.join(sorted(ALLOWED_STORAGE_SPACES))}")
    return value


DEFAULT_TTL_DAYS = {
    "personal": 365,
    "shared": 730,
    "development": 180,
}

IMPORTANCE_TTL_MULTIPLIER = {
    "critical": None,  # No TTL (durable)
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}


def normalise_ttl(ttl_days: Any, train_imatrix: bool, importance: str = "medium", storage_space: str = "personal") -> Optional[datetime]:
    if train_imatrix:
        if ttl_days is not None:
            raise ValueError("Calibration memories cannot have a TTL")
        return None

    # If user explicitly set ttl_days, use it (no multiplier)
    if ttl_days is not None:
        if isinstance(ttl_days, bool) or not isinstance(ttl_days, int):
            raise ValueError("ttl_days must be an integer or null")
        if ttl_days < 1 or ttl_days > 3650:
            raise ValueError("ttl_days must be between 1 and 3650")
        return datetime.now(timezone.utc) + timedelta(days=ttl_days)

    # ttl_days is None: apply importance-based default
    # Critical = no TTL (durable)
    multiplier = IMPORTANCE_TTL_MULTIPLIER.get(importance, 1.0)
    if multiplier is None:
        return None

    base_ttl = DEFAULT_TTL_DAYS.get(storage_space, 365)
    effective_ttl = int(base_ttl * multiplier)
    return datetime.now(timezone.utc) + timedelta(days=effective_ttl)


def keyword_tokens(query: str) -> list[str]:
    tokens = [token.casefold() for token in WORD_PATTERN.findall(query)]
    return list(dict.fromkeys(token for token in tokens if len(token) >= 2))[:12]


def load_preferences(conn, user_id: str) -> dict[str, str]:
    """Load all preferences for a user (used by get_preferences tool)."""
    if user_id == "anonymous":
        return {}
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT key, value
            FROM preferences
            WHERE user_id = %s
            ORDER BY key
            """,
            (user_id,),
        )
        return {key: value for key, value in cursor.fetchall()}


def load_autoloaded_preferences(conn, user_id: str) -> dict[str, str]:
    """Load autoloaded preferences for system prompt injection."""
    if user_id == "anonymous":
        return {}
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT key, value
            FROM preferences
            WHERE user_id = %s AND key IN (
                'name', 'pronouns', 'preferred_language', 'technical_level',
                'communication_style', 'work_style', 'gaming_preferences'
            )
            ORDER BY key
            """,
            (user_id,),
        )
        return {key: value for key, value in cursor.fetchall()}


def build_system_message(conn, user_id: str, client_system_text: str = "") -> str:
    # Anonymous mode: NO preferences loaded, NO user identity
    if user_id == "anonymous":
        preference_block = ""
        user_block = "\n\n=== ANONYMOUS MODE ===\nYou are in anonymous mode. No user identity, preferences, or personal information is available. Do not attempt to load or save any personal data."
        return SYSTEM_PROMPT + preference_block + user_block

    # Non-anonymous: load autoloaded preferences (name, pronouns, communication
    # style, technical level, language, work style, gaming preferences)
    autoloaded = load_autoloaded_preferences(conn, user_id)
    if autoloaded:
        preference_lines = "\n".join(
            f"- {key}: {value}" for key, value in autoloaded.items()
        )
        preference_block = (
            "\n\n=== AUTOLOADED USER PREFERENCES ===\n" + preference_lines
        )
    else:
        preference_block = "\n\n=== AUTOLOADED USER PREFERENCES ===\n(no preferences set yet)"

    # Username from URL doubles as routing ID
    user_block = f"\n\n=== USER IDENTITY ===\nCurrent user: {user_id}"

    client_block = ""
    if client_system_text.strip():
        client_block = (
            "\n\n=== CLIENT-SUPPLIED SYSTEM CONTEXT ===\n"
            + client_system_text.strip()
        )
    return SYSTEM_PROMPT + preference_block + user_block + client_block


def write_file_safe(path: Path, content: str) -> None:
    """Write content to file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_file_safe(path: Path) -> Optional[str]:
    """Read file content, returning None if not found."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def add_file_reference(conn, file_path: str, memory_id: str) -> None:
    """Track that a memory references a file."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO file_references (file_path, memory_id)
            VALUES (%s, %s)
            ON CONFLICT (file_path, memory_id) DO NOTHING
            """,
            (file_path, memory_id),
        )
    conn.commit()


def remove_file_reference(conn, file_path: str, memory_id: str) -> None:
    """Remove a file reference when memory is deleted or edited."""
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM file_references WHERE file_path = %s AND memory_id = %s",
            (file_path, memory_id),
        )
    conn.commit()


def get_file_reference_count(conn, file_path: str) -> int:
    """Get how many memories reference this file."""
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM file_references WHERE file_path = %s",
            (file_path,),
        )
        return cursor.fetchone()[0]


def archive_file(file_path: str, archive_type: str = "file", project_name: str = None) -> Optional[str]:
    """Archive a file before deletion. Returns archive path or None if failed."""
    import shutil
    import zipfile
    from datetime import datetime

    src = Path(file_path)
    if not src.exists():
        return None

    ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if archive_type == "project" and project_name:
        # Zip the entire project directory
        project_dir = src.parent
        archive_name = f"{project_name}_{timestamp}.zip"
        archive_dest = ARCHIVE_PATH / archive_name
        with zipfile.ZipFile(archive_dest, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in project_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(project_dir.parent)
                    zf.write(file, arcname)
        # Record in database
        _record_archive(str(archive_dest), str(project_dir), archive_type, project_name, len(list(project_dir.rglob("*"))), archive_dest.stat().st_size)
        return str(archive_dest)
    else:
        # Single file backup
        archive_name = f"{src.stem}_{timestamp}{src.suffix}"
        archive_dest = ARCHIVE_PATH / archive_name
        shutil.copy2(src, archive_dest)
        _record_archive(str(archive_dest), file_path, archive_type, project_name, 1, archive_dest.stat().st_size)
        return str(archive_dest)


def _record_archive(archive_path: str, original_path: str, archive_type: str, project_name: str, file_count: int, size_bytes: int) -> None:
    """Record archive metadata in database."""
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO archives (archive_path, original_path, archive_type, project_name, file_count, size_bytes)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (archive_path) DO NOTHING
                """,
                (archive_path, original_path, archive_type, project_name, file_count, size_bytes),
            )
        conn.commit()
    finally:
        release_db(conn)


def cleanup_expired_memories() -> dict:
    """Delete expired memories, archive their files if reference count is 0."""
    import shutil

    conn = get_db()
    try:
        with conn.cursor() as cursor:
            # Find expired memories
            cursor.execute(
                """
                SELECT id, file_path, storage_space, categories
                FROM memories
                WHERE ttl_expires IS NOT NULL AND ttl_expires < NOW()
                """
            )
            expired = cursor.fetchall()

            deleted = 0
            archived = 0
            skipped = 0

            for memory_id, file_path, storage_space, categories in expired:
                # Never delete imatrix
                if storage_space == "imatrix":
                    skipped += 1
                    continue

                if file_path:
                    ref_count = get_file_reference_count(conn, file_path)
                    if ref_count > 1:
                        # Other memories still reference this file, just remove this reference
                        remove_file_reference(conn, file_path, str(memory_id))
                        skipped += 1
                        continue

                    # Archive before deleting
                    is_project = "project_context" in (categories or [])
                    project_name = None
                    if is_project:
                        # Extract project name from path
                        path_parts = Path(file_path).parts
                        if "Projects" in path_parts:
                            idx = list(path_parts).index("Projects")
                            if idx + 1 < len(path_parts):
                                project_name = path_parts[idx + 1]

                    archive_file(file_path, "project" if is_project else "file", project_name)

                    # Remove file from disk
                    try:
                        Path(file_path).unlink(missing_ok=True)
                    except Exception:
                        pass

                    remove_file_reference(conn, file_path, str(memory_id))
                    archived += 1

                # Delete the memory
                cursor.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
                deleted += 1

            conn.commit()

        return {
            "status": "cleanup_complete",
            "deleted": deleted,
            "archived": archived,
            "skipped": skipped,
        }
    finally:
        release_db(conn)


TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_to_memory",
            "description": (
                "Create a new persistent memory. For personal memories, use storage_space='personal'. "
                "For shared technical knowledge, use storage_space='shared'. "
                "For calibration samples, use storage_space='imatrix' and provide file_path. "
                "For workspace files, use storage_space='ai-drive' and provide file_path."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "storage_space": {
                        "type": "string",
                        "enum": sorted(ALLOWED_STORAGE_SPACES),
                        "description": "Storage location: personal (user-private), shared (global), ai-drive (workspace), imatrix (calibration)",
                    },
                    "context": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Memory content. For imatrix, a brief description of the sample; "
                            "write the full worked example to the file first via write_imatrix_sample. "
                            "For ai-drive, describe the file or include file content."
                        ),
                    },
                    "categories": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
                    },
                    "train_imatrix": {
                        "type": "boolean",
                        "description": "True only for complete calibration-quality technical samples.",
                    },
                    "ttl_days": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 3650,
                        "description": "Expiry in days, or null for durable memory. Calibration and shared memories must not expire.",
                    },
                    "file_path": {
                        "type": ["string", "null"],
                        "description": "Path to file on disk. Required for ai-drive and imatrix storage. Example: 'D:\\AI Drive\\imatrix\\sample.txt'",
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "How important/confident this memory is. Default: medium.",
                    },
                },
                "required": [
                    "storage_space",
                    "context",
                    "categories",
                    "train_imatrix",
                    "ttl_days",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_imatrix_sample",
            "description": (
                "Write a calibration sample file to the imatrix directory. You own the file "
                "content: pass the COMPLETE self-contained worked example in the standardized "
                "format. After writing, call save_to_memory with storage_space='imatrix' and the "
                "same file_path to register the database reference; the proxy will not overwrite "
                "the file."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path inside D:\\AI Drive\\imatrix\\, e.g. 'D:\\AI Drive\\imatrix\\algo_graph_shortest_path.txt'",
                    },
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Full worked example in the standardized iMatrix format ([TAG:type:topic] + Problem/Context/Working/Result/Pitfalls/General Principle).",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search memory using keywords and/or category filters. The server always "
                "enforces the current user and requested visibility scope."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused keywords or phrase; may be empty when categories are supplied.",
                    },
                    "categories": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
                    },
                    "scope": {
                        "type": "string",
                        "enum": sorted(ALLOWED_SCOPES),
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "include_history": {"type": "boolean"},
                    "date_from": {
                        "type": ["string", "null"],
                        "description": "ISO date string. Only return memories created or updated after this date.",
                    },
                    "date_to": {
                        "type": ["string", "null"],
                        "description": "ISO date string. Only return memories created or updated before this date.",
                    },
                    "importance": {
                        "type": ["string", "null"],
                        "enum": ["low", "medium", "high", "critical", None],
                        "description": "Filter by minimum importance level.",
                    },
                },
                "required": [
                    "query",
                    "categories",
                    "scope",
                    "top_k",
                    "include_history",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_memory",
            "description": "Replace an owned memory with a complete updated version.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid"},
                    "content": {"type": "string", "minLength": 1},
                    "categories": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
                    },
                    "train_imatrix": {"type": "boolean"},
                    "ttl_days": {"type": ["integer", "null"], "minimum": 1, "maximum": 3650},
                    "reason": {"type": "string", "minLength": 1},
                    "file_path": {
                        "type": ["string", "null"],
                        "description": "New file path if changing the linked file. Null to keep current.",
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "How important/confident this memory is.",
                    },
                },
                "required": [
                    "memory_id",
                    "content",
                    "categories",
                    "train_imatrix",
                    "ttl_days",
                    "reason",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "share_memory",
            "description": (
                "Make an owned private memory globally readable while retaining its "
                "original owner. Requires explicit user consent."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid"},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["memory_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unshare_memory",
            "description": "Make an owned shared memory private again.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid"},
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["memory_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_preference",
            "description": "Create or update an autoloaded preference for the current user. Not available in anonymous mode.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string", "enum": sorted(ALLOWED_PREFERENCE_KEYS)},
                    "value": {"type": "string", "minLength": 1},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_preferences",
            "description": "Return all autoloaded preferences for the current user. Not available in anonymous mode.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": "Return all available memory categories with their descriptions. Use this to discover what categories exist before saving or recalling memories.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_category",
            "description": "Create a new memory category with a description. Not available in anonymous mode. Use when no existing category fits the content you need to save.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "description": "Category name (lowercase, underscores only)",
                    },
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "description": "What this category covers",
                    },
                },
                "required": ["name", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_archives",
            "description": "Search archived files and projects by filename keywords. Returns archive metadata including path, size, and creation date. Use to find archived project files that can be restored.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for in archive filenames and paths",
                    },
                    "archive_type": {
                        "type": "string",
                        "enum": ["file", "project", "all"],
                        "description": "Filter by archive type: file, project, or all (default: all)",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of results to return (default: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restore_archive",
            "description": "Restore an archived file or project back to AI Drive and recreate memory entries. Not available in anonymous mode.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "archive_id": {
                        "type": "integer",
                        "description": "ID of the archive to restore (from list_archives)",
                    },
                    "restore_path": {
                        "type": "string",
                        "description": "Custom restore path (optional, defaults to original location)",
                    },
                },
                "required": ["archive_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_history",
            "description": "View the edit history of a specific memory. Shows all previous versions with timestamps and reasons. Use to understand how a memory evolved over time.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory_id": {"type": "string", "format": "uuid"},
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consolidate_memories",
            "description": "Merge two or more related memories into a single, comprehensive memory. The source memories are archived and deleted. Use when you have fragmented or overlapping memories about the same topic.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "minItems": 2,
                        "uniqueItems": True,
                        "items": {"type": "string", "format": "uuid"},
                        "description": "IDs of memories to merge (minimum 2)",
                    },
                    "merged_content": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The consolidated content combining all source memories",
                    },
                    "categories": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Importance of the consolidated memory",
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["memory_ids", "merged_content", "categories", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disk_stats",
            "description": "Check disk space for AI Drive workspace. Returns total, used, and free space. Only checks the AI Drive directory unless another path is specified.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to check (default: AI Drive workspace)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_conversations",
            "description": "Search past conversation turns with this user, including turns from other tabs/sessions. Returns matching turns with timestamps and their conversation id. Use to recall what was discussed earlier, in other tabs, or to reconstruct details from an older conversation.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Text to search for in past conversation turns (keywords, phrases, code, IDs)",
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Restrict search to a specific conversation (optional)",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of turns to return (default: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(conn, tool_name: str, arguments: dict[str, Any], user_id: str) -> dict[str, Any]:
    try:
        if tool_name == "write_imatrix_sample":
            file_path = require_nonempty_string(arguments["file_path"], "file_path")
            content = require_nonempty_string(arguments["content"], "content")
            file_path_obj = Path(file_path)
            if not str(file_path_obj).startswith(str(IMATRIX_PATH)):
                raise ValueError(f"imatrix files must be in {IMATRIX_PATH}")
            write_file_safe(file_path_obj, content)
            return {"status": "written", "file_path": str(file_path_obj)}

        if tool_name == "save_to_memory":
            storage_space = validate_storage_space(arguments["storage_space"], user_id)
            context = require_nonempty_string(arguments["context"], "context")
            categories = validate_categories(arguments["categories"])
            train_imatrix = bool(arguments["train_imatrix"])
            file_path = arguments.get("file_path")
            importance = arguments.get("importance", "medium")
            if importance not in ("low", "medium", "high", "critical"):
                importance = "medium"
            ttl_expires = normalise_ttl(arguments.get("ttl_days"), train_imatrix, importance, storage_space)

            # Validate storage space rules
            if storage_space == "personal" and user_id == "anonymous":
                raise ValueError("Anonymous users cannot save personal memories")
            if storage_space == "development" and user_id == "anonymous":
                raise ValueError("Anonymous users cannot save development memories")
            if storage_space == "development" and train_imatrix:
                raise ValueError("Development memories cannot be calibration samples")
            if storage_space == "shared" and train_imatrix:
                raise ValueError("Shared memories cannot be calibration samples")
            if storage_space == "imatrix" and not train_imatrix:
                raise ValueError("imatrix storage requires train_imatrix=true")
            if storage_space == "imatrix" and not file_path:
                raise ValueError("imatrix storage requires file_path")
            if storage_space == "ai-drive" and not file_path:
                raise ValueError("ai-drive storage requires file_path")

            # Write file to disk if file_path provided
            if file_path:
                file_path_obj = Path(file_path)
                # Validate file path is within allowed directories
                if storage_space == "imatrix":
                    if not str(file_path_obj).startswith(str(IMATRIX_PATH)):
                        raise ValueError(f"imatrix files must be in {IMATRIX_PATH}")
                    # Model owns the file: never overwrite an existing sample. The
                    # proxy only registers the reference. Fallback to context only
                    # if the model forgot to write the file first.
                    if not file_path_obj.exists():
                        write_file_safe(file_path_obj, context)
                elif storage_space == "ai-drive":
                    if not str(file_path_obj).startswith(str(AI_DRIVE_PATH)):
                        raise ValueError(f"ai-drive files must be in {AI_DRIVE_PATH}")
                    write_file_safe(file_path_obj, context)

            # Determine actual storage space for database
            # Store as 'personal' in DB; user_id column handles ownership
            db_storage_space = storage_space

            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO memories (
                        user_id, storage_space, visibility, context,
                        categories, train_imatrix, ttl_expires, file_path, importance
                    )
                    VALUES (%s, %s, 'private', %s, %s, %s, %s, %s, %s)
                    RETURNING id, created
                    """,
                    (
                        user_id,
                        db_storage_space,
                        context,
                        categories,
                        train_imatrix,
                        ttl_expires,
                        file_path,
                        importance,
                    ),
                )
                memory_id, created = cursor.fetchone()
            conn.commit()

            # Track file reference
            if file_path:
                add_file_reference(conn, file_path, str(memory_id))

            return {
                "status": "saved",
                "memory_id": str(memory_id),
                "owner_user_id": user_id,
                "storage_space": storage_space,
                "visibility": "private",
                "categories": categories,
                "train_imatrix": train_imatrix,
                "file_path": file_path,
                "importance": importance,
                "ttl_expires": ttl_expires.isoformat() if ttl_expires else None,
                "created": created.isoformat(),
            }

        if tool_name == "recall_memory":
            query = str(arguments.get("query", "")).strip()
            categories_raw = arguments.get("categories", [])
            if not isinstance(categories_raw, list):
                raise ValueError("categories must be an array")
            categories = list(dict.fromkeys(categories_raw))
            invalid_categories = [c for c in categories if c not in ALLOWED_CATEGORIES]
            if invalid_categories:
                raise ValueError(f"Invalid categories: {', '.join(invalid_categories)}")

            scope = arguments.get("scope", "both")
            if scope not in ALLOWED_SCOPES:
                raise ValueError("Invalid scope")
            top_k = max(1, min(int(arguments.get("top_k", 5)), 50))
            include_history = bool(arguments.get("include_history", False))
            date_from = arguments.get("date_from")
            date_to = arguments.get("date_to")
            importance_filter = arguments.get("importance")
            tokens = keyword_tokens(query)

            # Validate date parameters
            if date_from:
                try:
                    date_from = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                except ValueError:
                    raise ValueError("date_from must be a valid ISO date string")
            if date_to:
                try:
                    date_to = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                except ValueError:
                    raise ValueError("date_to must be a valid ISO date string")

            # Build storage space filter based on scope
            where_params: list[Any] = []
            if scope == "personal":
                if user_id == "anonymous":
                    return {"status": "found", "count": 0, "memories": []}
                storage_filter = "m.user_id = %s AND m.storage_space = 'personal' AND m.visibility = 'private'"
                where_params.append(user_id)
            elif scope == "shared":
                storage_filter = "(m.storage_space = 'shared' OR m.visibility = 'shared')"
            elif scope == "owned_shared":
                storage_filter = "m.user_id = %s AND m.visibility = 'shared'"
                where_params.append(user_id)
            elif scope == "development":
                # Development memories are always accessible to the current user
                if user_id == "anonymous":
                    return {"status": "found", "count": 0, "memories": []}
                storage_filter = "m.user_id = %s AND m.storage_space = 'development'"
                where_params.append(user_id)
            else:  # both
                if user_id == "anonymous":
                    storage_filter = "(m.storage_space = 'shared' OR m.visibility = 'shared')"
                else:
                    storage_filter = (
                        "(m.user_id = %s AND m.storage_space = 'personal' AND m.visibility = 'private') "
                        "OR (m.storage_space = 'shared' OR m.visibility = 'shared') "
                        "OR (m.user_id = %s AND m.storage_space = 'development')"
                    )
                    where_params.append(user_id)
                    where_params.append(user_id)

            conditions: list[str] = [storage_filter]

            # Date range filtering
            if date_from:
                conditions.append("m.updated >= %s")
                where_params.append(date_from)
            if date_to:
                conditions.append("m.updated <= %s")
                where_params.append(date_to)

            # Importance filtering
            if importance_filter:
                importance_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
                min_level = importance_order.get(importance_filter, 0)
                allowed_importance = [k for k, v in importance_order.items() if v >= min_level]
                conditions.append("m.importance = ANY(%s)")
                where_params.append(allowed_importance)

            if categories:
                conditions.append("m.categories && %s::text[]")
                where_params.append(categories)

            # BM25-style ranking with importance boost
            relevance_parts: list[str] = []
            relevance_params: list[Any] = []
            keyword_conditions: list[str] = []
            keyword_params: list[Any] = []

            # Importance boost: critical=4, high=3, medium=2, low=1
            importance_boost = "(CASE m.importance WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END)"
            relevance_parts.append(importance_boost)

            for token in tokens:
                pattern = f"%{token}%"
                keyword_conditions.append(
                    "(m.context ILIKE %s OR array_to_string(m.categories, ' ') ILIKE %s)"
                )
                keyword_params.extend([pattern, pattern])
                # Keyword match: context match = 3 points, category match = 1 point
                relevance_parts.append(
                    "(CASE WHEN m.context ILIKE %s THEN 3 ELSE 0 END + "
                    "CASE WHEN array_to_string(m.categories, ' ') ILIKE %s THEN 1 ELSE 0 END)"
                )
                relevance_params.extend([pattern, pattern])

            if keyword_conditions:
                conditions.append("(" + " OR ".join(keyword_conditions) + ")")
                where_params.extend(keyword_params)

            # Recency boost: more recent = higher score
            relevance_parts.append(
                "(EXTRACT(EPOCH FROM (NOW() - m.updated)) / 86400.0 * -0.1)"
            )

            relevance_sql = " + ".join(relevance_parts) if relevance_parts else "0"
            history_column = "m.edit_history" if include_history else "NULL::jsonb"
            sql = f"""
                SELECT
                    m.id,
                    m.user_id,
                    m.storage_space,
                    m.visibility,
                    m.context,
                    m.categories,
                    m.train_imatrix,
                    m.file_path,
                    m.created,
                    m.updated,
                    m.ttl_expires,
                    m.importance,
                    {history_column} AS edit_history,
                    ({relevance_sql}) AS relevance
                FROM memories AS m
                WHERE {' AND '.join(conditions)}
                  AND (m.ttl_expires IS NULL OR m.ttl_expires >= NOW())
                ORDER BY relevance DESC, m.updated DESC
                LIMIT %s
            """
            params = [*relevance_params, *where_params, top_k]

            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()

            memories = []
            for row in rows:
                item = {
                    "id": str(row[0]),
                    "owner_user_id": row[1],
                    "storage_space": row[2],
                    "visibility": row[3],
                    "context": row[4],
                    "categories": row[5],
                    "train_imatrix": row[6],
                    "file_path": row[7],
                    "created": row[8].isoformat(),
                    "updated": row[9].isoformat(),
                    "ttl_expires": row[10].isoformat() if row[10] else None,
                    "importance": row[11],
                    "relevance": float(row[13] or 0),
                }
                if include_history:
                    item["edit_history"] = row[12]

                # Read file content if file_path exists and context is short
                if row[7] and len(row[4] or "") < 200:
                    file_content = read_file_safe(Path(row[7]))
                    if file_content:
                        item["file_content"] = file_content[:1000]  # Preview

                memories.append(item)

            return {
                "status": "found",
                "count": len(memories),
                "current_user_id": user_id,
                "scope": scope,
                "query": query,
                "categories": categories,
                "memories": memories,
            }

        if tool_name == "edit_memory":
            memory_id = uuid.UUID(arguments["memory_id"])
            memory_id_str = str(memory_id)
            content = require_nonempty_string(arguments["content"], "content")
            categories = validate_categories(arguments["categories"])
            train_imatrix = bool(arguments["train_imatrix"])
            reason = require_nonempty_string(arguments["reason"], "reason")
            new_file_path = arguments.get("file_path")
            importance = arguments.get("importance")
            if importance and importance not in ("low", "medium", "high", "critical"):
                importance = None

            # Get current file_path and importance before update
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT file_path, importance FROM memories WHERE id = %s AND user_id = %s",
                    (memory_id_str, user_id),
                )
                old_row = cursor.fetchone()
            if not old_row:
                return {"status": "not_found_or_not_owned", "memory_id": str(memory_id)}
            old_file_path, old_importance = old_row

            # Use current importance if not provided
            effective_importance = importance or old_importance or "medium"
            ttl_expires = normalise_ttl(arguments.get("ttl_days"), train_imatrix, effective_importance)

            # Get current visibility to decide TTL handling
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT visibility FROM memories WHERE id = %s AND user_id = %s",
                    (memory_id_str, user_id),
                )
                vis_row = cursor.fetchone()
            # Shared memories must not have TTL
            final_ttl = None if (vis_row and vis_row[0] == 'shared') else ttl_expires

            with conn.cursor() as cursor:
                # Set reason for trigger to capture
                cursor.execute("SET LOCAL app.memory_edit_reason = %s", (reason,))
                cursor.execute(
                    """
                    UPDATE memories
                    SET context = %s,
                        categories = %s,
                        train_imatrix = %s,
                        file_path = COALESCE(%s, file_path),
                        importance = COALESCE(%s, importance),
                        ttl_expires = %s
                    WHERE id = %s AND user_id = %s
                    RETURNING id, storage_space, visibility, file_path, updated
                    """,
                    (content, categories, train_imatrix, new_file_path, importance, final_ttl, memory_id_str, user_id),
                )
                row = cursor.fetchone()
            conn.commit()
            if not row:
                return {"status": "not_found_or_not_owned", "memory_id": str(memory_id)}

            updated_file_path = row[3]

            # Handle file reference changes
            if new_file_path and new_file_path != old_file_path:
                # Remove old reference
                if old_file_path:
                    remove_file_reference(conn, old_file_path, str(memory_id))
                    # Archive old file if no more references
                    ref_count = get_file_reference_count(conn, old_file_path)
                    if ref_count == 0:
                        archive_file(old_file_path, "file")
                        try:
                            Path(old_file_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                # Add new reference
                add_file_reference(conn, new_file_path, str(memory_id))
            elif updated_file_path:
                # File exists and wasn't changed, just update content
                write_file_safe(Path(updated_file_path), content)

            return {
                "status": "edited",
                "memory_id": str(row[0]),
                "storage_space": row[1],
                "visibility": row[2],
                "file_path": updated_file_path,
                "updated": row[4].isoformat(),
                "reason": reason,
            }

        if tool_name == "share_memory":
            memory_id = str(uuid.UUID(arguments["memory_id"]))
            reason = require_nonempty_string(arguments["reason"], "reason")
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE memories
                    SET visibility = 'shared', ttl_expires = NULL
                    WHERE id = %s AND user_id = %s AND visibility = 'private'
                    RETURNING id, user_id, storage_space, updated
                    """,
                    (memory_id, user_id),
                )
                row = cursor.fetchone()
            conn.commit()
            if not row:
                return {"status": "not_found_not_owned_or_already_shared", "memory_id": str(memory_id)}
            return {
                "status": "shared",
                "memory_id": str(row[0]),
                "owner_user_id": row[1],
                "storage_space": row[2],
                "visibility": "shared",
                "updated": row[3].isoformat(),
                "reason": reason,
            }

        if tool_name == "unshare_memory":
            memory_id = str(uuid.UUID(arguments["memory_id"]))
            reason = require_nonempty_string(arguments["reason"], "reason")
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE memories
                    SET visibility = 'private'
                    WHERE id = %s AND user_id = %s AND visibility = 'shared'
                    RETURNING id, user_id, storage_space, updated
                    """,
                    (memory_id, user_id),
                )
                row = cursor.fetchone()
            conn.commit()
            if not row:
                return {"status": "not_found_not_owned_or_already_private", "memory_id": str(memory_id)}
            return {
                "status": "unshared",
                "memory_id": str(row[0]),
                "owner_user_id": row[1],
                "storage_space": row[2],
                "visibility": "private",
                "updated": row[3].isoformat(),
                "reason": reason,
            }

        if tool_name == "set_preference":
            if user_id == "anonymous":
                return {"status": "error", "error": "Cannot set preferences in anonymous mode"}
            key = arguments["key"]
            if key not in ALLOWED_PREFERENCE_KEYS:
                raise ValueError("Invalid preference key")
            value = require_nonempty_string(arguments["value"], "value")
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO preferences (user_id, key, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    RETURNING key, value, updated
                    """,
                    (user_id, key, value),
                )
                row = cursor.fetchone()
            conn.commit()
            return {
                "status": "set",
                "key": row[0],
                "value": row[1],
                "updated": row[2].isoformat(),
            }

        if tool_name == "get_preferences":
            if user_id == "anonymous":
                return {"status": "error", "error": "Cannot get preferences in anonymous mode"}
            return {
                "status": "found",
                "preferences": load_preferences(conn, user_id),
            }

        if tool_name == "list_categories":
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT name, description FROM categories ORDER BY name"
                )
                rows = cursor.fetchall()
            return {
                "status": "found",
                "count": len(rows),
                "categories": {name: desc for name, desc in rows},
            }

        if tool_name == "create_category":
            if user_id == "anonymous":
                return {"status": "error", "error": "Cannot create categories in anonymous mode"}
            name = require_nonempty_string(arguments["name"], "name").lower()
            description = require_nonempty_string(arguments["description"], "description")
            # Validate name format: lowercase, underscores only
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
                return {"status": "error", "error": "Category name must be lowercase, start with a letter, and use only letters, numbers, and underscores"}
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO categories (name, description, created_by)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description
                    RETURNING name, description, created_at
                    """,
                    (name, description, user_id),
                )
                row = cursor.fetchone()
            conn.commit()
            # Also add to in-memory set so validation works immediately
            ALLOWED_CATEGORIES.add(name)
            return {
                "status": "created",
                "name": row[0],
                "description": row[1],
                "created_by": user_id,
                "created_at": row[2].isoformat(),
            }

        if tool_name == "list_archives":
            query = require_nonempty_string(arguments["query"], "query").lower()
            archive_type = arguments.get("archive_type", "all")
            top_k = max(1, min(int(arguments.get("top_k", 10)), 50))

            # Build search conditions
            conditions = []
            params = []

            if archive_type and archive_type != "all":
                conditions.append("archive_type = %s")
                params.append(archive_type)

            # Search in original_path, project_name, and archive_path
            if query:
                conditions.append(
                    "(LOWER(original_path) ILIKE %s OR LOWER(coalesce(project_name, '')) ILIKE %s OR LOWER(archive_path) ILIKE %s)"
                )
                pattern = f"%{query}%"
                params.extend([pattern, pattern, pattern])

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            sql = f"""
                SELECT id, archive_path, original_path, archive_type, project_name,
                       file_count, size_bytes, created_at, restored_at
                FROM archives
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(top_k)

            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()

            archives = []
            for row in rows:
                archives.append({
                    "id": row[0],
                    "archive_path": row[1],
                    "original_path": row[2],
                    "archive_type": row[3],
                    "project_name": row[4],
                    "file_count": row[5],
                    "size_bytes": row[6],
                    "created_at": row[7].isoformat(),
                    "restored_at": row[8].isoformat() if row[8] else None,
                })

            return {
                "status": "found",
                "count": len(archives),
                "archives": archives,
            }

        if tool_name == "restore_archive":
            if user_id == "anonymous":
                return {"status": "error", "error": "Cannot restore archives in anonymous mode"}

            archive_id = int(arguments["archive_id"])
            restore_path = arguments.get("restore_path")

            # Get archive metadata
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, archive_path, original_path, archive_type, project_name, file_count
                    FROM archives WHERE id = %s
                    """,
                    (archive_id,),
                )
                row = cursor.fetchone()

            if not row:
                return {"status": "error", "error": "Archive not found"}

            archive_path = Path(row[1])
            original_path = Path(row[2])
            archive_type = row[3]
            project_name = row[4]

            if not archive_path.exists():
                return {"status": "error", "error": f"Archive file not found on disk: {archive_path}"}

            # Determine restore destination
            if restore_path:
                dest = Path(restore_path)
            else:
                dest = original_path

            try:
                import shutil
                import zipfile

                if archive_type == "project" and archive_path.suffix == ".zip":
                    # Extract zip to parent of original path
                    dest_parent = dest if dest.suffix == "" else dest.parent
                    dest_parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(archive_path, 'r') as zf:
                        zf.extractall(dest_parent)
                    restored_files = [str(f) for f in dest_parent.rglob("*") if f.is_file()]
                else:
                    # Single file restore
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(archive_path, dest)
                    restored_files = [str(dest)]

                # Mark as restored
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE archives SET restored_at = NOW() WHERE id = %s",
                        (archive_id,),
                    )
                conn.commit()

                return {
                    "status": "restored",
                    "archive_id": archive_id,
                    "archive_type": archive_type,
                    "restored_to": str(dest),
                    "files_restored": len(restored_files),
                    "files": restored_files[:20],  # Cap at 20 for response size
                }
            except Exception as exc:
                return {"status": "error", "error": f"Restore failed: {str(exc)}"}

        if tool_name == "get_memory_history":
            memory_id = str(uuid.UUID(arguments["memory_id"]))
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, context, categories, importance, edit_history, created, updated
                    FROM memories WHERE id = %s AND user_id = %s
                    """,
                    (memory_id, user_id),
                )
                row = cursor.fetchone()
            if not row:
                return {"status": "not_found_or_not_owned", "memory_id": str(memory_id)}

            history = row[4] or []
            return {
                "status": "found",
                "memory_id": str(row[0]),
                "current_context": row[1],
                "current_categories": row[2],
                "current_importance": row[3],
                "created": row[5].isoformat(),
                "updated": row[6].isoformat(),
                "history": history,
                "version_count": len(history),
            }

        if tool_name == "consolidate_memories":
            if user_id == "anonymous":
                return {"status": "error", "error": "Cannot consolidate memories in anonymous mode"}

            memory_ids = [str(uuid.UUID(mid)) for mid in arguments["memory_ids"]]
            merged_content = require_nonempty_string(arguments["merged_content"], "merged_content")
            categories = validate_categories(arguments["categories"])
            reason = require_nonempty_string(arguments["reason"], "reason")
            importance = arguments.get("importance", "medium")
            if importance not in ("low", "medium", "high", "critical"):
                importance = "medium"

            # Verify all memories exist and are owned by user
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, file_path, storage_space FROM memories WHERE id = ANY(%s) AND user_id = %s",
                    (memory_ids, user_id),
                )
                rows = cursor.fetchall()

            if len(rows) != len(memory_ids):
                return {"status": "error", "error": "One or more memories not found or not owned"}

            # Verify all memories are in the same storage space
            storage_spaces = {row[2] for row in rows}
            if len(storage_spaces) > 1:
                return {"status": "error", "error": f"Cannot consolidate memories from different storage spaces: {', '.join(sorted(storage_spaces))}"}
            target_space = storage_spaces.pop()

            # Create consolidated memory in the same storage space (durable, no TTL)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO memories (
                        user_id, storage_space, visibility, context,
                        categories, train_imatrix, ttl_expires, importance
                    )
                    VALUES (%s, %s, 'private', %s, %s, false, NULL, %s)
                    RETURNING id, created
                    """,
                    (user_id, target_space, merged_content, categories, importance),
                )
                new_id, created = cursor.fetchone()
            conn.commit()

            # Archive old memories (don't delete files yet, they may be shared)
            archived_count = 0
            for mid in memory_ids:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE memories
                        SET context = '[CONSOLIDATED into ' || %s::text || '] ' || context,
                            categories = ARRAY['miscellaneous'],
                            importance = 'low',
                            ttl_expires = NOW() + INTERVAL '30 days'
                        WHERE id = %s
                        """,
                        (str(new_id), mid),
                    )
                archived_count += 1
            conn.commit()

            return {
                "status": "consolidated",
                "new_memory_id": str(new_id),
                "created": created.isoformat(),
                "memories_consolidated": archived_count,
                "importance": importance,
            }

        if tool_name == "disk_stats":
            check_path = Path(arguments.get("path", str(AI_DRIVE_PATH)))

            # Security: only allow checking AI Drive by default, or explicitly allowed paths
            allowed_roots = [AI_DRIVE_PATH, ARCHIVE_PATH, IMATRIX_PATH]
            if not any(str(check_path).startswith(str(root)) for root in allowed_roots):
                # Allow if user explicitly provides a path
                if "path" not in arguments:
                    check_path = AI_DRIVE_PATH

            if not check_path.exists():
                return {"status": "error", "error": f"Path does not exist: {check_path}"}

            import shutil
            usage = shutil.disk_usage(check_path)

            # Calculate directory size
            try:
                total_size = sum(f.stat().st_size for f in check_path.rglob("*") if f.is_file())
                file_count = sum(1 for _ in check_path.rglob("*") if _.is_file())
                dir_count = sum(1 for _ in check_path.rglob("*") if _.is_dir())
            except (PermissionError, OSError):
                total_size = 0
                file_count = 0
                dir_count = 0

            return {
                "status": "ok",
                "path": str(check_path),
                "disk_total_gb": round(usage.total / (1024**3), 2),
                "disk_used_gb": round(usage.used / (1024**3), 2),
                "disk_free_gb": round(usage.free / (1024**3), 2),
                "disk_used_percent": round((usage.used / usage.total) * 100, 1),
                "directory_size_gb": round(total_size / (1024**3), 2),
                "file_count": file_count,
                "directory_count": dir_count,
            }

        if tool_name == "recall_conversations":
            if user_id == "anonymous":
                return {"status": "found", "count": 0, "conversations": []}

            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("query must be a non-empty string")
            tokens = keyword_tokens(query)
            if not tokens:
                return {"status": "found", "count": 0, "conversations": []}
            top_k = max(1, min(int(arguments.get("top_k", 10)), 50))
            conversation_id = arguments.get("conversation_id")
            if conversation_id is not None:
                conversation_id = str(conversation_id).strip()

            score_conditions: list[str] = []
            score_params: list[Any] = []
            for token in tokens:
                pattern = f"%{token}%"
                score_conditions.append(
                    "(CASE WHEN (user_message ILIKE %s OR assistant_message ILIKE %s) THEN 1 ELSE 0 END)"
                )
                score_params.extend([pattern, pattern])
            score_expr = " + ".join(score_conditions)

            params: list[Any] = [*score_params, user_id]
            inner_conditions = ["user_id = %s", "(expires IS NULL OR expires > NOW())"]
            if conversation_id:
                inner_conditions.append("conversation_id = %s")
                params.append(conversation_id)

            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT conversation_id, user_message, assistant_message, tools_used, created, score
                    FROM (
                        SELECT conversation_id, user_message, assistant_message, tools_used, created,
                               ({score_expr}) AS score
                        FROM session_turns
                        WHERE {' AND '.join(inner_conditions)}
                    ) AS t
                    WHERE t.score >= 1
                    ORDER BY t.score DESC, t.created DESC
                    LIMIT %s
                    """,
                    (*params, top_k),
                )
                rows = cursor.fetchall()

            conversations = [
                {
                    "conversation_id": row[0],
                    "created": row[4].isoformat(),
                    "user_message": row[1],
                    "assistant_message": row[2],
                    "tools_used": row[3] or [],
                    "score": row[5],
                }
                for row in rows
            ]
            return {"status": "found", "count": len(conversations), "conversations": conversations}

        return {"status": "error", "error": f"Unknown memory tool: {tool_name}"}

    except (KeyError, TypeError, ValueError) as exc:
        conn.rollback()
        return {"status": "invalid_arguments", "error": str(exc)}
    except psycopg2.Error as exc:
        conn.rollback()
        logger.exception("Database error while executing %s", tool_name)
        return {"status": "database_error", "error": exc.pgerror or str(exc)}


async def execute_memory_tools(
    conn,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    user_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    new_messages = list(messages)
    new_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        try:
            arguments = json.loads(function.get("arguments", "{}"))
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            result = execute_tool(conn, tool_name, arguments, user_id)
        except (json.JSONDecodeError, ValueError) as exc:
            result = {"status": "invalid_arguments", "error": str(exc)}

        logger.info("Memory tool %s called: user=%s session=%s", tool_name, user_id, conversation_id)
        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result, default=str),
            }
        )
    return new_messages


def merge_system_messages(messages: list[dict[str, Any]], system_text: str) -> list[dict[str, Any]]:
    client_system_parts = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system" and message.get("content")
    ]
    non_system_messages = [message for message in messages if message.get("role") != "system"]
    combined = system_text
    if client_system_parts:
        combined += "\n\n=== CLIENT-SUPPLIED SYSTEM CONTEXT ===\n" + "\n\n".join(client_system_parts)
    return [{"role": "system", "content": combined}, *non_system_messages]


def completion_response_with_external_calls(
    response_data: dict[str, Any],
    external_calls: list[dict[str, Any]],
) -> Response:
    filtered = json.loads(json.dumps(response_data))
    message = filtered["choices"][0]["message"]
    message["tool_calls"] = external_calls
    filtered["choices"][0]["finish_reason"] = "tool_calls"
    return Response(
        content=json.dumps(filtered),
        status_code=200,
        media_type="application/json",
    )


def stream_tool_calls_generator(response_data: dict[str, Any], tool_calls: list[dict[str, Any]]):
    message = response_data.get("choices", [{}])[0].get("message") or {}
    reasoning = message.get("reasoning_content") or ""
    usage = response_data.get("usage")

    async def gen():
        for index in range(0, len(reasoning), 32):
            chunk = reasoning[index : index + 32]
            payload = {
                "id": response_data.get("id"),
                "object": "chat.completion.chunk",
                "created": response_data.get("created"),
                "model": response_data.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            if usage:
                payload["usage"] = usage
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.005)

        for call_index, call in enumerate(tool_calls):
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments", "")

            header_payload = {
                "id": response_data.get("id"),
                "object": "chat.completion.chunk",
                "created": response_data.get("created"),
                "model": response_data.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": call_index,
                                    "id": call.get("id"),
                                    "type": call.get("type", "function"),
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            if usage:
                header_payload["usage"] = usage
            yield f"data: {json.dumps(header_payload)}\n\n"
            await asyncio.sleep(0.005)

            for arg_index in range(0, len(arguments), 32):
                arg_chunk = arguments[arg_index : arg_index + 32]
                arg_payload = {
                    "id": response_data.get("id"),
                    "object": "chat.completion.chunk",
                    "created": response_data.get("created"),
                    "model": response_data.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": call_index, "function": {"arguments": arg_chunk}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                if usage:
                    arg_payload["usage"] = usage
                yield f"data: {json.dumps(arg_payload)}\n\n"
                await asyncio.sleep(0.005)

        final_payload = {
            "id": response_data.get("id"),
            "object": "chat.completion.chunk",
            "created": response_data.get("created"),
            "model": response_data.get("model"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
        if usage:
            final_payload["usage"] = usage
        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return gen()


def response_with_external_calls(
    response_data: dict[str, Any],
    external_calls: list[dict[str, Any]],
    client_stream: bool,
) -> Response:
    if client_stream:
        return StreamingResponse(
            stream_tool_calls_generator(response_data, external_calls),
            status_code=200,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return completion_response_with_external_calls(response_data, external_calls)


def extract_user_message(messages: list[dict[str, Any]]) -> str:
    """Last real user utterance: skip system and tool-result messages."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        if message.get("tool_call_id"):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def extract_tools_used(messages: list[dict[str, Any]]) -> list[str]:
    """Distinct tool names called across the turn, in call order."""
    tools: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            if name and name not in tools:
                tools.append(name)
    return tools


def persist_session(
    user_id: str,
    conversation_id: str,
    messages: list[dict[str, Any]],
    response_data: dict[str, Any],
) -> None:
    """Store one compact turn: user text, assistant text, and tools used."""
    user_message = extract_user_message(messages)
    if not user_message:
        return
    choices = response_data.get("choices") or [{}]
    assistant_message = (choices[0].get("message") or {}).get("content") or ""
    tools_used = extract_tools_used(messages)
    ttl_days = int(config.get("memory", {}).get("session_ttl_days", 30))
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO session_turns (
                    user_id, conversation_id, user_message,
                    assistant_message, tools_used, expires
                )
                VALUES (%s, %s, %s, %s, %s, NOW() + %s * INTERVAL '1 day')
                """,
                (
                    user_id,
                    conversation_id,
                    user_message,
                    assistant_message,
                    tools_used,
                    ttl_days,
                ),
            )
        conn.commit()
    except psycopg2.Error:
        conn.rollback()
        logger.exception("Unable to persist session turn")
    finally:
        release_db(conn)


def build_response_data(
    meta: dict[str, Any],
    content: str,
    reasoning: str,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": meta.get("id"),
        "object": "chat.completion",
        "created": meta.get("created"),
        "model": meta.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content or None,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        result["usage"] = usage
    return result


async def tool_call_chunks(
    response_data: dict[str, Any], tool_calls: list[dict[str, Any]]
):
    usage = response_data.get("usage")
    for call_index, call in enumerate(tool_calls):
        function = call.get("function", {})
        name = function.get("name", "")
        arguments = function.get("arguments", "")

        header_payload = {
            "id": response_data.get("id"),
            "object": "chat.completion.chunk",
            "created": response_data.get("created"),
            "model": response_data.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": call_index,
                                "id": call.get("id"),
                                "type": call.get("type", "function"),
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        if usage:
            header_payload["usage"] = usage
        yield f"data: {json.dumps(header_payload)}\n\n"
        await asyncio.sleep(0.005)

        for arg_index in range(0, len(arguments), 32):
            arg_chunk = arguments[arg_index : arg_index + 32]
            arg_payload = {
                "id": response_data.get("id"),
                "object": "chat.completion.chunk",
                "created": response_data.get("created"),
                "model": response_data.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": call_index, "function": {"arguments": arg_chunk}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            if usage:
                arg_payload["usage"] = usage
            yield f"data: {json.dumps(arg_payload)}\n\n"
            await asyncio.sleep(0.005)

    final_payload = {
        "id": response_data.get("id"),
        "object": "chat.completion.chunk",
        "created": response_data.get("created"),
        "model": response_data.get("model"),
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    if usage:
        final_payload["usage"] = usage
    yield f"data: {json.dumps(final_payload)}\n\n"


def finish_chunk_sse(response_data: dict[str, Any], finish_reason: str, usage: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "id": response_data.get("id"),
        "object": "chat.completion.chunk",
        "created": response_data.get("created"),
        "model": response_data.get("model"),
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if usage:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


async def stream_proxy_loop(
    request_body: dict[str, Any],
    max_iterations: int,
    user_id: str,
    conversation_id: str,
):
    llama_host = config["llama"]["host"]
    llama_port = config["llama"]["external_port"]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=600, write=30, pool=30)
    ) as client:
        for iteration in range(max_iterations):
            async with client.stream(
                "POST",
                f"http://{llama_host}:{llama_port}/v1/chat/completions",
                json=request_body,
            ) as response:
                if response.status_code != 200:
                    raw = (await response.aread()).decode("utf-8", errors="replace")
                    logger.error("Upstream error %s: %s", response.status_code, raw[:500])
                    payload = json.dumps(
                        {
                            "error": {
                                "message": raw,
                                "type": "upstream_error",
                                "code": response.status_code,
                            }
                        }
                    )
                    yield f"data: {payload}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                meta = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "created": int(datetime.now().timestamp()),
                    "model": request_body.get("model", "llama"),
                }
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_acc: dict[int, dict[str, Any]] = {}
                finish_reason = "stop"
                upstream_usage: dict[str, Any] | None = None

                def make_chunk(delta: dict) -> dict[str, Any]:
                    return {
                        "id": meta["id"],
                        "object": "chat.completion.chunk",
                        "created": meta["created"],
                        "model": meta["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }

                try:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []

                        # Capture usage from top-level of final chunk (OpenAI streaming format)
                        if not upstream_usage and data.get("usage"):
                            upstream_usage = data["usage"]

                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                        if "tool_calls" in delta:
                            for call in delta["tool_calls"]:
                                index = call.get("index", 0)
                                entry = tool_acc.setdefault(
                                    index,
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if call.get("id"):
                                    entry["id"] = call["id"]
                                if call.get("type"):
                                    entry["type"] = call["type"]
                                function = call.get("function") or {}
                                if function.get("name"):
                                    entry["function"]["name"] = function["name"]
                                if function.get("arguments"):
                                    entry["function"]["arguments"] += function["arguments"]
                            continue

                        if delta.get("content") is not None:
                            content_parts.append(delta.get("content") or "")
                        if delta.get("reasoning_content") is not None:
                            reasoning_parts.append(delta.get("reasoning_content") or "")
                        yield f"data: {json.dumps(make_chunk(delta))}\n\n"
                except Exception:
                    logger.exception("Streaming round %d failed during read", iteration + 1)
                    payload = json.dumps(
                        {
                            "error": {
                                "message": "Proxy lost upstream connection during generation",
                                "type": "proxy_error",
                                "code": 500,
                            }
                        }
                    )
                    yield f"data: {payload}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                content = "".join(content_parts)
                reasoning = "".join(reasoning_parts)
                tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
                memory_calls = [
                    call
                    for call in tool_calls
                    if call.get("function", {}).get("name") in MEMORY_TOOL_NAMES
                ]
                external_calls = [
                    call
                    for call in tool_calls
                    if call.get("function", {}).get("name") not in MEMORY_TOOL_NAMES
                ]
                logger.info(
                    "Stream round %d: reasoning=%d chars, content=%d chars, tools=%s, finish=%s",
                    iteration + 1,
                    len(reasoning),
                    len(content),
                    [c.get("function", {}).get("name") for c in tool_calls],
                    finish_reason,
                )

                # The round is over. Per the reasoning-only policy, memory-tool
                # rounds carry no visible content; content only appears once tool
                # work is complete, so it streams live with no spill.
                if memory_calls:
                    try:
                        conn = get_db()
                        try:
                            request_body["messages"] = await execute_memory_tools(
                                conn,
                                request_body["messages"],
                                memory_calls,
                                user_id,
                                conversation_id,
                            )
                        finally:
                            release_db(conn)
                    except Exception:
                        logger.exception(
                            "Memory tool execution failed (round %d)", iteration + 1
                        )
                        payload = json.dumps(
                            {
                                "error": {
                                    "message": "Memory tool execution failed",
                                    "type": "proxy_error",
                                    "code": 500,
                                }
                            }
                        )
                        yield f"data: {payload}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # External calls must still reach the client.
                    if external_calls:
                        response_data = build_response_data(
                            meta, content, reasoning, external_calls, "tool_calls", upstream_usage
                        )
                        async for chunk in tool_call_chunks(response_data, external_calls):
                            yield chunk
                        yield "data: [DONE]\n\n"
                        persist_session(
                            user_id,
                            conversation_id,
                            request_body["messages"],
                            response_data,
                        )
                        return
                    continue

                # No server-side memory calls remain. Persist the session and finish.
                if external_calls:
                    response_data = build_response_data(
                        meta, content, reasoning, external_calls, "tool_calls", upstream_usage
                    )
                    async for chunk in tool_call_chunks(response_data, external_calls):
                        yield chunk
                else:
                    response_data = build_response_data(
                        meta, content, reasoning, None, finish_reason or "stop", upstream_usage
                    )
                    yield finish_chunk_sse(response_data, finish_reason or "stop", upstream_usage)
                yield "data: [DONE]\n\n"
                persist_session(
                    user_id,
                    conversation_id,
                    request_body["messages"],
                    response_data,
                )
                return

    payload = json.dumps({"error": "Maximum memory-tool iterations reached"})
    yield f"data: {payload}\n\n"
    yield "data: [DONE]\n\n"


async def proxy_chat_completions(request: Request, user_id: str):
    body = await request.json()
    client_stream = bool(body.get("stream"))
    original_messages = list(body.get("messages", []))
    probe = {
        k: body.get(k)
        for k in (
            "session_id",
            "conversation_id",
            "thread_id",
            "conversation",
            "title",
            "id",
        )
        if k in body
    }
    headers = dict(request.headers)
    hdr_probe = {
        k: v
        for k, v in headers.items()
        if any(
            token in k.lower()
            for token in ("session", "conversation", "thread", "id", "trace")
        )
    }
    logger.info(
        "Req probe user=%s top_keys=%s probe=%s n_messages=%d header_keys=%s hdr_probe=%s",
        user_id,
        sorted(body.keys()),
        probe,
        len(original_messages),
        sorted(headers.keys()),
        hdr_probe,
    )
    # Conversation identity: prefer OpenCode's per-tab x-session-id header;
    # fall back to a body session_id (legacy clients) then a fresh UUID.
    conversation_id = request.headers.get("x-session-id")
    if not conversation_id:
        try:
            conversation_id = str(uuid.UUID(str(body.get("session_id", uuid.uuid4()))))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="session_id must be a UUID"
            ) from exc

    conn = get_db()
    try:
        # Auto-set name preference for non-anonymous users
        if user_id != "anonymous":
            preferences = load_preferences(conn, user_id)
            if "name" not in preferences:
                # Auto-set name from username
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO preferences (user_id, key, value)
                        VALUES (%s, 'name', %s)
                        ON CONFLICT (user_id, key) DO NOTHING
                        """,
                        (user_id, user_id),
                    )
                conn.commit()
                preferences = load_preferences(conn, user_id)
        else:
            preferences = {}
    finally:
        release_db(conn)

    injected_prompt = build_system_message(conn, user_id)
    messages = merge_system_messages(original_messages, injected_prompt)

    client_tools = list(body.get("tools", []))
    all_tools = TOOL_SCHEMA + client_tools

    request_body = dict(body)
    request_body.update({"messages": messages, "tools": all_tools})
    request_body.pop("session_id", None)
    request_body.setdefault("stream_options", {"include_usage": True})

    llama_host = config["llama"]["host"]
    llama_port = config["llama"]["external_port"]
    max_iterations = int(config.get("memory", {}).get("max_tool_iterations", 10))

    if client_stream:
        request_body["stream"] = True
        return StreamingResponse(
            stream_proxy_loop(
                request_body, max_iterations, user_id, conversation_id
            ),
            status_code=200,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    request_body["stream"] = False

    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(max_iterations):
            response = await client.post(
                f"http://{llama_host}:{llama_port}/v1/chat/completions",
                json=request_body,
            )
            if response.status_code != 200:
                return Response(status_code=response.status_code, content=response.text)

            response_data = response.json()
            choices = response_data.get("choices", [])
            if not choices:
                return Response(
                    content=json.dumps(response_data),
                    status_code=200,
                    media_type="application/json",
                )

            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls") or []
            memory_calls = [
                call
                for call in tool_calls
                if call.get("function", {}).get("name") in MEMORY_TOOL_NAMES
            ]
            external_calls = [call for call in tool_calls if call not in memory_calls]

            if memory_calls:
                conn = get_db()
                try:
                    request_body["messages"] = await execute_memory_tools(
                        conn,
                        request_body["messages"],
                        memory_calls,
                        user_id,
                        conversation_id,
                    )
                finally:
                    release_db(conn)

                # Memory operations are complete. Forward unrelated calls to the client.
                if external_calls:
                    return response_with_external_calls(response_data, external_calls, client_stream)
                continue

            # No server-side memory calls remain. Persist the turn and return.
            persist_session(
                user_id,
                conversation_id,
                request_body["messages"],
                response_data,
            )

            # External tool calls must remain a normal JSON response for the client.
            if tool_calls:
                return response_with_external_calls(response_data, tool_calls, client_stream)

            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            finish_reason = choices[0].get("finish_reason", "stop")

            upstream_usage = response_data.get("usage")

            async def stream_chunks():
                for index in range(0, len(reasoning), 32):
                    chunk = reasoning[index : index + 32]
                    payload = {
                        "id": response_data.get("id"),
                        "object": "chat.completion.chunk",
                        "created": response_data.get("created"),
                        "model": response_data.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    if upstream_usage:
                        payload["usage"] = upstream_usage
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0.005)

                for index in range(0, len(content), 32):
                    chunk = content[index : index + 32]
                    payload = {
                        "id": response_data.get("id"),
                        "object": "chat.completion.chunk",
                        "created": response_data.get("created"),
                        "model": response_data.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    if upstream_usage:
                        payload["usage"] = upstream_usage
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0.005)

                final_payload = {
                    "id": response_data.get("id"),
                    "object": "chat.completion.chunk",
                    "created": response_data.get("created"),
                    "model": response_data.get("model"),
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }
                if upstream_usage:
                    final_payload["usage"] = upstream_usage
                yield f"data: {json.dumps(final_payload)}\n\n"
                yield "data: [DONE]\n\n"

            if client_stream:
                return StreamingResponse(
                    stream_chunks(),
                    status_code=200,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"},
                )

            return Response(
                content=json.dumps(response_data),
                status_code=200,
                media_type="application/json",
            )

    return Response(
        status_code=500,
        content=json.dumps({"error": "Maximum memory-tool iterations reached"}),
        media_type="application/json",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    import asyncio

    logger.info("Starting unified memory proxy")
    init_db()
    # Ensure directories exist
    IMATRIX_PATH.mkdir(parents=True, exist_ok=True)
    ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"AI Drive path: {AI_DRIVE_PATH}")
    logger.info(f"iMatrix path: {IMATRIX_PATH}")
    logger.info(f"Archive path: {ARCHIVE_PATH}")

    # Background cleanup task
    async def cleanup_task():
        while True:
            try:
                result = cleanup_expired_memories()
                if result["deleted"] > 0 or result["archived"] > 0:
                    logger.info(f"Cleanup: {result}")
            except Exception as exc:
                logger.exception("Cleanup task error: %s", exc)
            # Run every 24 hours
            await asyncio.sleep(86400)

    task = asyncio.create_task(cleanup_task())
    yield
    task.cancel()
    logger.info("Stopping unified memory proxy")
    if db_pool is not None:
        db_pool.closeall()


app = FastAPI(title="Unified Memory Proxy", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def proxy_chat_default(request: Request):
    user_id = resolve_user_id(request)
    return await proxy_chat_completions(request, user_id)


@app.post("/{username}/v1/chat/completions")
async def proxy_chat_for_user(username: str, request: Request):
    user_id = resolve_user_id(request, username)
    return await proxy_chat_completions(request, user_id)


@app.get("/health")
async def health():
    return {"status": "ok", "server": "unified-memory-proxy"}


@app.post("/admin/cleanup")
async def admin_cleanup():
    """Manually trigger expired memory cleanup and archiving."""
    try:
        result = cleanup_expired_memories()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/models")
async def models():
    llama_host = config["llama"]["host"]
    llama_port = config["llama"]["external_port"]
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"http://{llama_host}:{llama_port}/models")
    return Response(
        content=response.text,
        status_code=response.status_code,
        media_type="application/json",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.get("server", {}).get("host", "0.0.0.0"),
        port=config.get("server", {}).get("port", 8082),
    )
