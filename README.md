# Unified Memory Proxy

A FastAPI-based proxy that adds persistent memory capabilities to any local LLM running via llama.cpp (or any OpenAI-compatible endpoint).

## Overview

The Unified Memory Proxy intercepts chat completion requests, executes memory tool calls server-side, and loops the model for round-2+ content. This allows models to save, recall, and edit memories across sessions without the client needing to understand the memory system.

- **Python version:** 3.11.0
- **Framework:** FastAPI
- **Database:** PostgreSQL
- **Dependencies:** See `requirements.txt`

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure `config.json` (see Configuration section below).

3. Initialize the database:
   ```bash
   psql -U your_user -f schema.sql
   ```

4. Run the proxy:
   ```bash
   python unified_memory_proxy.py
   ```

5. Point your client to `http://localhost:8082/v1/chat/completions` (or `http://localhost:8082/{username}/v1/chat/completions` for user-specific storage).

## Architecture

### Request Flow

1. Client sends a chat completion request to the proxy
2. Proxy injects the system prompt and autoloaded preferences into the messages
3. Proxy forwards the request to the upstream LLM endpoint (llama.cpp or compatible)
4. If the model returns memory tool calls (e.g., `save_to_memory`), the proxy executes them server-side
5. The proxy feeds the tool results back to the model and loops until no more memory tools are called
6. The final response is streamed back to the client

### Storage Spaces

| Space | Description | Scope | TTL Default |
|-------|-------------|-------|-------------|
| `personal` | User-private memories | User-specific | 365 days |
| `shared` | Globally readable memories | Global | 730 days |
| `ai-drive` | Assistant workspace files | File system | N/A (file-backed) |
| `imatrix` | Calibration samples | File system | Never expires |
| `development` | Model learning notes | User-specific | 180 days |

### Tool Calls

The proxy supports 17 memory tools, executed server-side:

| Tool | Description |
|------|-------------|
| `save_to_memory` | Create a new persistent memory |
| `write_imatrix_sample` | Write a calibration sample file to imatrix directory |
| `recall_memory` | Search memory by keywords and/or categories |
| `recall_conversations` | Search past conversation turns |
| `edit_memory` | Update an owned memory |
| `share_memory` | Make a private memory globally readable |
| `unshare_memory` | Make a shared memory private again |
| `set_preference` | Create or update an autoloaded preference |
| `get_preferences` | Return all autoloaded preferences |
| `list_categories` | Return all available memory categories |
| `create_category` | Create a new memory category |
| `list_archives` | Search archived files and projects |
| `restore_archive` | Restore an archived file or project |
| `get_memory_history` | View the edit history of a memory |
| `consolidate_memories` | Merge multiple memories into one |
| `disk_stats` | Check disk space for AI Drive workspace |

### Configuration

`config.json` properties:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `server.port` | int | 8082 | Proxy server port |
| `server.host` | string | "0.0.0.0" | Proxy server host |
| `proxy.port` | int | 8082 | Proxy port (alias for server.port) |
| `proxy.host` | string | "0.0.0.0" | Proxy host (alias for server.host) |
| `llama.external_port` | int | 8080 | Upstream LLM endpoint port |
| `llama.host` | string | "127.0.0.1" | Upstream LLM endpoint host |
| `database.host` | string | "localhost" | PostgreSQL host |
| `database.port` | int | 5432 | PostgreSQL port |
| `database.name` | string | "ai_memory" | Database name |
| `database.user` | string | "memory_user" | Database user |
| `database.password` | string | "memory_pass_123" | Database password |
| `memory.ai_drive_path` | string | "D:\AI Drive" | AI Drive workspace path |
| `memory.archive_path` | string | "D:\AI Archives" | Archive storage path |
| `memory.ttl_days` | int | 30 | Default TTL for memories |
| `memory.session_ttl_days` | int | 30 | TTL for session turns |
| `memory.context_inject_top_k` | int | 3 | Number of top memories to inject |
| `memory.max_tool_iterations` | int | 10 | Max memory tool loop iterations |
| `eureka_detection.enabled` | bool | true | Enable eureka detection |
| `eureka_detection.signals` | array | ["user_satisfaction", "problem_solved", "non_obvious_solution"] | Eureka detection signals |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (anonymous) |
| POST | `/{username}/v1/chat/completions` | Chat completion (user-specific) |
| GET | `/health` | Health check |
| POST | `/admin/cleanup` | Manually trigger expired memory cleanup |
| GET | `/models` | Proxy upstream models endpoint |

## Database Schema

The database contains 6 tables:

- **memories** — Stores memory entries with TTL, categories, and file references
- **preferences** — Stores user preferences (name, pronouns, technical level, etc.)
- **file_references** — Tracks which memories reference which files
- **archives** — Metadata for archived files and projects
- **categories** — Pre-populated list of valid memory categories
- **session_turns** — Compact per-turn conversation records for recall

## Security

- User isolation via URL username or `X-User-Id` header
- Path validation for file operations (enforces AI Drive / imatrix directories)
- Storage space and category validation against allowlists
- Database connection pooling with psycopg2
- TTL-based automatic cleanup of expired memories

## License

Private project — not intended for public distribution without explicit permission.
