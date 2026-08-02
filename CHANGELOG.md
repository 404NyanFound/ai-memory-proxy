# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Usage tokens not appearing in client responses (usage chunk dropped due to empty `choices` early return)

### Added
- Initial release of Unified Memory Proxy
- 17 memory tools for persistent memory management
- Server-side tool execution with model looping
- PostgreSQL backend with TTL-based cleanup
- iMatrix calibration sample support
- Session recall and conversation history
- File archiving and restoration
- User isolation via URL username or header
- FastAPI-based proxy with streaming support
- Comprehensive documentation and schema
