-- AI Memory System Schema
-- PostgreSQL database for persistent memory storage
-- Consolidated schema — represents final database state

-- ============================================================
-- DATABASE CREATION
-- ============================================================
CREATE DATABASE ai_memory;

\c ai_memory;

-- ============================================================
-- MEMORIES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    storage_space VARCHAR(50) NOT NULL CHECK (storage_space IN ('personal', 'ai-drive', 'shared', 'imatrix', 'development')),
    visibility VARCHAR(20) NOT NULL DEFAULT 'private' CHECK (visibility IN ('private', 'shared')),
    context TEXT NOT NULL,
    categories TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    train_imatrix BOOLEAN DEFAULT FALSE,
    importance TEXT DEFAULT 'medium' CHECK (importance IN ('low', 'medium', 'high', 'critical')),
    created TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ttl_expires TIMESTAMP WITH TIME ZONE,
    edit_history JSONB DEFAULT '[]'::jsonb,
    file_path TEXT,
    CONSTRAINT shared_memory_no_ttl_check CHECK (visibility <> 'shared' OR ttl_expires IS NULL)
);

-- ============================================================
-- PREFERENCES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value TEXT NOT NULL,
    created TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, key)
);

-- ============================================================
-- FILE REFERENCES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS file_references (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(file_path, memory_id)
);

-- ============================================================
-- ARCHIVES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS archives (
    id SERIAL PRIMARY KEY,
    archive_path TEXT NOT NULL UNIQUE,
    original_path TEXT NOT NULL,
    archive_type TEXT NOT NULL CHECK (archive_type IN ('file', 'project')),
    project_name TEXT,
    file_count INTEGER NOT NULL DEFAULT 1,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    restored_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================
-- CATEGORIES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT 'system',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pre-populate with default categories
INSERT INTO categories (name, description, created_by) VALUES
-- User & conversation
('user_preferences', 'User preferences, settings, and personal choices', 'system'),
('programming_language', 'Programming language features, syntax, and idioms', 'system'),
('project_context', 'Project goals, architecture, decisions, and status', 'system'),
('coding_style', 'Code formatting, naming conventions, and style preferences', 'system'),
('technical_skill', 'Technical skills, expertise levels, and learning goals', 'system'),
('work_habit', 'Work patterns, productivity habits, and routines', 'system'),
('communication_preference', 'How the user prefers to communicate and receive information', 'system'),
('eureka_moment', 'Breakthrough insights, discoveries, and aha moments', 'system'),
('correction', 'Corrections to previous mistakes or misunderstandings', 'system'),
('future_interest', 'Topics the user wants to explore or learn about later', 'system'),
('banter', 'Casual conversation, jokes, and personal rapport', 'system'),
('gaming', 'Gaming preferences, experiences, and recommendations', 'system'),
('hardware_specs', 'Hardware specifications, configurations, and benchmarks', 'system'),
('software_stack', 'Software tools, frameworks, and versions in use', 'system'),
('tool_preferences', 'Preferences for specific tools and workflows', 'system'),
('file_organization', 'File structure, naming conventions, and organization patterns', 'system'),
('running_topics', 'Ongoing discussions and threads across conversations', 'system'),
('cultural_reference', 'Cultural context, references, and shared understanding', 'system'),
('creative_idea', 'Creative concepts, designs, and brainstorming results', 'system'),
('imatrix_sample', 'iMatrix calibration sample metadata', 'system'),
('development_note', 'Model development notes and learning insights', 'system'),
('miscellaneous', 'Content that does not fit neatly into other categories', 'system'),
-- Technical domains
('moe', 'Mixture of Experts architecture, routing, and expert specialization', 'system'),
('quantization', 'Model quantization techniques (GPTQ, AWQ, GGUF, bitsandbytes)', 'system'),
('cuda', 'CUDA programming, kernels, memory management, and optimization', 'system'),
('gpu', 'GPU hardware, drivers, VRAM management, and multi-GPU setups', 'system'),
('training', 'Model training, fine-tuning, LoRA, and optimization techniques', 'system'),
('inference', 'Model inference, serving, and deployment', 'system'),
('optimization', 'Performance optimization, profiling, and tuning', 'system'),
('debugging', 'Debugging techniques, error analysis, and troubleshooting', 'system'),
('algorithms', 'Algorithm design, analysis, and implementation', 'system'),
('data_structures', 'Data structure selection, implementation, and complexity', 'system'),
('architecture', 'System architecture, design patterns, and component design', 'system'),
('performance', 'Performance analysis, benchmarking, and optimization', 'system'),
('security', 'Security practices, vulnerabilities, and hardening', 'system'),
('networking', 'Network protocols, configuration, and troubleshooting', 'system'),
('databases', 'Database design, queries, indexing, and optimization', 'system'),
('devops', 'CI/CD, containerization, deployment, and infrastructure', 'system'),
('machine_learning', 'Machine learning concepts, models, and techniques', 'system'),
('deep_learning', 'Deep learning architectures, training, and optimization', 'system'),
('nlp', 'Natural language processing, tokenization, and text analysis', 'system'),
('computer_vision', 'Image processing, object detection, and visual AI', 'system'),
('math', 'Mathematical concepts, proofs, and calculations', 'system'),
('science', 'Scientific concepts, research, and experimentation', 'system'),
('engineering', 'Engineering principles, design, and problem-solving', 'system'),
('technical', 'General technical knowledge and best practices', 'system'),
('code', 'Code patterns, snippets, and implementation examples', 'system'),
('system', 'System-level programming, OS concepts, and low-level details', 'system'),
('model', 'Model architecture, behavior, and characteristics', 'system'),
('data', 'Data processing, analysis, and management', 'system'),
('research', 'Research methodology, papers, and findings', 'system')
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- SESSION TURNS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS session_turns (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id varchar(255) NOT NULL,
    conversation_id varchar(255) NOT NULL,
    user_message text NOT NULL,
    assistant_message text,
    tools_used text[] NOT NULL DEFAULT '{}',
    created timestamptz NOT NULL DEFAULT now(),
    expires timestamptz
);

-- ============================================================
-- INDEXES
-- ============================================================
-- Memories
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_storage_space ON memories(storage_space);
CREATE INDEX IF NOT EXISTS idx_memories_categories ON memories USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_memories_train_imatrix ON memories(train_imatrix);
CREATE INDEX IF NOT EXISTS idx_memories_ttl ON memories(ttl_expires);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_owner_space_visibility ON memories(user_id, storage_space, visibility);
CREATE INDEX IF NOT EXISTS idx_memories_shared ON memories(visibility) WHERE visibility = 'shared';
CREATE INDEX IF NOT EXISTS idx_memories_development ON memories(storage_space) WHERE storage_space = 'development';
CREATE INDEX IF NOT EXISTS idx_memories_imatrix ON memories(storage_space) WHERE storage_space = 'imatrix';
CREATE INDEX IF NOT EXISTS idx_memories_file_path ON memories(file_path) WHERE file_path IS NOT NULL;

-- File references
CREATE INDEX IF NOT EXISTS idx_file_references_path ON file_references(file_path);
CREATE INDEX IF NOT EXISTS idx_file_references_memory ON file_references(memory_id);

-- Archives
CREATE INDEX IF NOT EXISTS idx_archives_original ON archives(original_path);
CREATE INDEX IF NOT EXISTS idx_archives_project ON archives(project_name);
CREATE INDEX IF NOT EXISTS idx_archives_name ON archives USING GIN(to_tsvector('english', coalesce(project_name, '') || ' ' || original_path));

-- Categories
CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);

-- Session turns
CREATE INDEX IF NOT EXISTS idx_turns_user_created ON session_turns (user_id, created DESC);
CREATE INDEX IF NOT EXISTS idx_turns_conv ON session_turns (user_id, conversation_id, created);
CREATE INDEX IF NOT EXISTS idx_turns_search ON session_turns USING GIN (
    to_tsvector('english', user_message || ' ' || COALESCE(assistant_message, ''))
);
CREATE INDEX IF NOT EXISTS idx_turns_expires ON session_turns (expires) WHERE expires IS NOT NULL;

-- ============================================================
-- TRIGGERS
-- ============================================================
-- Auto-update timestamp on memory edit
CREATE OR REPLACE FUNCTION set_updated_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER memories_set_updated
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_timestamp();

-- Auto-append edit history on memory edit
CREATE OR REPLACE FUNCTION append_memory_history()
RETURNS TRIGGER AS $$
DECLARE
    reason_text TEXT;
BEGIN
    reason_text := current_setting('app.memory_edit_reason', true);

    IF OLD.context IS DISTINCT FROM NEW.context
       OR OLD.categories IS DISTINCT FROM NEW.categories
       OR OLD.train_imatrix IS DISTINCT FROM NEW.train_imatrix
       OR OLD.visibility IS DISTINCT FROM NEW.visibility
       OR OLD.storage_space IS DISTINCT FROM NEW.storage_space
       OR OLD.file_path IS DISTINCT FROM NEW.file_path
       OR OLD.importance IS DISTINCT FROM NEW.importance THEN
        NEW.edit_history = COALESCE(OLD.edit_history, '[]'::jsonb) ||
            jsonb_build_array(jsonb_build_object(
                'edited_at', NOW(),
                'reason', reason_text,
                'previous_context', OLD.context,
                'previous_categories', OLD.categories,
                'previous_train_imatrix', OLD.train_imatrix,
                'previous_visibility', OLD.visibility,
                'previous_storage_space', OLD.storage_space,
                'previous_file_path', OLD.file_path,
                'previous_importance', OLD.importance
            ));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER memories_append_history
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION append_memory_history();
