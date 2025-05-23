## Project Overview

This is a **knowledge extraction and annotation project** focused on processing DJ comments from KEXP (Seattle's independent radio station). The primary goal is to create a comprehensive knowledge graph by annotating and extracting structured information from DJ commentary about music, artists, and events.

## Core Purpose and Domain

**Domain**: Music information extraction and knowledge graph construction  
**Data Source**: KEXP radio station DJ comments from their database  
**Goal**: Extract structured entities and relationships about artists, songs, albums, venues, events, and contextual information from unstructured DJ commentary

## Technology Stack

### Core Technologies

- **Python 3.x** - Main programming language
- **spaCy** - NLP processing and named entity recognition
- **Prodigy** - Data annotation platform (primary annotation tool)
- **SQLite** - Database interface for KEXP data
- **Sentence Transformers** - For semantic embeddings and similarity search
- **HNSWLib** - Approximate nearest neighbor search for annotation bootstrapping

### Key Dependencies

- **FastAPI & Uvicorn** - For web services
- **Pandas & NumPy** - Data manipulation
- **scikit-learn** - Machine learning utilities
- **Transformers & PyTorch** - Advanced NLP models
- **Peewee** - ORM for database operations

## Project Structure

### Source Code Organization

```
src/
├── kexp_processing_utils/          # Core processing utilities
│   ├── comment_parser.py           # Text segmentation and parsing
│   └── normalization.py           # Text normalization functions
└── scripts/                       # Data processing pipeline scripts
    ├── 00_extract_kexp_comments.py # Database extraction
    ├── 00b_segment_and_normalize_comments.py
    ├── 04_prelabel_for_prodigy.py  # Preprocessing for annotation
    └── create_gazetteers*.py       # Gazetteer creation
```

### Configuration & Data

```
config/
├── labels.txt                     # Annotation schema labels
└── gazetteers/                    # Entity lookup lists

data/
├── raw_kexp_data/                 # Raw database extracts
├── processed_examples/            # Processed segments
├── prodigy_exports/              # Annotation exports
├── models/                       # Trained models
└── annotated/                    # Final annotated datasets
```

## Annotation Schema & Entities

The project uses a sophisticated **dual-level annotation approach**:

### Phase A: Named Entity Recognition (NER)

- **LOCATION** - Geographical entities
- **DATE** - Temporal expressions

### Phase B: Span Categorization (SpanCat)

The project defines **26 distinct entity types** divided into:

#### `_SPAN` Labels (Contextual Information):

1. **ARTIST_BIO_SPAN** - Biographical information about artists
2. **ARTIST_LOC_ORGIN_SPAN** - Artist geographical origins
3. **ARTIST_ALIAS_SPAN** - Stage names and aliases
4. **NEW_RELEASE_SPAN** - Announcements of new music releases
5. **SOUND_DESCRIPTION_SPAN** - Musical style descriptions
6. **THEME_INSPO_MENTION_SPAN** - Lyrical themes and inspirations
7. **ARTIST_QUOTE_SPAN** - Attributed artist statements
8. **GROUP_COMP_SPAN** - Band member information
9. **COLLAB_MENTION_SPAN** - Artist collaborations
10. **INFLUENCE_MENTION_SPAN** - Musical influences
11. **RECORD_LABEL_SPAN** - Label information
12. **SHOW_DATE_SPAN** - Concert and event details
13. **SEE_MORE_SPAN** - Calls to action and URLs

#### `_TAG` Labels (Specific Entities):

14. **ARTIST_TAG** - Artist/band names
15. **ALBUM_TAG** - Album titles
16. **SONG_TAG** - Song titles
17. **RECORD_LABEL_TAG** - Record label names
18. **GENRE_TAG** - Musical genres
19. **ROLE_TAG** - Artist roles (producer, DJ, etc.)
20. **EVENT_TAG** - Events and festivals
21. **VENUE_TAG** - Performance venues
22. **INSTRUMENT_TAG** - Musical instruments
23. **STUDIO_TAG** - Recording studios
24. **LOC_TAG** - Locations
25. **DATE_TAG** - Dates and time expressions

## Data Processing Pipeline

### 1. Data Extraction (`00_extract_kexp_comments.py`)

- Connects to KEXP SQLite database via `KEXP_DB_PATH` environment variable
- Extracts DJ comments with associated metadata (songs, artists, shows, dates)
- Filters comments (minimum 20 characters, non-empty)
- Joins multiple database tables for comprehensive metadata

### 2. Text Processing (`comment_parser.py`)

- **Segmentation**: Splits comments into coherent segments using regex patterns
- **Normalization**: Standardizes text (handles em-dashes, whitespace, etc.)
- **Filtering**: Removes URL-only segments and empty content
- **Structuring**: Creates Prodigy-ready annotation tasks

### 3. Pre-labeling System

- **Metadata-based pre-labeling**: Uses database metadata to suggest entity spans
- **Gazetteer matching**: PhraseMatcher for known entities (artists, genres, roles)
- **Similarity-based bootstrapping**: Uses sentence transformers for finding similar segments

## Annotation Workflow

### Prodigy Integration

The project includes a sophisticated Prodigy recipe (`prelabel_ent_spans_manual.py`) that:

- Fetches data directly from the KEXP database
- Applies automated pre-labeling using multiple strategies
- Provides manual annotation interface for span categorization
- Supports annotation bootstrapping with similarity search
- Handles deduplication and quality control

### Quality Control Features

- Span boundary validation
- Consistency checking across annotation sessions
- Automated filtering of problematic segments
- Progress tracking and export capabilities

## Knowledge Graph Goals

The annotation schema is designed to support extraction of semantic relationships:

- **IS_ALIAS_OF**: Artist name relationships
- **HAS_MEMBER/MEMBER_OF**: Band composition
- **COLLABORATED_WITH**: Artist collaborations
- **INFLUENCED_BY/INFLUENCES**: Musical influences
- **ORIGINATES_FROM**: Artist geographical origins
- **PERFORMED_AT_EVENT**: Performance relationships

## Current Project State

**Phase**: Active annotation phase focusing on span categorization  
**Data Volume**: Processing thousands of DJ comments from KEXP database  
**Automation Level**: Highly automated pipeline with manual annotation verification  
**Output Format**: JSONL files with comprehensive metadata and span annotations

## Development Environment

**Setup Requirements**:

- Python virtual environment (`.venv/`)
- KEXP database access via `KEXP_DB_PATH` environment variable
- Prodigy license for annotation interface
- spaCy models for NLP processing

This project represents a sophisticated approach to extracting structured knowledge from informal, conversational text in the music domain, combining state-of-the-art NLP techniques with domain-specific annotation schemas to build a comprehensive music knowledge graph from radio DJ commentary.
