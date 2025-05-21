# Data Flow and Annotation Strategy for KEXP Comments

This document details the end-to-end data processing and annotation pipeline for KEXP DJ comments, aiming to produce high-quality datasets for Span Categorization (SpanCat).

## I. Overall Objectives

- Extract and prepare KEXP DJ comments for annotation.
- Annotate a rich set of SpanCat labels (`_SPAN` and `_TAG` types) to capture detailed information within the comments.
- Leverage custom Prodigy recipes and bootstrapping techniques for efficiency.

## II. Annotation Schema

- Refer to `docs/annotation_schema.md` for detailed definitions of all SpanCat labels.

## III. Data Processing and Annotation Pipeline

**The primary annotation workflow now revolves around a self-contained Prodigy recipe (`kexp.smart_prelabel_v2`) that handles database extraction, comment parsing (segmentation & normalization), and pre-labeling for SpanCat.**

Previous standalone scripts for extraction (`00_extract_db_comments.py`), segmentation (`01_segment_comments.py`), and filtering (`03_filter_link_only_segments.py`), along with their intermediate files (`01_extracted_db_comments.jsonl`, `02_normalized_segments.jsonl`, `03_filtered_normalized_segments.jsonl`), are **no longer part of the direct input chain for this Prodigy recipe**. They may be retained for other data analysis or export tasks.

**Step 0: (Integrated into Recipe) Database Extraction, Segmentation, and Normalization**

- **Action**: The custom Prodigy recipe (`kexp.smart_prelabel_v2`) now directly:
  1. Connects to the KEXP Database (via environment variable `KEXP_DB_PATH`).
  2. Fetches raw comments and associated metadata.
  3. Uses the `src.kexp_processing.comment_parser` module to segment each raw comment into smaller, manageable text units.
  4. Normalizes each segment.
  5. Filters out segments that are too short or become empty after normalization (logic within `comment_parser.py`).
- **Input for Recipe**: KEXP Database (via `KEXP_DB_PATH`).
- **Internal Stream**: The recipe generates a stream of normalized text segments, each with associated metadata from the parent comment (e.g., `db_artist_name`, `db_album_title`, `db_song_title`, `play_id`, etc.).

**Step 1: Pre-labeling and Annotation (SpanCat)**

- **Action**: This is the core annotation step using the custom Prodigy recipe. This recipe will:
  1.  Take the internally generated stream of normalized segments.
  2.  **Pre-apply SpanCat TAGS** (`ARTIST_NAME_TAG`, `ALBUM_TITLE_TAG`, `SONG_TITLE_TAG`, `RECORD_LABEL_NAME_TAG`) by matching metadata values (`db_artist_name`, etc.) against the segment text using spaCy's `PhraseMatcher`.
  3.  Present a stream for manual annotation of:
      - Correction/acceptance of pre-applied SpanCat TAGs.
      - Manual annotation of all other SpanCat `_TAG`s and `_SPAN`s as defined in the schema (`config/labels.txt`).
- **Custom Prodigy Recipe**: `prodigy_recipes.prelabel_ent_spans_manual.py::kexp.smart_prelabel_v2`
- **Prodigy Datasets (Iterative Process):**

  - `kexp_main_annotations`: The primary dataset where all reviewed SpanCat annotations are stored.
  - `kexp_ann_candidates_[label_type]`: Temporary datasets for candidates found via `prodigy-ann` (e.g., `kexp_ann_candidates_artist_bio_span`).
  - `kexp_lunr_candidates_[label_type]`: Temporary datasets for candidates found via `prodigy-lunr-plugin` (e.g., `kexp_lunr_candidates_event_info_span`).

- **Annotation Sub-Workflow (Iterative):**
  1.  **Initial Unified Annotation Session**:
      - **Recipe**: `kexp.smart_prelabel_v2` (the custom recipe)
      - **Command**: `python -m prodigy kexp.smart_prelabel_v2 kexp_main_annotations --labels-file config/labels.txt -F prodigy_recipes/prelabel_ent_spans_manual.py`
      - **Action**: Annotators review pre-applied SpanCat TAGs (`ARTIST_NAME_TAG`, etc.). They then manually add all other SpanCat `_SPAN`s and `_TAG`s. This builds the initial `kexp_main_annotations` dataset.
  2.  **Bootstrapping with `prodigy-lunr-plugin` (Focused Keyword Search for SpanCat labels)**:
      - **Recipe**: `lunr.spancat.search` (or similar `lunr` recipes adapted for this workflow)
      - **Input**: Raw segments can be fed again via a modified recipe or a dump if needed, or search within `kexp_main_annotations` for refinement.
      - **Output**: `kexp_lunr_candidates_[specific_label]` dataset.
      - **Action**: Use keywords to find segments likely to contain specific SpanCat labels (e.g., "live at" for `EVENT_INFO_SPAN`). Annotate these candidates.
  3.  **Bootstrapping with `prodigy-ann` (Focused Similarity Search for SpanCat labels)**:
      - **Recipe**: `ann.spancat.manual` (or similar `ann` recipes)
      - **Index Source**: `kexp_main_annotations` (using examples of already annotated specific SpanCat labels).
      - **Input for Annotation**: A stream of unannotated or partially annotated segments.
      - **Output**: `kexp_ann_candidates_[specific_label]` dataset.
      - **Action**: Find segments semantically similar to existing good examples of specific SpanCat labels. Annotate these candidates.
  4.  **Review and Merge**:
      - Periodically review annotations in `kexp_lunr_candidates_*` and `kexp_ann_candidates_*`.
      - Use `prodigy data-to-spacy` and `prodigy train` to evaluate model performance on subsets or use review recipes.
      - Merge high-quality, reviewed annotations from candidate datasets into `kexp_main_annotations`.

**Step 2: Final Exported Datasets**

- **Action**: Export the comprehensive, reviewed dataset.
- **Script**: `prodigy db-out`
- **Input**: `kexp_main_annotations` (Prodigy dataset)
- **Output**: `data/annotations/kexp_reviewed_spancat_annotations.jsonl`
  - _Format_: JSONL, each line contains the text and all accepted SpanCat spans (`_SPAN`s and `_TAG`s). This is the final dataset ready for training SpanCat models.

## IV. Custom Recipe (`kexp.smart_prelabel_v2`) Details

- **Input**: Directly connects to KEXP DB and processes raw comments using `src.kexp_processing.comment_parser`.
- **Internal Logic**:
  - Uses spaCy `PhraseMatcher` (case-insensitive) to find exact matches of `db_artist_name`, `db_album_title`, `db_song_title`, `db_record_label_name` from the task's `meta` and labels them as `ARTIST_NAME_TAG`, `ALBUM_TITLE_TAG`, `SONG_TITLE_TAG`, and `RECORD_LABEL_NAME_TAG` respectively.
  - The list of all possible SpanCat labels for the UI is loaded from the file specified by the `--labels-file` argument (defaulting to `config/labels.txt`).
- **Output Stream**: Tasks with pre-applied `spans` for metadata-derived SpanCat TAGs.
- **Interface**: `spans_manual` to allow review and addition of all SpanCat label types.

## V. Database Connection in Recipe

- The custom recipe `kexp.smart_prelabel_v2` now implements direct database connection, segmentation, and normalization.
- **Pros**: Always uses the freshest data, significantly reduces intermediate files and precursor script runs, making the annotation workflow more streamlined.
- **Cons**: Recipe is more complex (though DB logic is fairly contained); tight coupling with DB schema exists (but was already present in `00_extract_db_comments.py`).

This document provides a roadmap for the data flow and annotation strategy.
