import prodigy
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string, set_hashes
from prodigy.models.matcher import PatternMatcher as ProdigyPatternMatcher
import spacy
from typing import Optional, List, Dict, Any, Iterable, Iterator
import os
import sqlite3
from dotenv import load_dotenv
import logging
from spacy.matcher import PhraseMatcher
import pathlib
import re

# Import the new parser function
from src.kexp_processing.comment_parser import parse_comment_to_prodigy_tasks

# Configure logging - Prodigy uses its own logging, but this can be for DB part
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables for KEXP_DB_PATH
load_dotenv()

# Removed normalize_raw_db_comment_text function
# The raw text from DB will be passed directly to the comment_parser module.


def get_db_path_for_recipe():
    """Gets the database path from the KEXP_DB_PATH environment variable for the recipe."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error(
            "KEXP_DB_PATH environment variable not set for the recipe.")
        # In a recipe, raising an error that Prodigy can catch is good.
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def fetch_raw_comments_from_db(db_path: str, limit: Optional[int] = 2000, random_order: bool = False) -> Iterator[Dict[str, Any]]:
    """
    Fetches plays with comments from KEXP DB.
    Yields raw records: {"text": <raw_comment_from_db>, "meta": <db_metadata>}.
    The `text` is now the comment text as-is from the DB.
    The `comment_parser` module will handle all segmentation and segment normalization.
    MODIFIED: Fetches all data into a list first to avoid SQLite threading issues.
    """
    conn = None
    all_records = []  # Store all records here first
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = """
        SELECT
            p.id AS play_id, p.comment AS comment_text, p.airdate AS play_airdate,
            p.song_id, s.song AS song_title, s.album AS album_title,
            s.artist AS artist_name, s.release_date AS song_release_date,
            sh.host_names AS dj_host_names
        FROM plays p
        LEFT JOIN songs s ON p.song_id = s.song_id
        LEFT JOIN shows sh ON p.show = sh.id
        WHERE p.comment IS NOT NULL AND TRIM(p.comment) != '' AND LENGTH(TRIM(p.comment)) >= 20
        """
        if random_order:
            query += " ORDER BY RANDOM()"
        else:
            query += " ORDER BY p.airdate DESC"
        if limit:
            query += f" LIMIT {limit}"

        logging.info(
            f"RECIPE_DB: Executing query (first 300 chars): {query[:300]}...")
        cursor.execute(query)

        db_count = 0
        for row in cursor:
            raw_comment_text = row['comment_text'] if row['comment_text'] is not None else ""
            record: Dict[str, Any] = {
                "text": raw_comment_text,
                "meta": {
                    "play_id": row['play_id'], "dj_host_names": row['dj_host_names'],
                    "play_airdate": row['play_airdate'], "db_song_id": row['song_id'],
                    "db_song_title": row['song_title'], "db_album_title": row['album_title'],
                    "db_artist_name": row['artist_name'], "db_song_release_date": row['song_release_date']
                }
            }
            all_records.append(record)
            db_count += 1
            if db_count % 500 == 0:
                logging.info(
                    f"RECIPE_DB: Fetched {db_count} raw comments into memory...")
        logging.info(
            f"RECIPE_DB: Finished fetching. Total raw comments loaded into memory: {db_count}")

    except sqlite3.Error as e:
        logging.error(f"RECIPE_DB: Database error during fetch: {e}")
        raise
    except Exception as e:
        logging.error(
            f"RECIPE_DB: An unexpected error occurred during fetch: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logging.info("RECIPE_DB: Database connection closed.")

    # Now yield from the in-memory list
    yielded_count = 0
    for record in all_records:
        yield record
        yielded_count += 1
    logging.info(
        f"RECIPE_DB: Yielded {yielded_count} records from memory to stream.")

# --- REMOVE OLD EMBEDDED SEGMENTATION/NORMALIZATION LOGIC ---
# The following functions are being removed as this logic is now handled
# by the `parse_comment_to_prodigy_tasks` function from the
# `src.kexp_processing.comment_parser` module:
# - normalize_segment_text_for_prodigy
# - URL_REGEX_STR, TEXT_COLON_URL_LINE_PATTERN_FOR_FINDITER, etc. (regex patterns)
# - split_text_into_segments
# - segment_and_normalize_stream
# --- END OF REMOVAL ---


# --- New stream processor using comment_parser module ---
def process_db_stream_with_comment_parser(raw_db_stream: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """
    Processes a stream of raw DB records using the imported parse_comment_to_prodigy_tasks.
    For each raw comment from the DB, it calls parse_comment_to_prodigy_tasks
    and yields all resulting Prodigy tasks (segments).
    """
    processed_raw_comments = 0
    generated_tasks = 0
    for raw_record in raw_db_stream:
        raw_comment_text = raw_record.get("text")
        original_db_meta = raw_record.get("meta")
        processed_raw_comments += 1

        # Ensure that raw_comment_text and original_db_meta are not None
        # fetch_raw_comments_from_db should ensure 'text' is at least "" and 'meta' is a dict
        if raw_comment_text is None or original_db_meta is None:
            play_id_for_log = original_db_meta.get(
                "play_id", "UNKNOWN") if original_db_meta else "UNKNOWN"
            print(
                f"RECIPE: Skipping raw DB record with missing text or meta. Play ID: {play_id_for_log}")
            continue

        # parse_comment_to_prodigy_tasks is expected to handle segmentation,
        # normalization, and any filtering (e.g., min segment length).
        # It yields one or more Prodigy-ready tasks per raw comment.
        task_count_for_this_comment = 0
        for task in parse_comment_to_prodigy_tasks(raw_comment_text, original_db_meta):
            yield task
            task_count_for_this_comment += 1
            generated_tasks += 1

        if task_count_for_this_comment == 0:
            play_id_for_log = original_db_meta.get("play_id", "UNKNOWN")
            # This log helps identify if a specific comment yields no segments from the parser
            # print(f"RECIPE_PARSER_NOTE: Raw comment (Play ID: {play_id_for_log}) yielded no tasks from comment_parser. Text: '{raw_comment_text[:100]}...'")

    print(
        f"RECIPE: `process_db_stream_with_comment_parser` processed {processed_raw_comments} raw comments from DB, generating {generated_tasks} tasks for annotation.")
    if generated_tasks == 0 and processed_raw_comments > 0:
        print("RECIPE_WARN: The comment_parser generated zero tasks from all processed raw comments. "
              "Check the comment_parser logic and filters (e.g., MIN_SEGMENT_LENGTH) "
              "or the raw data itself.")


# --- NER and SpanCat Pre-labeling Logic (largely same as before) ---
# NER_PATTERNS_PRODIGY_FORMAT = [ # REMOVE THIS ENTIRE BLOCK
#     {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"SHAPE": "dd"}, {
#         "IS_PUNCT": True, "OP": "?"}, {"SHAPE": "dddd"}]},
#     {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"SHAPE": "dddd"}]},
#     {"label": "DATE", "pattern": [{"LOWER": {
#         "IN": ["spring", "summer", "fall", "winter", "autumn"]}}, {"SHAPE": "dddd"}]},
#     {"label": "DATE", "pattern": [{"SHAPE": "dddd", "LENGTH": 4}]},
#     {"label": "LOCATION", "pattern": [
#         {"IS_TITLE": True, "OP": "+"}, {"TEXT": ","}, {"IS_TITLE": True, "OP": "+"}]},
#     {"label": "LOCATION", "pattern": [{"LOWER": {"IN": [
#         "in", "from", "at", "near", "based", "based in"]}}, {"IS_TITLE": True, "OP": "+"}]}
# ]
METADATA_SPANCAT_TAGS_MAP = {
    "db_artist_name": "ARTIST_NAME_TAG", "db_album_title": "ALBUM_TITLE_TAG",
    "db_song_title": "SONG_TITLE_TAG", "db_record_label_name": "RECORD_LABEL_NAME_TAG"
}


def add_metadata_spans_and_finalize(stream: Iterable[Dict[str, Any]], nlp: spacy.Language) -> Iterable[Dict[str, Any]]:
    skipped_count = 0
    for task in stream:
        text_content = task.get("text")
        if not text_content:
            input_hash = task.get("_input_hash")
            task_hash = task.get("_task_hash")
            if input_hash is not None:
                print(
                    f"RECIPE_FIN: Skipping task with no text. Input Hash: {input_hash}")
            elif task_hash is not None:
                print(
                    f"RECIPE_FIN: Skipping task with no text and no input_hash. Task Hash: {task_hash}")
            else:
                print(
                    f"RECIPE_FIN: Skipping task with no text and no hashes.")
            skipped_count += 1
            continue

        doc = nlp(text_content)
        meta_content = task.get("meta", {})
        # Spans will now only come from metadata PhraseMatcher
        all_found_spans = []  # Initialize as empty, no NER spans to inherit

        metadata_phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns_added_to_matcher = False
        for meta_key, label_tag in METADATA_SPANCAT_TAGS_MAP.items():
            search_val = meta_content.get(meta_key)
            if search_val and str(search_val).strip():
                pattern_doc = nlp.make_doc(str(search_val).strip())
                if len(pattern_doc) > 0:
                    # Use a key that combines label_tag and meta_key for later retrieval
                    matcher_key = f"{label_tag}::{meta_key}"
                    metadata_phrase_matcher.add(matcher_key, [pattern_doc])
                    patterns_added_to_matcher = True

        if patterns_added_to_matcher:
            found_phrase_matches = metadata_phrase_matcher(doc)
            for match_id_hash, start_token, end_token in found_phrase_matches:
                match_id_str = nlp.vocab.strings[match_id_hash]
                # Split the matcher_key to get the label_tag and source_meta_field
                current_label_tag, source_meta_field = match_id_str.split(
                    "::", 1)

                span = doc[start_token:end_token]
                all_found_spans.append({
                    "start": span.start_char, "end": span.end_char, "label": current_label_tag,
                    # Corrected source
                    "text": span.text, "source": f"meta_{source_meta_field}",
                    "token_start": span.start, "token_end": span.end - 1
                })

        unique_spans = []
        seen_spans_tuples = set()
        all_found_spans.sort(key=lambda s: (
            s["start"], -(s["end"] - s["start"]), s["label"]))
        for span_dict in all_found_spans:
            if "token_start" not in span_dict or "token_end" not in span_dict:
                char_span = doc.char_span(span_dict["start"], span_dict["end"])
                if char_span:
                    span_dict["token_start"] = char_span.start
                    span_dict["token_end"] = char_span.end - 1
                else:
                    print(
                        f"RECIPE_FIN: Could not align span to tokens: {span_dict} in text: '{doc.text[:50]}...'")
                    continue
            span_tuple = (span_dict["token_start"],
                          span_dict["token_end"], span_dict["label"])
            if span_tuple not in seen_spans_tuples:
                unique_spans.append(span_dict)
                seen_spans_tuples.add(span_tuple)

        task["spans"] = unique_spans
        task = set_hashes(task)
        yield task
    if skipped_count > 0:
        print(
            f"RECIPE_FIN: Skipped {skipped_count} tasks due to missing text during finalization.")


# --- Utility for debugging stream counts ---
def log_stream_counts(stream: Iterator[Any], step_name: str) -> Iterator[Any]:
    print(
        f"RECIPE_STREAM_DEBUG: Step '{step_name}' received stream. Counting items...")
    # To count, we need to convert to list first if we want a pre-count,
    # but that would exhaust the iterator. So, we'll count as we yield.
    # Alternatively, just log before and after iteration.
    # For simplicity here, we'll just log entry and then count as we go.

    # If we want to count input items without exhausting:
    # temp_list = list(stream) # This would exhaust 'stream' if it's a generator
    # print(f"RECIPE_STREAM_DEBUG: Step '{step_name}' input item count: {len(temp_list)}")
    # stream_to_process = iter(temp_list)

    # Let's try a simpler approach: log when it starts, and let the existing logs from
    # process_db_stream_with_comment_parser tell us about its output.
    # For other steps, we will count items yielded.

    print(f"RECIPE_STREAM_DEBUG: Step '{step_name}' processing stream...")
    count = 0
    for item in stream:
        yield item
        count += 1
    print(f"RECIPE_STREAM_DEBUG: Step '{step_name}' yielded {count} items.")


@prodigy.recipe(
    "kexp.smart_prelabel_v2",
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("SpaCy model for tokenization and matching",
                 "option", "sm", str),
    labels_file=("Path to text file with labels (one per line)",
                 "option", "lf", str),
    db_limit=("Optional: Limit number of comments from DB (default: 2000)",
              "option", "dbl", int),
    db_random_order=("Fetch comments from DB in random order",
                     "flag", "dbr", bool)
)
def smart_prelabel_integrated_recipe(
    dataset: str,
    spacy_model: str = "en_core_web_sm",
    labels_file: str = "config/labels.txt",
    db_limit: int = 2000,
    db_random_order: bool = False,
):
    """
    Prodigy recipe to:
    1. Fetch raw comments from KEXP DB (via KEXP_DB_PATH env var).
    2. Segment comments and normalize each segment.
    3. Pre-label NER (LOCATION, DATE) using ProdigyPatternMatcher.
    4. Pre-label specific SpanCat TAGs from DB metadata using spaCy's PhraseMatcher.
    Presents a unified interface for NER and SpanCat annotation.
    """
    nlp = spacy.load(spacy_model)
    db_path = get_db_path_for_recipe()

    # Load labels from file
    labels_path = pathlib.Path(labels_file)
    if not labels_path.is_file():
        print(f"RECIPE: Labels file not found at {labels_file}")
        raise SystemExit(1)

    ui_labels_list = [
        line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    if not ui_labels_list:
        print(
            f"RECIPE: No labels found in {labels_file} or file is empty.")
        raise SystemExit(1)
    ui_labels_list = sorted(list(set(ui_labels_list)))  # Deduplicate and sort

    # --- Stream Processing Pipeline ---
    # 1. Fetch raw comments from DB
    raw_comment_stream_unlogged = fetch_raw_comments_from_db(
        db_path, limit=db_limit, random_order=db_random_order)
    raw_comment_stream = log_stream_counts(
        raw_comment_stream_unlogged, "0_fetch_raw_db")

    # 2. Use comment_parser to get Prodigy tasks from raw comments
    parsed_task_stream_unlogged = process_db_stream_with_comment_parser(
        raw_comment_stream)
    parsed_task_stream = log_stream_counts(
        parsed_task_stream_unlogged, "1_comment_parser")

    # 3. Add tokens
    tokenized_stream_unlogged = add_tokens(nlp, parsed_task_stream)
    tokenized_stream = log_stream_counts(
        tokenized_stream_unlogged, "2_add_tokens")

    # 4. NER Matching step is now REMOVED.
    # The tokenized_stream will be passed directly to the next step.
    # ner_pattern_matcher_component = ProdigyPatternMatcher( # REMOVED
    #     nlp, NER_PATTERNS_PRODIGY_FORMAT # REMOVED
    # ) # REMOVED
    # ner_applied_stream_unlogged = ner_pattern_matcher_component(tokenized_stream) # REMOVED
    # ner_applied_stream = log_stream_counts(ner_applied_stream_unlogged, "3_ner_matcher") # REMOVED
    # FOR TESTING: Feed tokenized_stream directly to the next step # REMOVED
    # ner_applied_stream = log_stream_counts(tokenized_stream, "3_ner_matcher_BYPASSED") # REMOVED

    # The stream from add_tokens is now directly used by add_metadata_spans_and_finalize
    # Let's rename tokenized_stream for clarity in the next step's input or use it directly
    # For logging purposes, we can wrap it again or rely on finalize_spans's log.
    # To keep the log sequence:
    stream_before_metadata_spans = log_stream_counts(
        tokenized_stream, "3_pre_metadata_spans")

    # 5. Add metadata-based SpanCat TAGs and finalize spans
    # This function now receives the output of add_tokens directly (wrapped for logging)
    final_stream_unlogged = add_metadata_spans_and_finalize(
        stream_before_metadata_spans, nlp)
    final_stream = log_stream_counts(
        iter(final_stream_unlogged), "4_finalize_spans")

    label_colors = {
        # "LOCATION": "#FF6347", # REMOVED
        # "DATE": "#4682B4", # REMOVED
        "ARTIST_NAME_TAG": "#FFD700",
        "ALBUM_TITLE_TAG": "#ADFF2F", "SONG_TITLE_TAG": "#87CEFA",
        "RECORD_LABEL_NAME_TAG": "#DA70D6"
    }
    default_color = "#D3D3D3"
    custom_colors = {label: label_colors.get(
        label, default_color) for label in ui_labels_list}

    return {
        "dataset": dataset, "stream": final_stream, "view_id": "spans_manual",
        "config": {
            "labels": ui_labels_list, "span_labels": ui_labels_list,
            "exclude_by_input": True, "custom_theme": {"labels": custom_colors},
        },
    }
