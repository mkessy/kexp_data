import prodigy
from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes
import spacy
from typing import Optional, List, Dict, Any, Iterable, Iterator
import os
import sqlite3
from dotenv import load_dotenv
import logging
from spacy.matcher import PhraseMatcher
import pathlib
import json
from datetime import datetime
from itertools import tee
import json
from sentence_transformers import SentenceTransformer
import hnswlib
from prodigy.components.filters import filter_duplicates
import re

# Import the new parser function
from src.kexp_processing_utils.comment_parser import parse_comment_to_prodigy_tasks

# Configure logging - Prodigy uses its own logging, but this can be for DB part
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables for KEXP_DB_PATH
load_dotenv()

# --- Helper constants and function for DATE entity filtering ---
MONTH_NAMES = [
    "january", "jan", "february", "feb", "march", "mar", "april", "apr",
    "may", "june", "jun", "july", "jul", "august", "aug", "september",
    "sep", "sept", "october", "oct", "november", "nov", "december", "dec"
]
DAY_NAMES = [
    "monday", "mon", "tuesday", "tue", "tues", "wednesday", "wed",
    "thursday", "thu", "thur", "thurs", "friday", "fri", "saturday",
    "sat", "sunday", "sun"
]
RELATIVE_DATE_TERMS = [
    "today", "tomorrow", "yesterday", "tonight",
    "next week", "last week", "next month", "last month", "next year", "last year",
    "this morning", "this afternoon", "this evening"
]
# Regex for typical time-only patterns (e.g., 10am, 10:30, 5 o'clock)
TIME_ONLY_REGEX = re.compile(
    r"^\\d{1,2}(:\\d{2})?(\\s*(am|pm|a\\.m\\.|p\\.m\\.))?(\\s*o'?clock)?$",
    re.IGNORECASE
)
# Regex for patterns that are likely just years or decades (e.g., 2023, 1990s, '90s)
YEAR_DECADE_REGEX = re.compile(r"^((\\d{4}s?)|('\\d{2}s))$", re.IGNORECASE)


def is_likely_date_not_just_time(ent_text: str) -> bool:
    """
    Checks if a spaCy DATE entity text is likely a date or date-related
    and not just a time of day.
    """
    text_lower = ent_text.lower()

    if any(month in text_lower for month in MONTH_NAMES):
        return True
    if any(day in text_lower for day in DAY_NAMES):
        return True
    if any(term in text_lower for term in RELATIVE_DATE_TERMS):
        return True
    if "/" in text_lower or "-" in text_lower:  # Common date separators
        # Avoid matching things like "end-of-day" if they are not full dates
        # A simple check: ensure there are digits involved if a hyphen is present
        if "-" in text_lower and not any(char.isdigit() for char in text_lower):
            pass  # Could be a phrase, not a date like "2023-10-05"
        else:
            return True
    if YEAR_DECADE_REGEX.match(text_lower):
        return True
    if re.search(r"\\b\\d{4}\\b", text_lower):  # 4-digit year
        return True

    # If it matches a strict time-only pattern, then it's likely just time.
    if TIME_ONLY_REGEX.fullmatch(text_lower):
        return False

    # Keep if it has letters (e.g., "the fifth", "mid-June") and wasn't caught by TIME_ONLY_REGEX
    if any(char.isalpha() for char in text_lower):
        return True

    # Keep if it's one or two digits (could be a day of the month)
    if re.fullmatch(r"\\d{1,2}", text_lower):
        return True

    # Fallback: if it contains digits, and wasn't filtered as time-only, lean towards including.
    # This is a bit broad but aims to catch more date formats.
    if any(char.isdigit() for char in text_lower):
        return True

    return False  # Default to excluding if very uncertain


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
    "db_artist_name": "ARTIST_TAG",
    "db_album_title": "ALBUM_TAG",
    "db_song_title": "SONG_TAG",
    "db_record_label_name": "RECORD_LABEL_TAG"
}

# Default path for gazetteers relative to the workspace root
# DEFAULT_GAZETTEER_DIR = pathlib.Path("data/gazetteers/") # This will be removed


def load_gazetteers_to_phrasematcher(nlp: spacy.Language, gazetteer_dir: pathlib.Path) -> PhraseMatcher:
    """
    Loads terms from gazetteer JSONL files into a spaCy PhraseMatcher.
    Assumes gazetteer files are named like 'artist_tag_gazetteer.jsonl'.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    gazetteer_files = {
        "ROLE_TAG": gazetteer_dir / "role_tag_gazetteer.jsonl",
        "GENRE_TAG": gazetteer_dir / "genre_tag_gazetteer.jsonl",
    }

    for label, filepath in gazetteer_files.items():
        if filepath.is_file():
            patterns = []
            with filepath.open('r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Ensure 'pattern' key exists and is a non-empty string
                    pattern_text = item.get("pattern")
                    if pattern_text and isinstance(pattern_text, str) and pattern_text.strip():
                        patterns.append(nlp.make_doc(pattern_text))
                    else:
                        logging.warning(
                            f"RECIPE_GAZ: Invalid or empty pattern in {filepath} for item: {item}")
            if patterns:
                matcher.add(label, patterns)
                logging.info(
                    f"RECIPE_GAZ: Loaded {len(patterns)} patterns for {label} from {filepath}")
            else:
                logging.warning(
                    f"RECIPE_GAZ: No valid patterns found for {label} in {filepath}")
        else:
            logging.warning(
                f"RECIPE_GAZ: Gazetteer file not found for {label}: {filepath}")
    return matcher


def add_metadata_spans_and_finalize(stream: Iterable[Dict[str, Any]], nlp: spacy.Language, global_gazetteer_matcher: Optional[PhraseMatcher] = None) -> Iterable[Dict[str, Any]]:
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

        # The `nlp` object used here will be the one loaded in the recipe,
        # which includes the NER component if the model has one.
        # `add_tokens` already runs `nlp(text)` and adds `doc` to the task if not present,
        # or uses existing `doc`. However, to be explicit and ensure `doc.ents` is fresh
        # for this text_content, we process it here.
        # If `task` already contains a `doc` from `add_tokens`, it should be for `text_content`.
        # Using `task.get("doc")` might be slightly more efficient if `add_tokens` guarantees it.
        # Let's assume `add_tokens` correctly populates `task["doc"]`.
        # If not, `doc = nlp(text_content)` is the fallback.
        doc = task.get("doc")
        if not doc:  # Fallback if doc not present from add_tokens
            doc = nlp(text_content)

        meta_content = task.get("meta", {})
        all_found_spans = []

        # 1. Add NER-derived spans using the main nlp model
        # This `nlp` model is loaded by `spacy.load(spacy_model)` in the recipe.
        # It will perform NER if the model (e.g., en_core_web_sm) has an NER pipe.
        if "ner" in nlp.pipe_names:  # Check if NER component exists
            for ent in doc.ents:
                label = None
                source_ner_label = ent.label_  # Original NER label for source tracking
                if ent.label_ == "PERSON":
                    label = "ARTIST_TAG"
                elif ent.label_ == "DATE":
                    if is_likely_date_not_just_time(ent.text):
                        label = "DATE_TAG"
                # Geopolitical Entity (countries, cities, states)
                elif ent.label_ == "GPE":
                    label = "LOC_TAG"
                # Potentially add other mappings like ORG -> RECORD_LABEL_TAG if appropriate and distinct enough

                if label:
                    all_found_spans.append({
                        "start": ent.start_char, "end": ent.end_char, "label": label,
                        "text": ent.text, "source": f"ner_spacy_{source_ner_label}",
                        "token_start": ent.start, "token_end": ent.end - 1
                    })

        # 2. Process play-specific metadata matches
        metadata_phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns_added_to_matcher = False
        for meta_key, label_tag in METADATA_SPANCAT_TAGS_MAP.items():
            search_val = meta_content.get(meta_key)
            if search_val and str(search_val).strip():
                # Use nlp.make_doc for consistency, PhraseMatcher works with Doc patterns
                pattern_doc = nlp.make_doc(str(search_val).strip())
                if len(pattern_doc) > 0:
                    # Key for PhraseMatcher
                    matcher_key = f"{label_tag}::{meta_key}"
                    metadata_phrase_matcher.add(matcher_key, [pattern_doc])
                    patterns_added_to_matcher = True

        if patterns_added_to_matcher:
            found_phrase_matches = metadata_phrase_matcher(doc)
            for match_id_hash, start_token, end_token in found_phrase_matches:
                raw_match_id = nlp.vocab.strings[match_id_hash]
                match_id_str = str(raw_match_id)

                if "::" in match_id_str:
                    current_label_tag, source_meta_field = match_id_str.split(
                        "::", 1)
                else:
                    logging.warning(
                        f"RECIPE_FIN: Metadata Match ID string '{match_id_str}' (from hash {match_id_hash}, raw_id '{raw_match_id}') does not contain '::'. Skipping span.")
                    continue

                span = doc[start_token:end_token]
                all_found_spans.append({
                    "start": span.start_char, "end": span.end_char, "label": current_label_tag,
                    "text": span.text, "source": f"meta_{source_meta_field}",
                    "token_start": span.start, "token_end": span.end - 1
                })

        # 3. Process global gazetteer matches
        if global_gazetteer_matcher:
            global_matches = global_gazetteer_matcher(doc)
            for match_id_hash, start_token, end_token in global_matches:
                label_from_matcher = nlp.vocab.strings[match_id_hash]
                current_label_tag = str(label_from_matcher)
                span = doc[start_token:end_token]
                all_found_spans.append({
                    "start": span.start_char, "end": span.end_char, "label": current_label_tag,
                    "text": span.text, "source": f"gazetteer_{current_label_tag.lower()}",
                    "token_start": span.start, "token_end": span.end - 1
                })

        # 4. Deduplicate and finalize spans
        # Sort by start_char, then by inverse length (longer preferred), then by label.
        # This helps in a consistent order for deduplication.
        all_found_spans.sort(key=lambda s: (
            s["start"], -(s["end"] - s["start"]), s["label"]))

        unique_spans = []
        seen_spans_tuples = set()  # Stores (token_start, token_end, label)

        for span_dict in all_found_spans:
            # Ensure token_start and token_end are present.
            # NER and PhraseMatcher spans should have them directly.
            # If they were derived only from char offsets, align to tokens.
            if "token_start" not in span_dict or "token_end" not in span_dict:
                # This case should be less common if sources provide token indices
                char_span_for_alignment = doc.char_span(
                    span_dict["start"], span_dict["end"], label=span_dict["label"])
                if char_span_for_alignment:
                    span_dict["token_start"] = char_span_for_alignment.start
                    span_dict["token_end"] = char_span_for_alignment.end - 1
                else:
                    # This can happen if char offsets don't align to token boundaries.
                    # Log and skip this span.
                    # print(
                    #     f"RECIPE_FIN: Could not align char span to tokens: {span_dict} in text: '{doc.text[:70]}...' "
                    #     f"Play ID: {meta_content.get('play_id', 'N/A')}, Segment Hash: {task.get('_input_hash', 'N/A')}"
                    # )
                    continue

            span_tuple = (span_dict["token_start"],
                          span_dict["token_end"], span_dict["label"])

            if span_tuple not in seen_spans_tuples:
                unique_spans.append(span_dict)
                seen_spans_tuples.add(span_tuple)
            # If span_tuple is already seen, this span (from a potentially different source
            # or just a duplicate) is skipped, effectuating deduplication.

        task["spans"] = unique_spans
        task = set_hashes(task)  # Re-hash task if spans changed
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


# --- NEW: Function to write stream to JSONL file and yield tasks ---
def stream_to_file_and_yield(stream: Iterator[Dict[str, Any]], output_file: Optional[str], step_name: str) -> Iterator[Dict[str, Any]]:
    """
    Wraps a stream. If output_file is provided, writes each item to the file (JSONL).
    Then yields the item, effectively tapping into the stream.
    """
    print(
        f"RECIPE_IO_TAP: Step '{step_name}' is now being tapped. Output file: '{output_file if output_file else 'None'}'")
    count = 0
    if output_file:
        # Ensure the directory for the output file exists
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf8") as outfile:
            for task in stream:
                outfile.write(json.dumps(task) + "\n")
                count += 1
                if count % 200 == 0:
                    print(
                        f"RECIPE_IO_TAP: Written {count} tasks to '{output_file}' from step '{step_name}'")
                yield task
        print(
            f"RECIPE_IO_TAP: Finished writing. Total {count} tasks written to '{output_file}' from step '{step_name}'.")
    else:
        # If no output file, just pass through the stream and count
        for task in stream:
            count += 1
            # No periodic log here to avoid confusion if file not being written
            yield task
        print(
            f"RECIPE_IO_TAP: Step '{step_name}' (no output file) yielded {count} tasks.")


# --- NEW: Helper function to write tasks to JSONL and return them as a list ---
def write_tasks_to_jsonl_and_return_list(stream: Iterator[Dict[str, Any]], output_jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Consumes a stream of tasks, writes each task to a JSONL file,
    and returns all tasks as a list.
    """
    tasks_list = []
    count = 0
    output_path = pathlib.Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"RECIPE_PERSIST: Writing tasks to '{output_jsonl_path}' and collecting list...")
    with output_path.open("w", encoding="utf8") as outfile:
        for task in stream:
            outfile.write(json.dumps(task) + "\n")
            tasks_list.append(task)
            count += 1
            if count % 500 == 0:
                print(
                    f"RECIPE_PERSIST: Processed {count} tasks to '{output_jsonl_path}'...")
    print(
        f"RECIPE_PERSIST: Finished. Total {count} tasks written to '{output_jsonl_path}' and collected into list.")
    return tasks_list

# --- NEW: Helper function to build and save ANN index ---


def build_and_save_ann_index(tasks: List[Dict[str, Any]], index_path: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Builds an HNSWLib ANN index from the 'text' field of the provided tasks
    and saves it to the specified path.
    """
    if not tasks:
        print("RECIPE_ANN: No tasks provided to build ANN index. Skipping.")
        return

    print(
        f"RECIPE_ANN: Initializing sentence transformer model '{model_name}'...")
    model = SentenceTransformer(model_name)

    texts = [task.get("text", "") for task in tasks if task.get("text")]
    if not texts:
        print("RECIPE_ANN: No text found in tasks to build ANN index. Skipping.")
        return

    print(
        f"RECIPE_ANN: Encoding {len(texts)} texts for ANN index. This may take a while...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)

    # Ensure index_path's directory exists
    index_file_path = pathlib.Path(index_path)
    index_file_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"RECIPE_ANN: Initializing HNSWLib index at '{index_path}' with {len(embeddings)} elements.")
    index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
    index.add_items(embeddings, ids=list(
        range(len(embeddings))))  # Simple 0-based IDs

    print(f"RECIPE_ANN: Saving ANN index to '{index_path}'...")
    index.save_index(index_path)
    print(f"RECIPE_ANN: ANN index saved successfully.")


@prodigy.recipe(
    "kexp.smart_prelabel_v2",
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("SpaCy model for tokenization, NER, and matching",  # Clarified usage
                 "option", "sm", str),
    labels_file=("Path to text file with labels (one per line)",
                 "option", "lf", str),
    db_limit=("Optional: Limit number of comments from DB (default: 2000)",
              "option", "dbl", int),
    db_random_order=("Fetch comments from DB in random order",
                     "flag", "dbr", bool),
    output_file=("Optional: Path to save all processed tasks to a JSONL file",
                 "option", "out", Optional[str])
)
def smart_prelabel_integrated_recipe(
    dataset: str,
    spacy_model: str = "en_core_web_sm",
    labels_file: str = "config/labels.txt",
    db_limit: int = 2000,
    db_random_order: bool = False,
    output_file: Optional[str] = None
):
    """
    Prodigy recipe to:
    1. Fetch raw comments from KEXP DB (via KEXP_DB_PATH env var).
    2. Segment comments and normalize each segment.
    3. Pre-label specific SpanCat TAGs from DB metadata using spaCy's PhraseMatcher.
    4. Pre-label entities from global gazetteers (ARTIST, ALBUM, SONG, ROLE, GENRE).
    Presents a unified interface for SpanCat annotation.
    """
    # Load the spaCy model. This model will be used for tokenization by add_tokens,
    # for NER in add_metadata_spans_and_finalize, and for PhraseMatcher.
    # Ensure the model has an NER component if NER pre-labeling is desired.
    # Standard models like en_core_web_sm/md/lg include NER.
    nlp = spacy.load(spacy_model)
    logging.info(
        f"RECIPE_CONFIG: Loaded spaCy model '{spacy_model}'. Components: {nlp.pipe_names}")
    if "ner" not in nlp.pipe_names:
        logging.warning(
            f"RECIPE_CONFIG: SpaCy model '{spacy_model}' does not have an NER component. "
            "NER-based pre-labeling for PERSON and DATE will not occur."
        )

    db_path = get_db_path_for_recipe()

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

    # --- Load Global Gazetteers (path is now hardcoded) ---
    actual_gazetteer_dir = pathlib.Path("config/gazetteers/")
    logging.info(
        f"RECIPE_CONFIG: Attempting to load gazetteers from hardcoded path: {actual_gazetteer_dir.resolve()}")
    global_entity_matcher = load_gazetteers_to_phrasematcher(
        nlp, actual_gazetteer_dir)

    # --- Determine output file path for processed tasks ---
    final_output_path: Optional[str] = None
    if output_file:
        final_output_path = output_file
        print(
            f"RECIPE_CONFIG: User specified output file: {final_output_path}")
    else:
        default_dir = pathlib.Path(
            os.getcwd()) / "data" / "processed_examples"  # More robust default
        # default_dir = pathlib.Path("/Users/pooks/Dev/kexp_data/data/processed_examples/") # Old path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{dataset}_examples_{timestamp}.jsonl"
        final_output_path = str(default_dir / default_filename)
        print(
            f"RECIPE_CONFIG: No output file specified by user. Defaulting to: {final_output_path}")

    # The stream_to_file_and_yield function will handle directory creation if needed for final_output_path

    # --- Stream Processing Pipeline ---
    # 1. Fetch raw comments from DB
    raw_comment_stream_unlogged = fetch_raw_comments_from_db(
        db_path, limit=db_limit, random_order=db_random_order)
    raw_comment_stream_with_counts_pre_filter = log_stream_counts(
        raw_comment_stream_unlogged, "0a_fetch_raw_db_unfiltered")

    # 1b. Hash raw comments based on their text content and filter duplicates
    def hash_raw_comments(stream: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for eg in stream:
            # Set hashes based on the raw comment text only for initial de-duplication
            # The key "text" here refers to the raw comment text from the DB
            yield set_hashes(eg, input_keys=("text",), task_keys=())

    hashed_raw_comment_stream = hash_raw_comments(
        raw_comment_stream_with_counts_pre_filter)
    deduplicated_raw_comment_stream_unlogged = filter_duplicates(
        hashed_raw_comment_stream, by_input=True, by_task=False)

    raw_comment_stream = log_stream_counts(
        deduplicated_raw_comment_stream_unlogged, "0b_fetch_raw_db_filtered_deduped")

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
        stream_before_metadata_spans, nlp, global_entity_matcher)
    final_stream_logged_counts = log_stream_counts(
        iter(final_stream_unlogged), "4_finalize_spans_incl_ner_meta_gaz")

    # --- NEW: Split stream, save to file, build ANN index, and serve to Prodigy ---
    # final_stream_for_prodigy = stream_to_file_and_yield( # OLD LINE TO BE REMOVED
    #     final_stream_logged_counts, # OLD LINE TO BE REMOVED
    #     final_output_path,  # MODIFIED to use the determined path # OLD LINE TO BE REMOVED
    #     step_name="5_final_stream_to_prodigy_and_file" # OLD LINE TO BE REMOVED
    # ) # OLD LINE TO BE REMOVED

    # 1. Split the final processed stream
    print(f"RECIPE_PIPELINE: Splitting final stream for ANN indexing/file saving and Prodigy UI.")
    stream_for_ann_and_file, stream_for_prodigy_ui = tee(
        final_stream_logged_counts, 2)

    # 2. Persist tasks to JSONL and get them as a list
    # final_output_path is already determined earlier in the recipe
    if final_output_path:
        print(
            f"RECIPE_PIPELINE: Persisting tasks to {final_output_path} and collecting for ANN index.")
        processed_tasks_list = write_tasks_to_jsonl_and_return_list(
            stream_for_ann_and_file,
            final_output_path
        )

        # 3. Build and save ANN index
        # Derive ANN index path from the JSONL output path
        ann_index_path = str(pathlib.Path(
            final_output_path).with_suffix('.ann_index'))
        print(
            f"RECIPE_PIPELINE: Building and saving ANN index to {ann_index_path}.")
        build_and_save_ann_index(processed_tasks_list, ann_index_path)
    else:
        # This case should ideally not happen if final_output_path is always set (e.g. to a default)
        # If it can, we need to decide how to handle stream_for_ann_and_file
        # For now, assume final_output_path is always valid and this branch is less likely.
        print("RECIPE_PIPELINE_WARN: final_output_path is not set. Skipping file persistence and ANN indexing.")
        # To prevent stream_for_ann_and_file from being unconsumed if we don't use it for file writing:
        # We must consume it if we don't use it for file writing, otherwise tee can cause issues.
        # However, write_tasks_to_jsonl_and_return_list consumes it.
        # If final_output_path is None, then processed_tasks_list won't be created.
        # And build_and_save_ann_index won't be called.
        # stream_for_prodigy will still work.
        # If we *really* wanted to drain stream_for_ann_and_file without writing:
        # for _ in stream_for_ann_and_file: pass
        pass

    label_colors = {
        "ARTIST_TAG": "#FFD700",
        "ALBUM_TAG": "#ADFF2F",
        "SONG_TAG": "#87CEFA",
        "RECORD_LABEL_TAG": "#DA70D6",
        "ROLE_TAG": "#FF7F50",
        "GENRE_TAG": "#6495ED",
        "DATE_TAG": "#D3D3D3",
        "LOC_TAG": "#D3D3D3",
        "ARTIST_BIO_SPAN": "#D3D3D3", "NEW_RELEASE_SPAN": "#D3D3D3",
        "GROUP_COMP_SPAN": "#D3D3D3", "SHOW_DATE_SPAN": "#D3D3D3"
    }
    default_color = "#BEBEBE"  # Slightly different default for truly unspecified
    custom_colors = {label: label_colors.get(
        label, default_color) for label in ui_labels_list}

    # Ensure DATE_TAG is in labels if NER is adding it.
    # ui_labels_list comes from labels_file, so it should be there.
    if "ner" in nlp.pipe_names and "DATE_TAG" not in ui_labels_list:  # Check if NER is active for this warning
        logging.warning("RECIPE_CONFIG: DATE_TAG is being added by NER but not found in labels file. "
                        "Add it to config/labels.txt for it to appear in UI selection.")
        # ui_labels_list.append("DATE_TAG") # Optionally auto-add, but better to require in file

    if "ner" in nlp.pipe_names and "LOC_TAG" not in ui_labels_list:  # Check for LOC_TAG
        logging.warning("RECIPE_CONFIG: LOC_TAG is being added by NER (from GPE entities) but not found in labels file. "
                        "Add it to config/labels.txt for it to appear in UI selection.")

    return {
        "dataset": dataset,
        "stream": stream_for_prodigy_ui,
        "view_id": "spans_manual",
        "config": {
            "labels": ui_labels_list,
            # Keep for compatibility, though "labels" is primary for new Prodigy
            "span_labels": ui_labels_list,
            # Added DATE_TAG, LOC_TAG
            "labels_priority": ["ARTIST_TAG", "ALBUM_TAG", "SONG_TAG", "DATE_TAG", "ROLE_TAG", "GENRE_TAG", "LOC_TAG"],
            "exclude_by_input": True,
            "custom_theme": {"labels": custom_colors},
            "show_whitespace": False,  # Optional: might make UI cleaner
            "show_flag": True,  # Enable flagging UI
        },
    }
