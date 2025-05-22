import sqlite3
import json
import os
import pathlib
import logging
from dotenv import load_dotenv

# Assuming normalization functions are in src.kexp_processing_utils
# Adjust the import path if your project structure is different
from kexp_processing_utils.normalization import normalize_text_for_gazetteer, clean_term, is_well_formed

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    load_dotenv()
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def fetch_and_process_terms(cursor, query, label):
    """
    Fetches terms using the given query, processes them, and returns a list of dicts.
    Each dict: {"text": original_cleaned_term, "normalized": normalized_match_term, "label": label}
    """
    processed_terms = []
    raw_count = 0
    processed_count = 0
    cursor.execute(query)
    for row in cursor.fetchall():
        raw_term_text = row[0]
        raw_count += 1
        if raw_term_text is None or not str(raw_term_text).strip():
            continue

        cleaned_original_term = clean_term(str(raw_term_text))

        if cleaned_original_term and is_well_formed(cleaned_original_term):
            normalized_match_term = normalize_text_for_gazetteer(
                cleaned_original_term)
            if normalized_match_term:  # Ensure normalization didn't result in None/empty
                processed_terms.append({
                    "text": cleaned_original_term,
                    "normalized": normalized_match_term,
                    "label": label
                })
                processed_count += 1
    logging.info(
        f"Fetched {raw_count} raw terms for {label}. After cleaning and validation, {processed_count} terms were processed.")
    return processed_terms


def generate_gazetteers(db_path, output_dir_path_str):
    """
    Generates gazetteers for artists, albums, and songs from the KEXP database.
    Outputs JSONL files to the specified directory.
    """
    output_dir = pathlib.Path(output_dir_path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir.resolve()}")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info(f"Successfully connected to database: {db_path}")

        entities_to_extract = {
            "ARTIST_TAG": [
                "SELECT DISTINCT artist FROM songs WHERE artist IS NOT NULL AND TRIM(artist) != ''",
                "SELECT DISTINCT artist FROM song_artist WHERE artist IS NOT NULL AND TRIM(artist) != ''"
            ],
            "ALBUM_TAG": [
                "SELECT DISTINCT album FROM songs WHERE album IS NOT NULL AND TRIM(album) != ''",
                "SELECT DISTINCT album FROM song_artist WHERE album IS NOT NULL AND TRIM(album) != ''"
            ],
            "SONG_TAG": [
                "SELECT DISTINCT song FROM songs WHERE song IS NOT NULL AND TRIM(song) != ''"
            ]
        }

        all_processed_entities = {}

        for label, queries in entities_to_extract.items():
            logging.info(f"Processing {label}...")
            current_label_terms = []
            for query in queries:
                logging.info(f"Executing query for {label}: {query[:100]}...")
                current_label_terms.extend(
                    fetch_and_process_terms(cursor, query, label))

            # Deduplicate by normalized form for the current label
            unique_normalized_terms = {}
            for item in current_label_terms:
                # Prefer shorter original "text" representations if normalized forms are identical
                # or simply the first one encountered. Here, we take the first one.
                if item['normalized'] not in unique_normalized_terms:
                    unique_normalized_terms[item['normalized']] = item

            all_processed_entities[label] = list(
                unique_normalized_terms.values())
            logging.info(
                f"Found {len(all_processed_entities[label])} unique normalized terms for {label}.")

        # Write gazetteers
        for label, items in all_processed_entities.items():
            output_file_path = output_dir / f"{label.lower()}_gazetteer.jsonl"
            with open(output_file_path, 'w', encoding='utf-8') as f:
                count = 0
                for item in items:
                    # Using "pattern" key as per user request for PhraseMatcher compatibility
                    f.write(json.dumps(
                        {"label": item["label"], "text": item["text"], "pattern": item["normalized"]}) + '\n')
                    count += 1
            logging.info(f"Wrote {count} entries to {output_file_path}")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    try:
        db_path = get_db_path()
        # Default output directory relative to this script or a fixed path
        # For consistency with prodigy recipe, let's use a subdir in data/
        script_dir = pathlib.Path(__file__).parent.resolve()
        default_output_dir = script_dir.parent / "data" / "gazetteers"

        # Allow overriding output directory via environment variable if desired
        output_directory = os.environ.get(
            "GAZETTEER_OUTPUT_DIR", str(default_output_dir))

        logging.info("Starting gazetteer generation...")
        generate_gazetteers(db_path, output_directory)
        logging.info("Gazetteer generation finished successfully.")
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"Unhandled error in main execution: {e}")
