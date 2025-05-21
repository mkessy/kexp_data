#!/usr/bin/env python3
import sqlite3
import json
import os
import argparse
import logging
from dotenv import load_dotenv
from kexp_processing.normalization import normalize_text
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env variables
load_dotenv()


def get_db_path():
    """Gets the database path from the KEXP_DB_PATH environment variable."""
    db_path = os.environ.get('KEXP_DB_PATH')
    if not db_path:
        logging.error("KEXP_DB_PATH environment variable not set.")
        raise ValueError("KEXP_DB_PATH environment variable not set.")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at: {db_path}")
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    return db_path


def fetch_play_comments_with_details(db_path, limit=None, normalize=False, random=False):
    """
    Fetches plays with comments from the KEXP database and joins them with song
    (for song title, album, artist, release_date) and show (for host names) information.
    If normalize is True, normalizes the comment text using normalize_text.
    If random is True, selects comments in random order.

    Args:
        db_path (str): Path to the SQLite database file.
        limit (int, optional): Maximum number of records to fetch. Defaults to None (all).
        normalize (bool, optional): If True, normalize the comment text. Defaults to False.
        random (bool, optional): If True, select comments in random order. Defaults to False.

    Yields:
        dict: A dictionary representing a play's comment with its associated metadata.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()

        # Schema based on user input:
        # plays.song_id -> songs.song_id
        # songs table contains: song (title), artist (name), album (title), release_date
        # plays.show -> shows.id (to get host_names)
        # Comment is directly on the plays table.
        query = """
        SELECT
            p.id AS play_id,
            p.comment AS comment_text,
            p.airdate AS play_airdate,
            p.song_id,
            s.song AS song_title,          -- From songs.song
            s.album AS album_title,        -- From songs.album
            s.artist AS artist_name,       -- From songs.artist
            s.release_date AS song_release_date, -- From songs.release_date
            sh.host_names AS dj_host_names -- Get host names from the shows table
        FROM
            plays p
        LEFT JOIN
            songs s ON p.song_id = s.song_id -- Corrected join field for songs
        LEFT JOIN
            shows sh ON p.show = sh.id      -- Join with shows table
        WHERE
            p.comment IS NOT NULL AND TRIM(p.comment) != ''
            AND LENGTH(TRIM(p.comment)) >= 20
        """

        if random:
            query += " ORDER BY RANDOM()"
        else:
            query += " ORDER BY p.airdate DESC"

        if limit:
            query += f" LIMIT {limit}"

        logging.info(f"Executing query (first 500 chars): {query[:500]}...")
        cursor.execute(query)

        count = 0
        for row in cursor:
            # Ensure comment_text (which is p.comment) is a string and not None
            comment_text = row['comment_text'] if row['comment_text'] is not None else ""
            output_text = normalize_text(
                comment_text) if normalize else comment_text

            record = {
                # This is the DJ's comment (normalized if requested)
                "text": output_text,
                "meta": {
                    "play_id": row['play_id'],
                    "dj_host_names": row['dj_host_names'],
                    "play_airdate": row['play_airdate'],
                    # Keep the original song_id from DB
                    "db_song_id": row['song_id'],
                    "db_song_title": row['song_title'],
                    "db_album_title": row['album_title'],
                    "db_artist_name": row['artist_name'],
                    "db_song_release_date": row['song_release_date']
                }
            }
            yield record
            count += 1
            if count % 1000 == 0:
                logging.info(f"Processed {count} play comments...")

        logging.info(
            f"Finished processing. Total play comments fetched: {count}")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract KEXP play comments with song/artist/show details to JSONL for Prodigy.")
    parser.add_argument(
        "output_file",
        help="Path to the output JSONL file (e.g., data/raw_kexp_data/kexp_play_comments_with_meta.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Limit the number of comments to process (for testing)."
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="If set, normalize the comment text using the normalization utility."
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, select comments in random order (up to the limit)."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        db_path = get_db_path()
        logging.info(f"Starting data extraction from: {db_path}")
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in fetch_play_comments_with_details(db_path, args.limit, normalize=args.normalize_text, random=args.random):
                outfile.write(json.dumps(record) + '\n')
        logging.info(
            f"Successfully wrote play comments to: {args.output_file}")
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Setup error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
