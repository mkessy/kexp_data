# src/scripts/prelabel_config.py

# Default spaCy model
DEFAULT_SPACY_MODEL = "en_core_web_trf"

# Define label priorities for conflict resolution
# Lower number means higher priority
LABEL_PRIORITY = {
    "METADATA_ARTIST_NAME": 0,
    "METADATA_ALBUM_TITLE": 1,
    "METADATA_SONG_TITLE": 2,
    # Add other labels and their priorities as needed.
    # A higher number means lower priority.
    # Labels not in this dict will get a default priority (e.g., 99).
}

# Configuration for metadata fields to be extracted and matched
# Format: {metadata_key_in_input_json: (label_prefix_for_spacy, source_suffix_for_logging)}
METADATA_FIELDS_TO_CHECK = {
    "db_artist_name": ("ARTIST_NAME", "artist"),
    "db_song_title": ("SONG_TITLE", "song"),
    "db_album_title": ("ALBUM_TITLE", "album"),
}

# Source string for matched spans in Prodigy output
PRODIGY_SPAN_SOURCE = "metadata_match_v5_refactored"

# Disabled spaCy components for pre-labeling
# This speeds up processing by only keeping what's necessary (tokenizer).
SPACY_DISABLED_COMPONENTS = ["parser", "tagger",
                             "ner", "lemmatizer", "attribute_ruler"]
