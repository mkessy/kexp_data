import os
import sys
import yaml
import json
import datetime

# Utility functions moved from individual scripts


def load_annotation_schema(schema_path):
    """Loads the consolidated annotation schema YAML file."""
    if not os.path.exists(schema_path):
        print(f"Error: Annotation schema file not found at {schema_path}")
        # sys.exit(1) # Consider raising an exception instead for better programmatic handling
        raise FileNotFoundError(
            f"Error: Annotation schema file not found at {schema_path}")
    with open(schema_path, 'r') as f:
        try:
            schema = yaml.safe_load(f)
            # Basic validation (more comprehensive validation should be done by validate_schema.py)
            if 'schema_version' not in schema or 'labels' not in schema:
                print(
                    f"Error: Invalid schema format in {schema_path}. Missing 'schema_version' or 'labels'.")
                # sys.exit(1)
                raise ValueError(
                    f"Error: Invalid schema format in {schema_path}. Missing 'schema_version' or 'labels'.")
            return schema
        except yaml.YAMLError as e:
            print(f"Error parsing YAML schema file {schema_path}: {e}")
            # sys.exit(1)
            raise ValueError(
                f"Error parsing YAML schema file {schema_path}: {e}")


def update_dataset_registry(registry_path, dataset_info):
    """Updates the dataset registry CSV file."""
    header_fields = [
        "prodigy_dataset_name", "campaign_name", "creation_date", "status",
        "input_examples_file", "ann_index_file", "prodigy_export_file",
        "annotation_schema_version", "description", "source_prodigy_datasets_merged"
    ]
    header = ",".join(header_fields)

    # Ensure all fields are present in dataset_info, use 'N/A' for missing ones
    # Preserve order as in header_fields
    row_data = [str(dataset_info.get(field, 'N/A')) for field in header_fields]

    file_exists = os.path.isfile(registry_path)
    is_empty = not file_exists or os.path.getsize(registry_path) == 0

    with open(registry_path, 'a', newline='') as f:
        if is_empty:
            f.write(header + '\n')
        f.write(",".join(row_data) + '\n')
    print(f"Dataset registry updated: {registry_path}")


def load_jsonl(file_path):
    """Loads a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping line in {file_path} due to JSON decode error: {e} - Line: '{line.strip()}'")
    return data


def save_jsonl(file_path, data):
    """Saves data to a JSONL file."""
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Data saved to JSONL file: {file_path}")


def validate_prodigy_export_data(data, schema_labels=None, text_key='text', spans_key='spans'):
    """
    Validates a list of Prodigy-like export examples.
    - Checks for presence of text_key.
    - If spans_key exists, validates individual spans for required keys and char offsets.
    - If schema_labels is provided, checks if span labels are in the schema_labels set.
    Returns: list of errors (strings), list of warnings (strings)
    """
    errors = []
    warnings = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            errors.append(f"Example at index {i} is not a dictionary.")
            continue
        if text_key not in ex:
            errors.append(
                f"Example at index {i} (ID: {ex.get('_input_hash', 'N/A')}) missing '{text_key}'.")
            continue

        example_text = ex[text_key]
        if not isinstance(example_text, str):
            errors.append(
                f"Example at index {i} (ID: {ex.get('_input_hash', 'N/A')}) '{text_key}' is not a string.")
            # Cannot validate span offsets if text is not string
            continue

        if spans_key in ex:
            if ex[spans_key] is None:  # Explicit None check is important
                warnings.append(
                    f"Example at index {i} (ID: {ex.get('_input_hash', 'N/A')}) has 'spans': null.")
            elif not isinstance(ex[spans_key], list):
                errors.append(
                    f"Example at index {i} (ID: {ex.get('_input_hash', 'N/A')}) '{spans_key}' is not a list.")
            else:
                for j, span in enumerate(ex[spans_key]):
                    if not isinstance(span, dict):
                        errors.append(
                            f"Example {i}, Span {j}: not a dictionary.")
                        continue
                    required_span_keys = {'label', 'start', 'end'}
                    missing_keys = required_span_keys - set(span.keys())
                    if missing_keys:
                        errors.append(
                            f"Example {i}, Span {j} (label: {span.get('label', '?')}) missing keys: {missing_keys}.")

                    label = span.get('label')
                    start = span.get('start')
                    end = span.get('end')

                    if schema_labels and label not in schema_labels:
                        # This check is already done during merging in 05b, but good for a generic validator
                        warnings.append(
                            f"Example {i}, Span {j}: Label '{label}' not in provided schema_labels.")

                    if not isinstance(label, str):
                        # Truncate label in error msg
                        errors.append(
                            f"Example {i}, Span {j}: Label '{str(label)[:30]}' is not a string.")
                    if not isinstance(start, int):
                        errors.append(
                            f"Example {i}, Span {j} (label: {str(label)[:30]}): Start offset '{start}' is not an integer.")
                    if not isinstance(end, int):
                        errors.append(
                            f"Example {i}, Span {j} (label: {str(label)[:30]}): End offset '{end}' is not an integer.")

                    if isinstance(start, int) and isinstance(end, int):
                        if start < 0 or end < 0:
                            errors.append(
                                f"Example {i}, Span {j} (label: {str(label)[:30]}): Start/End offsets ({start},{end}) cannot be negative.")
                        if start > end:
                            errors.append(
                                f"Example {i}, Span {j} (label: {str(label)[:30]}): Start offset {start} > end offset {end}.")
                        if end > len(example_text):
                            errors.append(
                                f"Example {i}, Span {j} (label: {str(label)[:30]}): End offset {end} > text length {len(example_text)} (text ID: {ex.get('_input_hash', 'N/A')}).")

    return errors, warnings
