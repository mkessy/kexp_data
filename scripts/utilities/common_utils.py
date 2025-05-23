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
