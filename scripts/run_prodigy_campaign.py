import argparse
import datetime
import json
import os
import subprocess
import sys
import yaml

# Assuming build_ann_index.py and 04_prelabel_for_prodigy.py are in scripts/utilities and scripts/ respectively
# Adjust paths if necessary, or consider making them installable modules
# For now, let's assume they can be called as subprocesses or imported if structured as such.

# Placeholder for a utility function to load the consolidated schema


def load_annotation_schema(schema_path):
    """Loads the consolidated annotation schema YAML file."""
    if not os.path.exists(schema_path):
        print(f"Error: Annotation schema file not found at {schema_path}")
        sys.exit(1)
    with open(schema_path, 'r') as f:
        try:
            schema = yaml.safe_load(f)
            # Basic validation (more in validate_schema.py)
            if 'schema_version' not in schema or 'labels' not in schema:
                print(
                    f"Error: Invalid schema format in {schema_path}. Missing 'schema_version' or 'labels'.")
                sys.exit(1)
            return schema
        except yaml.YAMLError as e:
            print(f"Error parsing YAML schema file {schema_path}: {e}")
            sys.exit(1)

# Placeholder for a utility function to update the dataset registry


def update_dataset_registry(registry_path, dataset_info):
    """Updates the dataset registry CSV file."""
    header = "prodigy_dataset_name,campaign_name,creation_date,status,input_examples_file,ann_index_file,prodigy_export_file,annotation_schema_version,description,source_prodigy_datasets_merged"

    # Ensure all fields are present in dataset_info, use N/A for missing ones
    row_data = [
        dataset_info.get('prodigy_dataset_name', 'N/A'),
        dataset_info.get('campaign_name', 'N/A'),
        dataset_info.get('creation_date', 'N/A'),
        dataset_info.get('status', 'N/A'),
        dataset_info.get('input_examples_file', 'N/A'),
        dataset_info.get('ann_index_file', 'N/A'),
        dataset_info.get('prodigy_export_file', 'N/A'),
        dataset_info.get('annotation_schema_version', 'N/A'),
        dataset_info.get('description', 'N/A'),
        dataset_info.get('source_prodigy_datasets_merged', 'N/A')
    ]

    # Check if file exists to write header or append
    file_exists = os.path.isfile(registry_path)
    with open(registry_path, 'a', newline='') as f:
        # Using csv.writer could be more robust, but for simple append:
        if not file_exists or os.path.getsize(registry_path) == 0:
            f.write(header + '\n')
        f.write(",".join(map(str, row_data)) + '\n')
    print(f"Dataset registry updated: {registry_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a Prodigy annotation campaign session.")
    parser.add_argument("--campaign-config", required=True,
                        help="Path to the campaign's YAML configuration file.")
    parser.add_argument("--prodigy-dataset-name",
                        help="Optional. If provided, use this exact Prodigy dataset name.")
    parser.add_argument("--iteration-number", type=int, required=True,
                        help="Integer for the current iteration within the campaign (e.g., 1, 2).")
    parser.add_argument("--iteration-description", required=True,
                        help="Short string describing the iteration (e.g., 'bootstrap', 'ann_refined_pass1').")
    parser.add_argument("--input-examples-file",
                        help="Optional. Path to a specific JSONL file to use as input. Overrides campaign config's source_data_specification.")
    parser.add_argument("--force-rebuild-ann-index", action="store_true",
                        help="Flag, if input examples are provided, force rebuilding its ANN index.")
    parser.add_argument("--prodigy-port", default="8080",
                        help="Port for the Prodigy server.")
    parser.add_argument("--prodigy-host", default="0.0.0.0",
                        help="Host for the Prodigy server.")
    # Add any other standard Prodigy arguments you might want to pass through

    args = parser.parse_args()

    # 1. Load Configurations
    print("Loading configurations...")
    if not os.path.exists(args.campaign_config):
        print(
            f"Error: Campaign config file not found at {args.campaign_config}")
        sys.exit(1)

    with open(args.campaign_config, 'r') as f:
        try:
            campaign_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(
                f"Error parsing campaign config file {args.campaign_config}: {e}")
            sys.exit(1)

    schema_version = campaign_config.get('annotation_schema_version')
    if not schema_version:
        print(
            f"Error: 'annotation_schema_version' not found in campaign config {args.campaign_config}")
        sys.exit(1)

    # Construct schema path based on convention (e.g., config/schemas/kexp_annotation_schema_vX.Y.yaml)
    # This assumes schema files are in a known relative location to this script or workspace root.
    # For robustness, consider making schema path an absolute path or configurable.
    workspace_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..'))  # Assuming script is in scripts/
    schema_file_name = f"kexp_annotation_schema_v{schema_version}.yaml"
    schema_path = os.path.join(
        workspace_root, "config", "schemas", schema_file_name)

    annotation_schema = load_annotation_schema(schema_path)
    print(
        f"Loaded annotation schema version: {annotation_schema.get('schema_version')}")

    # 2. Determine Prodigy Dataset Name
    print("Determining Prodigy dataset name...")
    prodigy_dataset_name = args.prodigy_dataset_name
    if not prodigy_dataset_name:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        prodigy_dataset_name = f"{campaign_config['campaign_name']}_asv{schema_version}_iter{args.iteration_number}_{args.iteration_description}_{date_str}"
    print(f"Prodigy dataset name: {prodigy_dataset_name}")

    # 3. Prepare Input Stream & ANN Index
    # This is a complex part and will involve several sub-steps:
    # - If args.input_examples_file is provided:
    #   - Use this file.
    #   - Check if a corresponding .ann_index exists or if --force-rebuild-ann-index.
    #   - Call scripts/utilities/build_ann_index.py if needed.
    # - Else (no explicit input file):
    #   - Use source_data_specification from campaign_config.
    #   - If type is "jsonl_file", use that path. Build ANN index if needed.
    #   - If type is "database_query", implement logic to fetch data (placeholder).
    #   - If pre_labeling.use_script_04 is true:
    #     - Run 04_prelabel_for_prodigy.py on the source data.
    #     - The output of script 04 becomes the input for Prodigy and ANN indexing.
    # - Store paths to the final input JSONL (final_input_jsonl_path) and its ANN index (ann_index_path).

    print("Preparing input stream and ANN index (Placeholder)...")
    final_input_jsonl_path = "path/to/your/input_examples.jsonl"  # Placeholder
    ann_index_path = "path/to/your/ann_index.ann"  # Placeholder

    # Example: if args.input_examples_file:
    #   final_input_jsonl_path = args.input_examples_file
    #   # Logic for build_ann_index.py
    # elif campaign_config.get('source_data_specification', {}).get('type') == 'jsonl_file':
    #    final_input_jsonl_path = campaign_config['source_data_specification']['path']
    #    # Logic for prelabeling if campaign_config.get('pre_labeling',{}).get('use_script_04')
    #    # Logic for build_ann_index.py

    # Ensure final_input_jsonl_path is accessible from workspace root if not absolute
    if not os.path.isabs(final_input_jsonl_path) and workspace_root:
        final_input_jsonl_path_abs = os.path.join(
            workspace_root, final_input_jsonl_path)
        # Ensure the path exists, create dummy if needed for placeholder to work
        os.makedirs(os.path.dirname(final_input_jsonl_path_abs), exist_ok=True)
        if not os.path.exists(final_input_jsonl_path_abs):
            with open(final_input_jsonl_path_abs, 'w') as f:
                f.write("{}")  # Dummy jsonl line
        final_input_jsonl_path = final_input_jsonl_path_abs

    # 4. Configure Prodigy UI Labels
    # - Get target_labels_for_ui from campaign_config.
    # - Get full details for these labels from annotation_schema (e.g., for custom Prodigy config).
    print("Configuring Prodigy UI labels (Placeholder)...")
    target_labels = campaign_config.get('target_labels_for_ui', [])
    if not target_labels:
        print("Warning: No target_labels_for_ui specified in campaign config. Prodigy might show all labels from the recipe or schema.")

    # Example of creating a temporary prodigy.json if needed for label colors etc.
    # prodigy_config_content = {"labels": target_labels}
    # For example, if schema contains color info:
    # custom_theme_labels = {}
    # for label_def in annotation_schema.get('labels', []):
    #     if label_def['name'] in target_labels and 'color' in label_def:
    #         custom_theme_labels[label_def['name']] = label_def['color']
    # if custom_theme_labels:
    #     prodigy_config_content["custom_theme"] = {"labels": custom_theme_labels}
    #
    # temp_prodigy_config_path = os.path.join(workspace_root, f"temp_prodigy_config_{prodigy_dataset_name}.json")
    # with open(temp_prodigy_config_path, 'w') as f:
    #     json.dump(prodigy_config_content, f)
    # print(f"Temporary Prodigy config created at {temp_prodigy_config_path}")

    # 5. Launch Prodigy
    # - Construct the prodigy command.
    # - Recipe: prodigy_recipes.prelabel_ent_spans_manual (or make configurable)
    # - Pass relevant args: dataset_name, input_file, port, host.
    # - If a temporary prodigy.json was created, pass it via PRODIGY_CONFIG_PATH environment variable or --config.
    print("Launching Prodigy (Placeholder)...")
    prodigy_command = [
        sys.executable,  # Path to current python interpreter
        "-m", "prodigy",
        prodigy_dataset_name,
        final_input_jsonl_path,  # This needs to be the actual path to the prepared data
        "--recipe", "prelabel_ent_spans_manual",  # Make this configurable?
        "--port", args.prodigy_port,
        "--host", args.prodigy_host,
        # If using a temporary config:
        # "--config", temp_prodigy_config_path
    ]
    # If you have specific labels for the recipe (e.g. your recipe takes --labels arg)
    # prodigy_command.extend(["--label", ",".join(target_labels)]) # Example if recipe accepts comma-separated labels

    print(f"Executing Prodigy command: {' '.join(prodigy_command)}")

    # For actual execution:
    # try:
    #     # Set PRODIGY_LOGGING=verbose for more output if needed
    #     env = os.environ.copy()
    #     # if temp_prodigy_config_path:
    #     #    env["PRODIGY_CONFIG_PATH"] = temp_prodigy_config_path
    #     process = subprocess.Popen(prodigy_command, env=env)
    #     print(f"Prodigy started with PID: {process.pid}. Session: {prodigy_dataset_name}")
    #     # We might not wait for it here if we want to just launch and update registry
    #     # process.wait() # This would block until Prodigy exits
    # except FileNotFoundError:
    #     print("Error: Prodigy command not found. Make sure Prodigy is installed and in your PATH.")
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"Error launching Prodigy: {e}")
    #     sys.exit(1)

    # 6. Update Dataset Registry
    # - Add/update an entry in dataset_registry.csv.
    print("Updating dataset registry...")
    dataset_entry = {
        'prodigy_dataset_name': prodigy_dataset_name,
        'campaign_name': campaign_config['campaign_name'],
        'creation_date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'status': 'annotation_in_progress',  # Or 'defined' if Prodigy not launched yet
        'input_examples_file': os.path.relpath(final_input_jsonl_path, workspace_root) if workspace_root else final_input_jsonl_path,
        # Adjust if ann_index_path is not always set
        'ann_index_file': os.path.relpath(ann_index_path, workspace_root) if workspace_root and ann_index_path else ann_index_path,
        'prodigy_export_file': 'N/A',  # Will be filled after export
        'annotation_schema_version': schema_version,
        'description': f"Iteration {args.iteration_number} ({args.iteration_description}) for campaign '{campaign_config['campaign_name']}'",
        'source_prodigy_datasets_merged': 'N/A'
    }
    registry_file_path = os.path.join(
        workspace_root, "docs", "dataset_registry.csv")
    update_dataset_registry(registry_file_path, dataset_entry)

    print(
        f"\nProdigy campaign setup complete for dataset: {prodigy_dataset_name}")
    print(f"To start annotation, ensure Prodigy is running with the above command or launch it manually if needed.")
    # if temp_prodigy_config_path and os.path.exists(temp_prodigy_config_path):
    #    print(f"NOTE: A temporary Prodigy config was created at {temp_prodigy_config_path}. It can be removed after the session.")


if __name__ == "__main__":
    main()
