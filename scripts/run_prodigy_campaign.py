import argparse
import datetime
import json
import os
import subprocess
import sys
import yaml
import tempfile  # For temporary prodigy config

# Assuming build_ann_index.py and 04_prelabel_for_prodigy.py are in scripts/utilities and scripts/ respectively
# Adjust paths if necessary, or consider making them installable modules
# For now, let's assume they can be called as subprocesses or imported if structured as such.

# Added load_jsonl, save_jsonl for potential use
from scripts.utilities.common_utils import load_annotation_schema, update_dataset_registry, load_jsonl, save_jsonl

# Helper function for running subprocesses


def run_subprocess(command, cwd=None, env=None):
    """Runs a subprocess and handles common errors."""
    print(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error running command: {' '.join(command)}")
            print(f"Return code: {process.returncode}")
            if stdout:
                print(f"stdout:\\n{stdout}")
            if stderr:
                print(f"stderr:\\n{stderr}")
            # sys.exit(1) # Or raise an exception
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout, stderr=stderr)
        print(f"Command finished successfully.")
        if stdout:
            print(f"stdout:\\n{stdout}")
        if stderr:  # Often scripts print info to stderr
            print(f"stderr:\\n{stderr}")
        return stdout, stderr
    except FileNotFoundError:
        print(
            f"Error: Command not found (script missing or not executable?): {command[0]}")
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred while running command {' '.join(command)}: {e}")
        sys.exit(1)


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
                        help="Optional. Path to a specific JSONL file to use as input. Overrides campaign config's source_data_specification and pre-labeling.")
    parser.add_argument("--force-rebuild-ann-index", action="store_true",
                        help="Flag, if input examples are provided, force rebuilding its ANN index.")
    parser.add_argument("--prodigy-port", default="8080",
                        help="Port for the Prodigy server.")
    parser.add_argument("--prodigy-host", default="0.0.0.0",
                        help="Host for the Prodigy server.")
    parser.add_argument("--prodigy-recipe", default="prelabel_ent_spans_manual",
                        help="Prodigy recipe to use (e.g., prelabel_ent_spans_manual from prodigy_recipes).")
    # Add any other standard Prodigy arguments you might want to pass through

    args = parser.parse_args()

    # 0. Define workspace and key script paths
    workspace_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    script_04_prelabel_path = os.path.join(
        workspace_root, "scripts", "04_prelabel_for_prodigy.py")
    script_build_ann_index_path = os.path.join(
        workspace_root, "scripts", "utilities", "build_ann_index.py")

    # Ensure data directories exist as per plan
    source_material_dir = os.path.join(
        workspace_root, "data", "source_material")
    processed_examples_dir = os.path.join(
        workspace_root, "data", "processed_examples")
    os.makedirs(source_material_dir, exist_ok=True)
    os.makedirs(processed_examples_dir, exist_ok=True)

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

    schema_file_name = f"kexp_annotation_schema_v{schema_version}.yaml"
    schema_path = os.path.join(
        workspace_root, "config", "schemas", schema_file_name)

    try:
        annotation_schema = load_annotation_schema(schema_path)
        print(
            f"Loaded annotation schema version: {annotation_schema.get('schema_version')}")
    except (FileNotFoundError, ValueError) as e:
        print(f"CRITICAL ERROR: Could not load annotation schema: {e}")
        sys.exit(1)

    # 2. Determine Prodigy Dataset Name
    print("Determining Prodigy dataset name...")
    prodigy_dataset_name = args.prodigy_dataset_name
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    if not prodigy_dataset_name:
        prodigy_dataset_name = f"{campaign_config['campaign_name']}_asv{schema_version}_iter{args.iteration_number}_{args.iteration_description}_{date_str}"
    print(f"Prodigy dataset name: {prodigy_dataset_name}")

    # Define base for processed example filenames (without extension)
    # e.g., data/processed_examples/artist_biographical_details_v1_asv1.0_iter1_bootstrap_20231115
    processed_examples_base = os.path.join(
        processed_examples_dir, prodigy_dataset_name)

    # 3. Prepare Input Stream & ANN Index
    print("--- Preparing input stream and ANN index ---")

    # This will be the .jsonl fed to Prodigy
    final_input_jsonl_path_for_prodigy = None
    source_data_path = None  # Original source before any pre-labeling

    if args.input_examples_file:
        print(
            f"Using provided input examples file: {args.input_examples_file}")
        if not os.path.exists(args.input_examples_file):
            print(
                f"Error: Provided input_examples_file not found: {args.input_examples_file}")
            sys.exit(1)
        # Copy to processed_examples_dir with a consistent name for clarity and to avoid modifying original
        final_input_jsonl_path_for_prodigy = f"{processed_examples_base}_custom_input.jsonl"
        # shutil.copy(args.input_examples_file, final_input_jsonl_path_for_prodigy) # Use save_jsonl(load_jsonl()) to ensure format
        try:
            custom_data = load_jsonl(args.input_examples_file)
            save_jsonl(final_input_jsonl_path_for_prodigy, custom_data)
            print(
                f"Copied custom input to: {final_input_jsonl_path_for_prodigy}")
        except Exception as e:
            print(
                f"Error processing custom input file {args.input_examples_file}: {e}")
            sys.exit(1)
        # Pre-labeling is skipped if --input-examples-file is used
        print("Pre-labeling is skipped when --input-examples-file is provided.")
    else:
        source_spec = campaign_config.get('source_data_specification', {})
        source_type = source_spec.get('type')
        source_path_relative = source_spec.get('path')

        if not source_path_relative:
            print(
                f"Error: 'source_data_specification.path' not found in campaign config: {args.campaign_config}")
            sys.exit(1)

        # source_data_path is relative to workspace_root/data/source_material/ if not absolute
        # as per plan "data/source_material/unannotated_batch1.jsonl"
        if os.path.isabs(source_path_relative):
            source_data_path = source_path_relative
        else:
            source_data_path = os.path.join(
                source_material_dir, source_path_relative)

        if not os.path.exists(source_data_path):
            print(
                f"Error: Source data file not found: {source_data_path} (from campaign config)")
            sys.exit(1)
        print(f"Using source data from campaign config: {source_data_path}")

        if campaign_config.get('pre_labeling', {}).get('use_script_04', False):
            print(f"Pre-labeling with script: {script_04_prelabel_path}")
            # Output of pre-labeling script will be the input for Prodigy
            final_input_jsonl_path_for_prodigy = f"{processed_examples_base}_prelabeled.jsonl"

            prelabel_cmd = [
                sys.executable, script_04_prelabel_path,
                "--input-file", source_data_path,
                "--output-file", final_input_jsonl_path_for_prodigy,
                # TODO: Add other necessary args for 04_prelabel_for_prodigy.py
                # e.g., --config based on campaign_config.pre_labeling or config/prelabel_config.py
                # For now, assuming it can run with just input/output.
            ]
            # Potentially add prelabel_config.py if specified in campaign_config
            # prelabel_script_config = campaign_config.get('pre_labeling', {}).get('script_04_config_path') # Example
            # if prelabel_script_config:
            #    prelabel_cmd.extend(["--prelabel-config", os.path.join(workspace_root,"config", prelabel_script_config)])

            run_subprocess(prelabel_cmd)
            print(
                f"Pre-labeling complete. Output: {final_input_jsonl_path_for_prodigy}")
        else:
            # No pre-labeling, copy source directly to processed_examples with standard name
            print("No pre-labeling. Using source data directly.")
            final_input_jsonl_path_for_prodigy = f"{processed_examples_base}_source.jsonl"
            # shutil.copy(source_data_path, final_input_jsonl_path_for_prodigy)
            try:
                source_data_content = load_jsonl(source_data_path)
                save_jsonl(final_input_jsonl_path_for_prodigy,
                           source_data_content)
                print(
                    f"Copied source data to: {final_input_jsonl_path_for_prodigy}")
            except Exception as e:
                print(
                    f"Error processing source data file {source_data_path}: {e}")
                sys.exit(1)

    if not final_input_jsonl_path_for_prodigy or not os.path.exists(final_input_jsonl_path_for_prodigy):
        print(
            f"CRITICAL ERROR: final_input_jsonl_path_for_prodigy was not set or file does not exist: {final_input_jsonl_path_for_prodigy}")
        sys.exit(1)

    # Build ANN Index for the final_input_jsonl_path_for_prodigy
    ann_index_path = None
    ann_config = campaign_config.get('ann_bootstrapping')
    if ann_config and ann_config.get('sentence_transformer_model'):
        ann_index_path = final_input_jsonl_path_for_prodigy + ".ann_index"  # Convention
        if args.force_rebuild_ann_index or not os.path.exists(ann_index_path):
            print(
                f"Building ANN index for {final_input_jsonl_path_for_prodigy}...")
            print(f"Output ANN index: {ann_index_path}")
            build_ann_cmd = [
                sys.executable, script_build_ann_index_path,
                "--input-jsonl", final_input_jsonl_path_for_prodigy,
                "--output-index-file", ann_index_path,
                "--model-name", ann_config['sentence_transformer_model']
            ]
            run_subprocess(build_ann_cmd)
            print("ANN index building complete.")
        else:
            print(
                f"ANN index already exists and --force-rebuild-ann-index not set: {ann_index_path}")
    else:
        print(
            "ANN bootstrapping not configured in campaign config, skipping ANN index build.")

    print(f"Prodigy will use input file: {final_input_jsonl_path_for_prodigy}")
    if ann_index_path and os.path.exists(ann_index_path):
        print(f"Using ANN index: {ann_index_path}")
    elif ann_config and ann_config.get('sentence_transformer_model'):
        print(
            f"Warning: ANN index was expected but not found at {ann_index_path}")

    # 4. Configure Prodigy UI Labels
    print("--- Configuring Prodigy UI labels ---")
    target_labels = campaign_config.get('target_labels_for_ui', [])
    if not target_labels:
        print("Warning: No target_labels_for_ui specified in campaign config. Prodigy recipe might show all labels from the schema or its defaults.")

    # For passing rich label details (description, examples) to Prodigy,
    # the recipe needs to support it. One common way is via a --config override file.
    # The `prelabel_ent_spans_manual.py` recipe would need to be adapted to read this.

    prodigy_config_overrides = {}
    # Set labels for the UI (most recipes pick this up)
    if target_labels:
        prodigy_config_overrides["labels"] = target_labels

    # Example of adding custom theme for label colors IF schema supported it (plan section 5.1.4)
    # custom_theme_labels = {}
    # for label_def in annotation_schema.get('labels', []):
    #     if label_def['name'] in target_labels and 'color' in label_def: # Assuming 'color' key in schema
    #         custom_theme_labels[label_def['name']] = label_def['color']
    # if custom_theme_labels:
    #     prodigy_config_overrides["custom_theme"] = {"labels": custom_theme_labels}

    # The plan also mentioned: "From the loaded annotation schema, get full details (description, examples)
    # for these target labels to potentially pass to or display within Prodigy (if recipe supports it)."
    # This is highly recipe-dependent. If `prelabel_ent_spans_manual` can take a path to the main schema file
    # or specific label details, that would be the mechanism.
    # For now, we'll pass target_labels via prodigy_config_overrides["labels"].
    # The recipe itself might load the full schema based on campaign context if it's designed to do so.

    temp_prodigy_config_path = None
    if prodigy_config_overrides:
        # Using NamedTemporaryFile to ensure it's cleaned up
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir=workspace_root) as tmp_f:
            json.dump(prodigy_config_overrides, tmp_f)
            temp_prodigy_config_path = tmp_f.name
        print(
            f"Temporary Prodigy config created with UI labels: {temp_prodigy_config_path}")

    # 5. Launch Prodigy
    print("--- Launching Prodigy ---")
    prodigy_command = [
        sys.executable,
        "-m", "prodigy",
        args.prodigy_recipe,  # Use recipe from args
        prodigy_dataset_name,
        final_input_jsonl_path_for_prodigy,
        "--port", args.prodigy_port,
        "--host", args.prodigy_host,
    ]

    # Add config override if created
    # Prodigy typically merges prodigy.json in CWD, then PRODIGY_CONFIG_PATH, then --config parameter.
    # Using --config for explicitness.
    if temp_prodigy_config_path:
        prodigy_command.extend(["--config", temp_prodigy_config_path])

    # If the recipe *itself* takes a --label argument for filtering (some do, some use config)
    # And if we are NOT using the config override for labels (e.g. recipe handles it differently)
    # if target_labels and not (temp_prodigy_config_path and "labels" in prodigy_config_overrides) :
    #    prodigy_command.extend(["--label", ",".join(target_labels)])

    print(f"Constructed Prodigy command: {' '.join(prodigy_command)}")

    try:
        print(f"Starting Prodigy for dataset: {prodigy_dataset_name}...")
        print("You can stop Prodigy with Ctrl+C in its terminal window when finished.")
        # For actual execution, we run it and don't wait here. User interacts with Prodigy.
        # The Popen call here will just start it.
        # For development, can use subprocess.run with capture_output=True if Prodigy session is short/testable.
        # For long annotation sessions, Popen is more appropriate.

        # We want to see Prodigy's output directly, so not capturing stdout/stderr here with Popen.
        # The run_subprocess helper is more for script-like tools.
        process_env = os.environ.copy()
        # Ensure PRODIGY_LOGGING=basic or similar for useful output without being overly verbose
        if "PRODIGY_LOGGING" not in process_env:
            process_env["PRODIGY_LOGGING"] = "basic"

        process = subprocess.Popen(prodigy_command, env=process_env)
        print(f"Prodigy server starting with PID: {process.pid}...")
        print(f"Access at http://{args.prodigy_host}:{args.prodigy_port}")
        # Not waiting for process.wait() as Prodigy runs until manually stopped.

    except FileNotFoundError:
        print("Error: Prodigy command not found. Make sure Prodigy is installed and in your Python environment where this script is run.")
        # Cleanup temp config
        if temp_prodigy_config_path and os.path.exists(temp_prodigy_config_path):
            os.remove(temp_prodigy_config_path)
        sys.exit(1)
    except Exception as e:
        print(f"Error launching Prodigy: {e}")
        # Cleanup temp config
        if temp_prodigy_config_path and os.path.exists(temp_prodigy_config_path):
            os.remove(temp_prodigy_config_path)
        sys.exit(1)

    # 6. Update Dataset Registry
    print("--- Updating dataset registry ---")
    dataset_entry = {
        'prodigy_dataset_name': prodigy_dataset_name,
        'campaign_name': campaign_config['campaign_name'],
        'creation_date': date_str,  # Use date_str from earlier
        'status': 'annotation_in_progress',  # Prodigy has been launched
        'input_examples_file': os.path.relpath(final_input_jsonl_path_for_prodigy, workspace_root),
        'ann_index_file': os.path.relpath(ann_index_path, workspace_root) if ann_index_path and os.path.exists(ann_index_path) else 'N/A',
        'prodigy_export_file': 'N/A',
        'annotation_schema_version': schema_version,
        'description': f"Iteration {args.iteration_number} ({args.iteration_description}) for campaign \'{campaign_config['campaign_name']}\'",
        'source_prodigy_datasets_merged': 'N/A'
    }
    registry_file_path = os.path.join(
        workspace_root, "docs", "dataset_registry.csv")

    try:
        update_dataset_registry(registry_file_path, dataset_entry)
    except Exception as e:
        print(f"Error updating dataset registry: {e}")
        # Don't necessarily exit, Prodigy might be running.

    print(
        f"\nProdigy campaign '{prodigy_dataset_name}' has been set up and Prodigy server process initiated.")
    print("After completing your annotation session and stopping Prodigy (Ctrl+C), remember to:")
    print("1. Export annotations from Prodigy: ")
    print(
        f"   python -m prodigy db-out {prodigy_dataset_name} > path/to/your_prodigy_exports/{prodigy_dataset_name}_export_YYYYMMDDHHMM.jsonl")
    print("2. Update the 'prodigy_export_file' and 'status' in the dataset registry accordingly.")

    # temp_prodigy_config_path is cleaned up automatically if NamedTemporaryFile(delete=True) (default)
    # If delete=False as used, we should clean it up after Prodigy would theoretically exit.
    # Since Popen doesn't wait, we can't reliably clean it here.
    # Best to use delete=True or have user manage it, or clean up at start of next run.
    # For now, with delete=False, it will persist.
    if temp_prodigy_config_path and os.path.exists(temp_prodigy_config_path):
        print(
            f"NOTE: A temporary Prodigy config was created at {temp_prodigy_config_path}. It can be removed if no longer needed or if Prodigy has fully stopped.")


if __name__ == "__main__":
    main()
