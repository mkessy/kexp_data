import argparse
import os
import sys
import yaml
import re  # For filename version checking

# --- Utility to load schema (can be shared or moved to a common util) ---


def load_annotation_schema_for_validation(schema_path):
    """Loads a schema YAML for validation purposes."""
    if not os.path.exists(schema_path):
        # This function is used by campaign validation, so schema not existing is a validation error itself.
        return None, f"Annotation schema file not found at {schema_path}"
    with open(schema_path, 'r') as f:
        try:
            schema = yaml.safe_load(f)
            return schema, None
        except yaml.YAMLError as e:
            return None, f"Error parsing YAML schema file {schema_path}: {e}"

# --- Validation Functions ---


def validate_schema_file(file_path, workspace_root):
    """Validates a single kexp_annotation_schema_vX.Y.yaml file."""
    errors = []
    warnings = []

    # Validate filename convention
    filename = os.path.basename(file_path)
    if not re.match(r"kexp_annotation_schema_v\d+\.\d+\.yaml", filename):
        errors.append(
            f"Filename '{filename}' does not match expected pattern 'kexp_annotation_schema_vX.Y.yaml'.")
        # If filename is wrong, can't reliably get version from it for comparison
        schema_version_from_filename = None
    else:
        schema_version_from_filename = filename.replace(
            "kexp_annotation_schema_v", "").replace(".yaml", "")

    schema_data, err_msg = load_annotation_schema_for_validation(file_path)
    if err_msg:
        errors.append(err_msg)
        return errors, warnings  # Cannot proceed if file can't be loaded/parsed

    # Check top-level keys
    required_top_keys = ['schema_version', 'date',
                         'description', 'general_annotation_rules', 'labels']
    for key in required_top_keys:
        if key not in schema_data:
            errors.append(f"Missing required top-level key: '{key}'.")

    if 'schema_version' in schema_data and schema_version_from_filename:
        if str(schema_data['schema_version']) != schema_version_from_filename:
            errors.append(
                f"Schema version in content ('{schema_data['schema_version']}') does not match version in filename ('{schema_version_from_filename}').")

    if not isinstance(schema_data.get('general_annotation_rules'), list):
        errors.append("'general_annotation_rules' must be a list.")

    labels_data = schema_data.get('labels')
    if not isinstance(labels_data, list):
        errors.append("'labels' must be a list.")
    else:
        label_names = set()
        for i, label_def in enumerate(labels_data):
            if not isinstance(label_def, dict):
                errors.append(
                    f"Label definition at index {i} is not a dictionary.")
                continue

            required_label_keys = [
                'name', 'description', 'guidance', 'examples']
            for key in required_label_keys:
                if key not in label_def:
                    errors.append(
                        f"Label '{label_def.get('name', f'at index {i}')}' missing required key: '{key}'.")

            label_name = label_def.get('name')
            if label_name:
                if label_name in label_names:
                    errors.append(
                        f"Duplicate label name found: '{label_name}'.")
                label_names.add(label_name)

            examples = label_def.get('examples')
            if not isinstance(examples, list):
                errors.append(
                    f"Label '{label_name}' has 'examples' that is not a list.")
            else:
                for ex_idx, example in enumerate(examples):
                    if not isinstance(example, dict):
                        errors.append(
                            f"Label '{label_name}', example at index {ex_idx} is not a dictionary.")
                        continue
                    required_example_keys = ['text', 'results', 'reason']
                    for ex_key in required_example_keys:
                        if ex_key not in example:
                            errors.append(
                                f"Label '{label_name}', example at index {ex_idx} missing key: '{ex_key}'.")
                    if 'results' in example and not isinstance(example['results'], list):
                        errors.append(
                            f"Label '{label_name}', example at index {ex_idx} has 'results' that is not a list.")
    return errors, warnings


def validate_campaign_config_file(file_path, workspace_root):
    """Validates a single campaign configuration YAML file."""
    errors = []
    warnings = []

    campaign_data, err_msg = load_annotation_schema_for_validation(
        file_path)  # Reusing loader
    if err_msg:
        errors.append(err_msg)
        return errors, warnings

    required_keys = ['campaign_name', 'description', 'annotation_schema_version',
                     'target_labels_for_ui', 'source_data_specification']
    for key in required_keys:
        if key not in campaign_data:
            errors.append(f"Missing required key: '{key}'.")

    schema_version = campaign_data.get('annotation_schema_version')
    if schema_version:
        schema_file_name = f"kexp_annotation_schema_v{schema_version}.yaml"
        # Construct schema path relative to workspace_root, assuming campaign_configs are in config/
        schema_path = os.path.join(
            workspace_root, "config", "schemas", schema_file_name)
        referenced_schema, schema_err = load_annotation_schema_for_validation(
            schema_path)
        if schema_err or not referenced_schema:
            errors.append(
                f"Referenced annotation_schema_version '{schema_version}' (file: {schema_path}) could not be loaded or is invalid: {schema_err if schema_err else 'File not found or empty.'}")
        else:
            # Check if all target_labels_for_ui are defined in the referenced schema
            target_labels = campaign_data.get('target_labels_for_ui')
            if isinstance(target_labels, list) and referenced_schema.get('labels'):
                defined_label_names = {lbl['name']
                                       for lbl in referenced_schema['labels']}
                for target_label in target_labels:
                    if target_label not in defined_label_names:
                        errors.append(
                            f"Target label '{target_label}' in campaign config is not defined in schema v{schema_version}.")
            elif not isinstance(target_labels, list):
                errors.append("'target_labels_for_ui' must be a list.")
    else:
        errors.append(
            "'annotation_schema_version' is missing or empty, cannot validate target labels against a schema.")

    source_spec = campaign_data.get('source_data_specification')
    if not isinstance(source_spec, dict):
        errors.append("'source_data_specification' must be a dictionary.")
    # Basic check
    elif 'type' not in source_spec or 'path' not in source_spec.get('type', 'jsonl_file'):
        # More specific checks based on type could be added here
        pass  # Allowing flexibility for now as per plan

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate annotation schema and campaign configuration files.")
    parser.add_argument("--file-type", required=True,
                        choices=['schema', 'campaign_conf'], help="Type of file to validate.")
    parser.add_argument("--path", required=True,
                        help="Path to a specific YAML file or a directory containing them.")
    parser.add_argument("--workspace-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__),
                        '..', '..')), help="Path to the workspace root (default: two levels up from scripts/utilities).")

    args = parser.parse_args()

    all_errors = []
    all_warnings = []
    files_to_validate = []

    if os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for file in files:
                if file.endswith('.yaml') or file.endswith('.yml'):
                    files_to_validate.append(os.path.join(root, file))
    elif os.path.isfile(args.path):
        files_to_validate.append(args.path)
    else:
        print(
            f"Error: Path {args.path} is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    if not files_to_validate:
        print(f"No YAML files found to validate in {args.path}.")
        sys.exit(0)

    print(
        f"Validating {len(files_to_validate)} file(s) of type '{args.file_type}'...")

    for file_path in files_to_validate:
        print(f"-- Validating: {file_path}")
        errors, warnings = [], []
        if args.file_type == 'schema':
            errors, warnings = validate_schema_file(
                file_path, args.workspace_root)
        elif args.file_type == 'campaign_conf':
            errors, warnings = validate_campaign_config_file(
                file_path, args.workspace_root)

        for err in errors:
            print(f"  ERROR: {err}")
            all_errors.append(err)
        for warn in warnings:
            print(f"  WARNING: {warn}")
            all_warnings.append(warn)
        if not errors and not warnings:
            print("  OK")

    if all_errors:
        print(
            f"\nValidation finished with {len(all_errors)} error(s) and {len(all_warnings)} warning(s).", file=sys.stderr)
        sys.exit(1)
    elif all_warnings:
        print(
            f"\nValidation finished with {len(all_warnings)} warning(s) and 0 errors.")
        sys.exit(0)
    else:
        print("\nValidation finished successfully. All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
