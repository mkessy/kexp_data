import argparse
import json
import os
import sys

# Try to import optional dependencies, provide guidance if missing
try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def load_jsonl(file_path):
    """Loads a JSONL file, yielding each line as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def main():
    if not HAS_HNSWLIB or not HAS_SENTENCE_TRANSFORMERS:
        error_message = "Missing optional dependencies:"
        if not HAS_HNSWLIB:
            error_message += "\n - hnswlib: Please install with `pip install hnswlib`"
        if not HAS_SENTENCE_TRANSFORMERS:
            error_message += "\n - sentence-transformers: Please install with `pip install sentence-transformers`"
        print(error_message, file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Build an HNSWLib ANN index from a JSONL file containing text examples.")
    parser.add_argument("--input-jsonl", required=True,
                        help="Path to the input JSONL file (must contain a 'text' field).")
    parser.add_argument("--output-index-path", required=True,
                        help="Path to save the HNSWLib index (e.g., data.ann_index).")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                        help="Name of the SentenceTransformer model to use.")
    parser.add_argument("--force", action="store_true",
                        help="Flag to overwrite if index already exists.")
    parser.add_argument("--text-field", default="text",
                        help="Name of the field in the JSONL containing the text to embed.")
    parser.add_argument("--index-max-elements", type=int, default=100000,
                        help="Maximum number of elements for the HNSWLib index.")
    parser.add_argument("--index-ef-construction", type=int,
                        default=200, help="HNSWLib ef_construction parameter.")
    parser.add_argument("--index-m", type=int, default=16,
                        help="HNSWLib M parameter.")
    parser.add_argument("--index-space", default='cosine',
                        help="HNSWLib space ('l2', 'ip', or 'cosine').")

    args = parser.parse_args()

    if os.path.exists(args.output_index_path) and not args.force:
        print(
            f"Error: Output index path {args.output_index_path} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input_jsonl):
        print(
            f"Error: Input JSONL file not found at {args.input_jsonl}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading SentenceTransformer model: {args.model_name}")
    try:
        model = SentenceTransformer(args.model_name)
    except Exception as e:
        print(
            f"Error loading SentenceTransformer model '{args.model_name}': {e}", file=sys.stderr)
        print("Please ensure the model name is correct and you have an internet connection if it needs to be downloaded.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Reading texts from {args.input_jsonl} (using field: '{args.text_field}')")
    texts = []
    original_indices = []  # To map back to original line number or ID if needed later
    for i, item in enumerate(load_jsonl(args.input_jsonl)):
        text_content = item.get(args.text_field)
        if text_content and isinstance(text_content, str):
            texts.append(text_content)
            # Storing original index, could be item['id'] if present
            original_indices.append(i)
        else:
            print(
                f"Warning: Item at line {i+1} missing text field '{args.text_field}' or content is not a string. Skipping.", file=sys.stderr)

    if not texts:
        print("No valid texts found in the input file. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(texts)} texts to embed.")

    print("Encoding texts to embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    num_elements = len(embeddings)
    if num_elements > args.index_max_elements:
        print(
            f"Warning: Number of elements ({num_elements}) exceeds index_max_elements ({args.index_max_elements}). Consider increasing index_max_elements.")
        # Adjust max_elements if you want to proceed, or cap it.
        # For now, we will use the actual number of elements for the index size.

    print(
        f"Initializing HNSWLib index: space='{args.index_space}', dim={dim}, max_elements={num_elements}")
    index = hnswlib.Index(space=args.index_space, dim=dim)
    # Initialize index - the maximum number of elements should be known beforehand
    index.init_index(max_elements=num_elements,
                     ef_construction=args.index_ef_construction, M=args.index_m)

    print("Adding items to index...")
    # Add data to index
    # Data must be numpy or list of lists
    # Using original_indices as IDs
    index.add_items(embeddings, original_indices)

    # Controlling the recall by setting ef:
    # index.set_ef(50) # ef should always be > k

    output_dir = os.path.dirname(args.output_index_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Saving index to {args.output_index_path}")
    index.save_index(args.output_index_path)

    print("ANN index building complete.")
    print(f"Index saved with {num_elements} items.")


if __name__ == "__main__":
    main()
