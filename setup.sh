# Create the main project directory
mkdir kexp_knowledge_base
cd kexp_knowledge_base

# Create subdirectories
mkdir -p data/raw_kexp_data
mkdir -p data/gazetteers
mkdir -p data/prodigy_exports
mkdir -p data/spacy_training_data
mkdir models
mkdir scripts
mkdir notebooks
mkdir -p src/kexp_processing
mkdir configs

# Create some initial files (you'll populate these later)
touch data/raw_kexp_data/.gitkeep  # .gitkeep makes empty dirs trackable by git
touch data/gazetteers/artists.txt
touch data/gazetteers/songs.txt
touch data/gazetteers/albums.txt
touch data/gazetteers/venues.txt
touch data/gazetteers/festivals.txt
touch data/prodigy_exports/.gitkeep
touch data/spacy_training_data/.gitkeep
touch models/.gitkeep
touch scripts/01_extract_db_terms.py
touch scripts/02_compile_gazetteers.py
touch scripts/03_create_matchers.py
touch scripts/04_prelabel_for_prodigy.py
touch scripts/05_train_spacy_model.py
touch scripts/06_extract_relations.py
touch notebooks/data_exploration.ipynb
touch src/__init__.py
touch src/kexp_processing/__init__.py
touch src/kexp_processing/entity_extractors.py
touch configs/ner_config.cfg
touch .gitignore
touch README.md
touch requirements.txt

echo "Project directory structure for 'kexp_knowledge_base' created."
echo "Next steps:"
echo "1. cd kexp_knowledge_base"
echo "2. python3 -m venv venv  # Create a virtual environment"
echo "3. source venv/bin/activate # Activate it (on Linux/macOS)"
echo "4. pip install -r requirements.txt # (After you add dependencies to requirements.txt)"
echo "5. git init && git add . && git commit -m 'Initial project structure' # Initialize Git"

