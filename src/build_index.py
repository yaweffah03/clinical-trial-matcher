"""Step 2: embed every trial and store it in a Chroma vector database.

Run from the project folder, after fetch_trials.py:
    python src/build_index.py

Reads data/trials.json and writes the database to the chroma_db/ folder.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

TRIALS_FILE = Path("data/trials.json")
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "trials"
MODEL_NAME = "all-MiniLM-L6-v2"

# Used when a trial sets no age limit, so every trial has a number to filter on.
NO_MIN_AGE = 0
NO_MAX_AGE = 120


def age_in_years(text, default):
    """Convert API text such as "18 Years" or "6 Months" into a number of years."""
    if text is None:
        return default

    number, unit = text.split()
    number = float(number)

    if unit.startswith("Year"):
        return number
    if unit.startswith("Month"):
        return number / 12
    if unit.startswith("Week"):
        return number / 52
    if unit.startswith("Day"):
        return number / 365
    raise ValueError(f"Unknown age unit in: {text}")


def trial_text(trial):
    """Build the single piece of text that represents a trial to the model."""
    conditions = ", ".join(trial["conditions"])
    return (
        f"{trial['title']}\n"
        f"Conditions: {conditions}\n"
        f"{trial['eligibility_criteria']}"
    )


def trial_metadata(trial):
    """Build the facts stored next to each embedding, used later for filtering."""
    return {
        "title": trial["title"],
        "sex": trial["sex"],
        "min_age": age_in_years(trial["min_age"], NO_MIN_AGE),
        "max_age": age_in_years(trial["max_age"], NO_MAX_AGE),
    }


def main():
    with open(TRIALS_FILE, encoding="utf-8") as f:
        trials = json.load(f)

    ids = [trial["nct_id"] for trial in trials]
    texts = [trial_text(trial) for trial in trials]
    metadatas = [trial_metadata(trial) for trial in trials]

    # Turn each text into a list of 384 numbers (its embedding).
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Start from an empty collection so re-running never leaves old trials behind.
    client = chromadb.PersistentClient(path=DB_FOLDER)
    existing = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} trials in {DB_FOLDER}/")


if __name__ == "__main__":
    main()
