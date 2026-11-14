"""Step 5d: embed the TREC trials and store them in Chroma.

Run from the project folder, after fetch_trec_ages.py:
    python src/build_trec_index.py

This does for the TREC trials exactly what build_index.py does for the
diabetes trials: the same model, the same text (title, conditions and
eligibility criteria) and the same metadata (sex, minimum and maximum age).
The trials go into a separate collection, so the diabetes index is untouched.

There are about 26,000 trials, so this takes several minutes.
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from build_index import DB_FOLDER, MODEL_NAME, trial_metadata, trial_text
from fetch_trec import CORPUS_FILE
from fetch_trec_ages import AGES_FILE

TREC_COLLECTION_NAME = "trec_trials"
ADD_BATCH_SIZE = 5000  # Chroma limits how many items can be added at once

# Used for trials the registry no longer lists: no age limits, all sexes.
NO_LIMITS = {"min_age": None, "max_age": None, "sex": "ALL"}


def load_trials():
    """Read the TREC trials into the same form as data/trials.json."""
    with open(AGES_FILE, encoding="utf-8") as f:
        ages = json.load(f)

    trials = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            details = record["metadata"]
            limits = ages.get(record["_id"], NO_LIMITS)
            trials.append(
                {
                    "nct_id": record["_id"],
                    "title": record["title"],
                    "conditions": details["diseases_list"],
                    "eligibility_criteria": (
                        f"{details['inclusion_criteria']}\n"
                        f"Exclusion criteria{details['exclusion_criteria']}"
                    ),
                    "min_age": limits["min_age"],
                    "max_age": limits["max_age"],
                    "sex": limits["sex"],
                }
            )
    return trials


def main():
    trials = load_trials()

    ids = [trial["nct_id"] for trial in trials]
    texts = [trial_text(trial) for trial in trials]
    metadatas = [trial_metadata(trial) for trial in trials]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Start from an empty collection so re-running never leaves old trials behind.
    client = chromadb.PersistentClient(path=DB_FOLDER)
    existing = [collection.name for collection in client.list_collections()]
    if TREC_COLLECTION_NAME in existing:
        client.delete_collection(TREC_COLLECTION_NAME)
    collection = client.create_collection(
        TREC_COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    for start in range(0, len(ids), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"Stored {collection.count()} trials in {DB_FOLDER}/")


if __name__ == "__main__":
    main()
