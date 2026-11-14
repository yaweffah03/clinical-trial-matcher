"""Step 3: find the trials that best match a patient summary.

Run from the project folder, after build_index.py:
    python src/match.py --age 62 --sex FEMALE "Type 2 diabetes for 10 years, on metformin, early kidney disease"

Prints the top 20 trials the patient is old enough, young enough and the right sex for.
"""

import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from build_index import COLLECTION_NAME, DB_FOLDER, MODEL_NAME

TOP_N = 20

# Loaded once, when the file is first run or imported, because loading is slow.
model = SentenceTransformer(MODEL_NAME)
collection = chromadb.PersistentClient(path=DB_FOLDER).get_collection(COLLECTION_NAME)


def eligibility_filter(age, sex):
    """Build the Chroma filter for trials that accept this age and sex."""
    # Keep only trials whose age range includes the patient
    # and that accept the patient's sex ("ALL" trials accept everyone).
    return {
        "$and": [
            {"min_age": {"$lte": age}},
            {"max_age": {"$gte": age}},
            {"sex": {"$in": ["ALL", sex]}},
        ]
    }


def match_trials(summary, age, sex, top_n=TOP_N):
    """Return the top_n most similar trials that accept this age and sex."""
    eligible = eligibility_filter(age, sex)

    # The patient summary must be embedded the same way the trials were.
    query_embedding = model.encode([summary], normalize_embeddings=True)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_n,
        where=eligible,
    )

    # Chroma returns one list per query. We sent one query, so take item [0].
    matches = []
    for nct_id, metadata, distance in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append(
            {
                "nct_id": nct_id,
                "title": metadata["title"],
                "sex": metadata["sex"],
                "min_age": metadata["min_age"],
                "max_age": metadata["max_age"],
                # Cosine distance is 0 for identical meaning, so flip it:
                # a higher similarity means a closer match.
                "similarity": 1 - distance,
            }
        )
    return matches


def main():
    parser = argparse.ArgumentParser(description="Match a patient to clinical trials.")
    parser.add_argument("summary", help="Free-text description of the patient")
    parser.add_argument("--age", type=float, required=True, help="Age in years")
    parser.add_argument("--sex", choices=["FEMALE", "MALE"], required=True)
    args = parser.parse_args()

    matches = match_trials(args.summary, args.age, args.sex)

    for rank, match in enumerate(matches, start=1):
        print(f"{rank:>2}. {match['similarity']:.3f}  {match['nct_id']}  {match['title']}")


if __name__ == "__main__":
    main()
