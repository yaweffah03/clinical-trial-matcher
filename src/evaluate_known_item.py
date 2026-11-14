"""Step 4b: a retrieval test where the right answer is known.

Run from the project folder, after build_index.py:
    python src/evaluate_known_item.py

The index was built from each trial's title, conditions and eligibility
criteria. Each trial's summary was never embedded, so it is unseen text.
Here every summary is used as a search query, and we check whether the
matcher finds the trial that summary came from.

Recall@k is the share of trials found within the top k results.
"""

import json

from build_index import TRIALS_FILE, trial_metadata
from match import TOP_N, collection, model

CUTOFFS = [1, 5, 20]


def age_group(metadata):
    """Describe which ages a trial accepts, as one of three groups."""
    if metadata["min_age"] < 18:
        return "accepts under 18"
    if metadata["max_age"] < 65:
        return "adults under 65 only"
    return "accepts 65 and over"


def print_recall(label, ranks):
    """Print one table row: recall at each cutoff for a list of ranks."""
    row = f"{label:<22} {len(ranks):>6}"
    for k in CUTOFFS:
        found = sum(1 for rank in ranks if rank is not None and rank <= k)
        row += f" {found / len(ranks):>9.1%}"
    print(row)


def main():
    with open(TRIALS_FILE, encoding="utf-8") as f:
        trials = json.load(f)

    # Search with every summary at once. No age or sex filter is used here,
    # because this test is only about how well the text search works.
    summaries = [trial["summary"] for trial in trials]
    query_embeddings = model.encode(summaries, normalize_embeddings=True, show_progress_bar=True)
    results = collection.query(query_embeddings=query_embeddings, n_results=TOP_N)

    # For each trial, find the position of its own ID in its results.
    # None means it was not in the top 20 at all.
    all_ranks = []
    ranks_by_sex = {}
    ranks_by_age = {}
    for trial, returned_ids in zip(trials, results["ids"]):
        if trial["nct_id"] in returned_ids:
            rank = returned_ids.index(trial["nct_id"]) + 1
        else:
            rank = None

        metadata = trial_metadata(trial)
        all_ranks.append(rank)
        ranks_by_sex.setdefault(metadata["sex"], []).append(rank)
        ranks_by_age.setdefault(age_group(metadata), []).append(rank)

    header = f"{'group':<22} {'trials':>6}"
    for k in CUTOFFS:
        header += f" {'recall@' + str(k):>9}"
    print(header)
    print_recall("All trials", all_ranks)
    print()
    for sex in sorted(ranks_by_sex):
        print_recall(f"Sex: {sex}", ranks_by_sex[sex])
    print()
    for group in sorted(ranks_by_age):
        print_recall(group, ranks_by_age[group])


if __name__ == "__main__":
    main()
