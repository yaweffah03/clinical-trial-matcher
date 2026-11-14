"""Step 5e: score the matcher against the TREC 2021 expert labels.

Run from the project folder, after build_trec_index.py:
    python src/evaluate_trec.py

For each of the 75 TREC patients, the matcher returns its top 10 trials,
once with no filter and once with the age and sex filter. Experts labelled
patient-trial pairs: 2 = eligible, 1 = excluded, 0 = not relevant.

Three scores are printed:
  - NDCG@10:  ranking quality from 0 to 1. Higher labels near the top score more.
  - eligible: the share of the top 10 that experts labelled eligible (2).
  - excluded: the share of the top 10 that experts labelled excluded (1).
"""

import math

import chromadb

from build_index import DB_FOLDER
from build_trec_index import TREC_COLLECTION_NAME
from check_trec import load_topics, patient_age, patient_sex
from fetch_trec import QRELS_FILE
from match import eligibility_filter, model

TOP_K = 10

collection = chromadb.PersistentClient(path=DB_FOLDER).get_collection(TREC_COLLECTION_NAME)


def load_labels():
    """Read the expert labels into a dictionary: patient -> {trial ID: label}."""
    labels = {}
    with open(QRELS_FILE, encoding="utf-8") as f:
        for line in f:
            patient, _, nct_id, label = line.split()
            labels.setdefault(patient, {})[nct_id] = int(label)
    return labels


def search(text, where=None):
    """Return the IDs of the top 10 trials for a patient description."""
    query_embedding = model.encode([text], normalize_embeddings=True)
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K, where=where)
    return results["ids"][0]


def dcg(labels):
    """Discounted cumulative gain: each label counts for less the lower it is ranked."""
    return sum(label / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))


def ndcg(ranked_labels, all_labels):
    """DCG of our ranking divided by the DCG of the best possible ranking."""
    best = sorted(all_labels, reverse=True)[:TOP_K]
    return dcg(ranked_labels) / dcg(best)


def score(nct_ids, patient_labels):
    """Score one top-10 list against the expert labels for that patient."""
    # A trial the experts did not judge for this patient counts as 0.
    ranked_labels = [patient_labels.get(nct_id, 0) for nct_id in nct_ids]
    return {
        "ndcg": ndcg(ranked_labels, patient_labels.values()),
        "eligible": ranked_labels.count(2) / TOP_K,
        "excluded": ranked_labels.count(1) / TOP_K,
    }


def age_group(age):
    if age < 18:
        return "Age under 18"
    if age < 65:
        return "Age 18 to 64"
    return "Age 65 and over"


def print_row(label, scores):
    """Print the average of each score over a list of patients."""
    count = len(scores)
    print(
        f"{label:<22} {count:>8} "
        f"{sum(s['ndcg'] for s in scores) / count:>8.3f} "
        f"{sum(s['eligible'] for s in scores) / count:>9.0%} "
        f"{sum(s['excluded'] for s in scores) / count:>9.0%}"
    )


def main():
    topics = load_topics()
    labels = load_labels()

    plain_scores = []
    filtered_scores = []
    filtered_by_sex = {}
    filtered_by_age = {}
    eligible_total = 0
    eligible_kept = 0

    for number, text in topics.items():
        age = patient_age(text)
        sex = patient_sex(text)
        where = eligibility_filter(age, sex)

        plain_scores.append(score(search(text), labels[number]))
        filtered = score(search(text, where), labels[number])
        filtered_scores.append(filtered)
        filtered_by_sex.setdefault(sex, []).append(filtered)
        filtered_by_age.setdefault(age_group(age), []).append(filtered)

        # Check the filter itself: it should never remove a trial that the
        # experts said this patient is eligible for.
        eligible_ids = [nct_id for nct_id, label in labels[number].items() if label == 2]
        if eligible_ids:
            eligible_total += len(collection.get(ids=eligible_ids)["ids"])
            eligible_kept += len(collection.get(ids=eligible_ids, where=where)["ids"])

    print(f"{'group':<22} {'patients':>8} {'NDCG@10':>8} {'eligible':>9} {'excluded':>9}")
    print_row("No filter", plain_scores)
    print_row("Age and sex filter", filtered_scores)
    print()
    print("With the age and sex filter:")
    for sex in sorted(filtered_by_sex):
        print_row(sex, filtered_by_sex[sex])
    print()
    for group in sorted(filtered_by_age):
        print_row(group, filtered_by_age[group])
    print()
    removed = eligible_total - eligible_kept
    print(
        f"Expert-eligible trials removed by the filter: "
        f"{removed} of {eligible_total} ({removed / eligible_total:.1%})"
    )


if __name__ == "__main__":
    main()
