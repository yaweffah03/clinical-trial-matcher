"""Step 4c: check whether the top 5 trials for each test patient are suitable.

Run from the project folder, after build_index.py:
    python src/review.py

results/review.csv holds the top 5 trials per patient and two sets of labels
(1 = suitable, 0 = not suitable):
  - ai_label     filled in by an AI assistant (Claude), which read each trial's
                 eligibility criteria. ai_reason gives a one-line reason.
  - human_label  filled in by the project author, who read each trial's page
                 on ClinicalTrials.gov.

The script creates the file if it is missing. Otherwise it prints precision
at 5 (the share of top-5 trials judged suitable) for each set of labels, and
how often the two agree.
"""

import csv
from pathlib import Path

from evaluate import PATIENTS
from match import match_trials

REVIEW_FILE = Path("results/review.csv")
TOP_K = 5
COLUMNS = [
    "patient", "age", "sex", "summary", "rank", "nct_id", "title", "link",
    "ai_label", "ai_reason", "human_label",
]


def read_rows(path):
    """Read a CSV file into a list of dictionaries, one per row."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        # Excel saves CSV files with semicolons in some regions, so check
        # the header line to see which separator this file uses.
        header = f.readline()
        f.seek(0)
        delimiter = ";" if ";" in header else ","
        return list(csv.DictReader(f, delimiter=delimiter))


def create_review_file():
    """Write the top 5 trials for every patient, with empty label columns."""
    rows = []
    for number, patient in enumerate(PATIENTS, start=1):
        matches = match_trials(patient["summary"], patient["age"], patient["sex"], top_n=TOP_K)
        for rank, match in enumerate(matches, start=1):
            rows.append(
                {
                    "patient": number,
                    "age": patient["age"],
                    "sex": patient["sex"],
                    "summary": patient["summary"],
                    "rank": rank,
                    "nct_id": match["nct_id"],
                    "title": match["title"],
                    "link": f"https://clinicaltrials.gov/study/{match['nct_id']}",
                    "ai_label": "",
                    "ai_reason": "",
                    "human_label": "",
                }
            )
    REVIEW_FILE.parent.mkdir(exist_ok=True)
    with open(REVIEW_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {REVIEW_FILE}. Fill in the label columns with 1 or 0.")


def is_complete(rows, column):
    """Check that every row has a 1 or 0 in the given label column."""
    return all(row[column].strip() in ("0", "1") for row in rows)


def age_group(age):
    if age < 40:
        return "Age under 40"
    if age < 65:
        return "Age 40 to 64"
    return "Age 65 and over"


def print_precision(label, labels):
    """Print one row: how many labelled trials, and the share judged suitable."""
    print(f"{label:<16} {len(labels):>6} {sum(labels) / len(labels):>10.0%}")


def score(rows, column):
    """Print precision at 5 overall and by group, for one label column."""
    all_labels = []
    top_labels = []
    labels_by_sex = {}
    labels_by_age = {}
    for row in rows:
        label = int(row[column])
        all_labels.append(label)
        if row["rank"] == "1":
            top_labels.append(label)
        labels_by_sex.setdefault(row["sex"], []).append(label)
        labels_by_age.setdefault(age_group(float(row["age"])), []).append(label)

    print(f"Precision at 5, from {column}")
    print(f"{'group':<16} {'trials':>6} {'suitable':>10}")
    print_precision("All patients", all_labels)
    print_precision("Top trial only", top_labels)
    print()
    for sex in sorted(labels_by_sex):
        print_precision(sex, labels_by_sex[sex])
    print()
    for group in sorted(labels_by_age):
        print_precision(group, labels_by_age[group])
    print()


def print_agreement(rows):
    """Print how often the two label columns agree, and where they differ."""
    agreed = 0
    for row in rows:
        if row["ai_label"].strip() == row["human_label"].strip():
            agreed += 1
        else:
            print(
                f"Disagreement: patient {row['patient']}, rank {row['rank']}, {row['nct_id']} "
                f"(AI {row['ai_label']}, human {row['human_label']})"
            )
    print(f"The AI and human labels agree on {agreed} of {len(rows)} trials")


def main():
    if not REVIEW_FILE.exists():
        create_review_file()
        return

    rows = read_rows(REVIEW_FILE)
    if not is_complete(rows, "ai_label"):
        print("Some rows still need a 1 or 0 in 'ai_label'.")
        return
    score(rows, "ai_label")

    if not is_complete(rows, "human_label"):
        print("Some rows still need a 1 or 0 in 'human_label'.")
        return
    score(rows, "human_label")
    print_agreement(rows)


if __name__ == "__main__":
    main()
