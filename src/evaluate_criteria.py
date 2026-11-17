"""Step 6b: measure whether the criteria check improves the results.

Run from the project folder, after review.py and evaluate_trec.py work:
    export GEMINI_API_KEY="your key"
    python src/evaluate_criteria.py --tune     (while improving the check)
    python src/evaluate_criteria.py            (the final test)

The data is split in two, so the check is never tested on the same patients
it was adjusted on:
  - Tuning: TREC patients 11 to 30. Look at these results as often as you
    like and change the instructions in check_criteria.py to do better.
  - Test:   the 50 labelled diabetes pairs in results/review.csv, and TREC
    patients 1 to 10. Run this only when the instructions are final.

For each TREC patient the matcher's top 20 trials are checked and re-ordered
in two ways, and each new top 10 is compared with the old top 10 using the
expert labels:
  - by verdict: eligible first, excluded last;
  - by failed criteria: trials about the patient's condition first, then
    fewest failed criteria first. This is the main result. It was chosen on
    the tuning patients, before the test results were seen.
In both, trials that score the same keep the matcher's order.

The script first counts the requests it needs and asks before sending them.
Answers already saved are not requested again.
"""

import argparse

from check_criteria import MODEL, load_verdicts, patient_text, saved_or_new_verdict, tokens_used
from check_trec import load_topics, patient_age, patient_sex
from evaluate_trec import TOP_K, load_labels, print_row, score
from evaluate_trec import collection as trec_collection
from match import collection as diabetes_collection
from match import eligibility_filter, model
from review import REVIEW_FILE, read_rows

CANDIDATES = 20  # how many of the matcher's top trials are checked per TREC patient
TEST_PATIENTS = range(1, 11)  # TREC patients 1 to 10
TUNING_PATIENTS = range(11, 31)  # TREC patients 11 to 30

# Where each verdict goes when the list is re-ordered: lower numbers first.
VERDICT_ORDER = {"eligible": 0, "not_relevant": 1, "not_checked": 1, "excluded": 2}


def by_verdict(verdict):
    """Sort key: eligible first, then not relevant or not checked, excluded last."""
    return VERDICT_ORDER[verdict["verdict"]]


def by_failed_criteria(verdict):
    """Sort key: trials about the patient's condition first, then fewest failed criteria."""
    about_patient = verdict["verdict"] in ("eligible", "excluded")
    return (0 if about_patient else 1, len(verdict["failed_criteria"]))


def diabetes_jobs():
    """Build one job per labelled pair: (key, patient text, trial text, human label)."""
    rows = read_rows(REVIEW_FILE)
    stored = diabetes_collection.get(ids=list({row["nct_id"] for row in rows}))
    trial_texts = dict(zip(stored["ids"], stored["documents"]))

    jobs = []
    for row in rows:
        patient = patient_text(row["summary"], float(row["age"]), row["sex"])
        key = f"diabetes|{row['patient']}|{row['nct_id']}"
        jobs.append((key, patient, trial_texts[row["nct_id"]], int(row["human_label"])))
    return jobs


def trec_jobs(patient_numbers):
    """For the given TREC patients, build one job per trial in the matcher's top 20."""
    topics = load_topics()
    jobs_by_patient = {}
    for number in patient_numbers:
        number = str(number)
        text = topics[number]
        where = eligibility_filter(patient_age(text), patient_sex(text))
        query_embedding = model.encode([text], normalize_embeddings=True)
        results = trec_collection.query(
            query_embeddings=query_embedding,
            n_results=CANDIDATES,
            where=where,
            include=["documents"],
        )
        jobs_by_patient[number] = [
            (f"trec|{number}|{nct_id}", text, trial_text, nct_id)
            for nct_id, trial_text in zip(results["ids"][0], results["documents"][0])
        ]
    return jobs_by_patient


def report_diabetes(jobs, verdicts):
    """Compare the model's verdicts with the human labels."""
    agreed = 0
    caught = 0
    wrongly_rejected = 0
    suitable_total = sum(label for _, _, _, label in jobs)
    for key, _, _, human_label in jobs:
        verdict = verdicts[key]
        model_label = 1 if verdict["verdict"] == "eligible" else 0
        if model_label == human_label:
            agreed += 1
        else:
            print(f"  Disagreement: {key} human {human_label}, check says {verdict['verdict']} {verdict['failed_criteria']}")
        if human_label == 0 and model_label == 0:
            caught += 1
        if human_label == 1 and model_label == 0:
            wrongly_rejected += 1

    unsuitable_total = len(jobs) - suitable_total
    print(f"A. Diabetes pairs: check and human label agree on {agreed} of {len(jobs)}")
    print(f"   Unsuitable trials the check caught:    {caught} of {unsuitable_total}")
    print(f"   Suitable trials the check rejected:    {wrongly_rejected} of {suitable_total}")
    print()


def report_trec(jobs_by_patient, verdicts, show_mistakes):
    """Score the matcher's top 10 before and after re-ordering by verdict."""
    labels = load_labels()
    before = []
    after_verdict = []
    after_failed = []
    eligible_total = 0
    eligible_rejected = 0
    excluded_total = 0
    excluded_caught = 0
    for number, jobs in jobs_by_patient.items():
        original = [nct_id for _, _, _, nct_id in jobs]
        # sorted() keeps the original order among trials with the same sort key.
        reordered_by_verdict = sorted(original, key=lambda nct_id: by_verdict(verdicts[f"trec|{number}|{nct_id}"]))
        reordered_by_failed = sorted(original, key=lambda nct_id: by_failed_criteria(verdicts[f"trec|{number}|{nct_id}"]))
        before.append(score(original[:TOP_K], labels[number]))
        after_verdict.append(score(reordered_by_verdict[:TOP_K], labels[number]))
        after_failed.append(score(reordered_by_failed[:TOP_K], labels[number]))

        # Compare each verdict with the expert label (2 = eligible, 1 = excluded).
        for nct_id in original:
            verdict = verdicts[f"trec|{number}|{nct_id}"]
            expert_label = labels[number].get(nct_id)
            if expert_label == 2:
                eligible_total += 1
                if verdict["verdict"] != "eligible":
                    eligible_rejected += 1
                    if show_mistakes:
                        print(f"  Rejected an eligible trial: patient {number}, {nct_id}: {verdict['verdict']} {verdict['failed_criteria']}")
            if expert_label == 1:
                excluded_total += 1
                if verdict["verdict"] == "excluded":
                    excluded_caught += 1

    print(f"B. TREC, top {TOP_K} for {len(jobs_by_patient)} patients")
    print(f"{'group':<22} {'patients':>8} {'NDCG@10':>8} {'eligible':>9} {'excluded':>9}")
    print_row("Matcher only", before)
    print_row("Check, by verdict", after_verdict)
    print_row("Check, by failed", after_failed)
    print(f"   Expert-excluded trials the check caught:   {excluded_caught} of {excluded_total}")
    print(f"   Expert-eligible trials the check rejected: {eligible_rejected} of {eligible_total}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the criteria check.")
    parser.add_argument("--tune", action="store_true", help="Use the tuning patients instead of the test data")
    args = parser.parse_args()

    if args.tune:
        diabetes = []
        trec = trec_jobs(TUNING_PATIENTS)
    else:
        diabetes = diabetes_jobs()
        trec = trec_jobs(TEST_PATIENTS)
    all_jobs = diabetes + [job for jobs in trec.values() for job in jobs]

    verdicts = load_verdicts()
    new_requests = sum(1 for job in all_jobs if job[0] not in verdicts)
    print(f"{len(all_jobs)} verdicts needed, {new_requests} not yet saved.")
    if new_requests > 0:
        answer = input(f"Send {new_requests} requests to {MODEL}? (y/n) ")
        if answer.strip().lower() != "y":
            print("Stopped. Nothing was sent.")
            return

    for count, (key, patient, trial_text, _) in enumerate(all_jobs, start=1):
        saved_or_new_verdict(verdicts, key, patient, trial_text)
        print(f"\rChecked {count} of {len(all_jobs)}", end="")
    print("\n")

    if args.tune:
        print("TUNING DATA (TREC patients 11 to 30)")
    else:
        print("TEST DATA (50 diabetes pairs, TREC patients 1 to 10)")
        report_diabetes(diabetes, verdicts)
    report_trec(trec, verdicts, show_mistakes=args.tune)

    unsupported = sum(verdicts[job[0]].get("unsupported", 0) for job in all_jobs)
    print(f"Answers ignored because their evidence was not in the description: {unsupported}")
    print(f"Tokens used in this run: {tokens_used['input']} in, {tokens_used['output']} out")


if __name__ == "__main__":
    main()
