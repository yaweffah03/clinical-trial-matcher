"""Step 6c: fetch the age limits and sex of every TREC trial.

Run from the project folder, after fetch_trec.py:
    python src/fetch_trec_ages.py

The TREC trial file holds only text. The age and sex filter needs each
trial's minimum age, maximum age and sex, so this script asks the
ClinicalTrials.gov API for those three fields, 200 trials at a time.

Writes data/trec/trial_ages.json: a dictionary from trial ID to
{"min_age": ..., "max_age": ..., "sex": ...}, in the same form as
fetch_trials.py uses (ages kept as text such as "18 Years", or None).
"""

import json

import requests

from fetch_trec import CORPUS_FILE, TREC_FOLDER
from fetch_trials import API_URL

AGES_FILE = TREC_FOLDER / "trial_ages.json"
BATCH_SIZE = 200  # how many trial IDs to ask about in one request


def read_trial_ids():
    """Read the ID of every trial in the TREC trial file."""
    ids = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["_id"])
    return ids


def fetch_batch(nct_ids):
    """Ask the API about one batch of trials and return its list of studies."""
    params = {
        "filter.ids": ",".join(nct_ids),
        "fields": "NCTId,MinimumAge,MaximumAge,Sex",
        "pageSize": BATCH_SIZE,
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()["studies"]


def main():
    nct_ids = read_trial_ids()

    ages = {}
    for start in range(0, len(nct_ids), BATCH_SIZE):
        batch = nct_ids[start : start + BATCH_SIZE]
        for study in fetch_batch(batch):
            protocol = study["protocolSection"]
            eligibility = protocol.get("eligibilityModule", {})
            ages[protocol["identificationModule"]["nctId"]] = {
                "min_age": eligibility.get("minimumAge"),
                "max_age": eligibility.get("maximumAge"),
                "sex": eligibility.get("sex", "ALL"),
            }
        print(f"Fetched {len(ages)} of {len(nct_ids)} trials")

    with open(AGES_FILE, "w", encoding="utf-8") as f:
        json.dump(ages, f)

    missing = len(nct_ids) - len(ages)
    print(f"Saved {len(ages)} trials to {AGES_FILE}. Not found in the registry: {missing}")


if __name__ == "__main__":
    main()
