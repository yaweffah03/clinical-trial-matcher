"""Step 1: download recruiting type 2 diabetes trials from ClinicalTrials.gov.

Run from the project folder:
    python src/fetch_trials.py

Writes data/trials.json: a list with one small dictionary per trial.
"""

import json
from pathlib import Path

import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"
CONDITION = "type 2 diabetes"
STATUS = "RECRUITING"
PAGE_SIZE = 200  # the API allows up to 1000 per request
OUTPUT_FILE = Path("data/trials.json")

# Ask the API for only the fields we use, so each response stays small.
FIELDS = [
    "NCTId",
    "BriefTitle",
    "Condition",
    "BriefSummary",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Sex",
]


def fetch_page(page_token=None):
    """Request one page of results and return it as a Python dictionary."""
    params = {
        "query.cond": CONDITION,
        "filter.overallStatus": STATUS,
        "fields": ",".join(FIELDS),
        "pageSize": PAGE_SIZE,
        "countTotal": "true",
    }
    if page_token is not None:
        params["pageToken"] = page_token

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def simplify(study):
    """Turn one nested API record into a small flat dictionary."""
    protocol = study["protocolSection"]
    identification = protocol["identificationModule"]
    eligibility = protocol.get("eligibilityModule", {})

    return {
        "nct_id": identification["nctId"],
        "title": identification.get("briefTitle", ""),
        "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
        "summary": protocol.get("descriptionModule", {}).get("briefSummary", ""),
        "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
        # Ages are text such as "18 Years". None means the trial sets no limit.
        "min_age": eligibility.get("minimumAge"),
        "max_age": eligibility.get("maximumAge"),
        "sex": eligibility.get("sex"),  # "ALL", "FEMALE" or "MALE"
    }


def fetch_all_trials():
    """Request page after page until the API says there are no more."""
    trials = []
    page_token = None

    while True:
        page = fetch_page(page_token)

        # The total is only included in the first page of results.
        if "totalCount" in page:
            print(f"The API reports {page['totalCount']} matching trials")

        for study in page["studies"]:
            trials.append(simplify(study))
        print(f"Fetched {len(trials)} so far")

        page_token = page.get("nextPageToken")
        if page_token is None:
            break

    return trials


def main():
    trials = fetch_all_trials()

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(trials)} trials to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
