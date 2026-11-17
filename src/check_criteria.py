"""Step 6: ask a language model to check a patient against a trial's criteria.

Run from the project folder, after build_index.py:
    export GEMINI_API_KEY="your key"
    python src/check_criteria.py --age 62 --sex FEMALE "Type 2 diabetes for 10 years, on metformin, early kidney disease"

The matcher (match.py) ranks trials by how similar their text is to the
patient. It never reads the criteria. This script adds that step.

The model does not give the verdict. It only answers small questions about
each criterion: does the description show the patient has this, and which
words show it? The verdict is then worked out by ordinary code, in
decide(), using fixed rules:
  - an exclusion criterion rules the patient out only if the description
    shows they have it;
  - an inclusion criterion rules them out only if the description shows they
    do not meet it;
  - anything the description does not mention never rules anyone out;
  - the model's quoted evidence must really appear in the description.

The free tier limits how many requests can be sent, so answers are saved in
the results folder and never requested twice.

This is not a clinical tool. The model can be wrong, and a short patient
description never contains everything a trial asks about.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from google import genai
from pydantic import BaseModel

from match import collection, match_trials

# The model that reads the criteria. Gemini has a free tier. The larger
# "gemini-3.8-flash" and "gemini-3.6-flash" models did not answer within the
# timeout on a free key (14 November 2026), so the smaller Flash-Lite is used.
MODEL = "gemini-3.5-flash-lite"

# One file per model, so changing MODEL never reuses another model's answers.
VERDICTS_FILE = Path(f"results/criterion_checks_{MODEL}.json")

# The free tier allows only so many requests per minute. When the API says
# "too many requests", wait this long and try again, up to this many times.
WAIT_SECONDS = 60
MAX_TRIES = 5

# Give up on a request that has had no answer after this many seconds,
# so the script reports an error instead of waiting forever.
TIMEOUT_SECONDS = 120

INSTRUCTIONS = """\
You compare a patient description with the eligibility criteria of a clinical
trial. This is for a research project that measures how well automatic trial
matching works. It is not used to enrol patients.

First decide "relevant": is the trial about a condition the patient has?

Then go through the criteria and fill in one entry for each:
- every inclusion criterion;
- an exclusion criterion only if the description shows the patient has it.
  Leave out all other exclusion criteria.

For each entry:
- criterion: the criterion, quoted from the trial text.
- kind: "inclusion" or "exclusion".
- about: what the criterion asks for.
    "condition"   a disease or diagnosis
    "treatment"   a medicine or therapy the patient takes or took
    "procedure"   an operation or other past procedure
    "measurement" a number or test result, such as HbA1c, eGFR, BMI, blood
                  pressure, a dose, an age, or how long something has lasted
    "other"       anything else, such as consent, habits, or where the
                  patient is treated
- patient_has: does the description show that the patient has what the
  criterion describes?
    "yes"         the description says so
    "no"          the description says something that contradicts it
    "not_stated"  the description does not say
- evidence: for "yes" or "no", copy the exact words from the patient
  description that show it. For "not_stated", leave it empty.

Be literal. Do not guess a number from other facts. A past problem is not a
current one: a previous ulcer is not a current ulcer. A milder condition is
not a more severe one: mild cognitive impairment is not dementia, and early
kidney disease is not severe kidney disease. If the description gives no
date, do not assume something happened recently.
"""


class CriterionCheck(BaseModel):
    """The model's answer about one criterion."""

    criterion: str
    kind: Literal["inclusion", "exclusion"]
    about: Literal["condition", "treatment", "procedure", "measurement", "other"]
    patient_has: Literal["yes", "no", "not_stated"]
    evidence: str


class TrialCheck(BaseModel):
    """The shape of the whole answer we require from the model."""

    relevant: bool
    criteria: list[CriterionCheck]


# Reads the key from the GEMINI_API_KEY environment variable.
client = genai.Client()

# Running totals, so a script can report how many tokens it used.
tokens_used = {"input": 0, "output": 0}


def ask_model(patient, trial_text):
    """Send one request, waiting and retrying if the free tier's rate limit is hit."""
    for attempt in range(MAX_TRIES):
        try:
            return client.interactions.create(
                model=MODEL,
                system_instruction=INSTRUCTIONS,
                input=f"PATIENT\n{patient}\n\nTRIAL\n{trial_text}",
                # Makes the reply JSON that matches the TrialCheck class.
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": TrialCheck.model_json_schema(),
                },
                # Ask Google not to keep a copy of the request on its servers.
                store=False,
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as error:
            # The API answers "too many requests" with status code 429.
            # Any other error is a real problem, so let it stop the script.
            too_many_requests = getattr(error, "status_code", None) == 429
            if not too_many_requests or attempt == MAX_TRIES - 1:
                raise
            print(f"\nRate limit reached. Waiting {WAIT_SECONDS} seconds...")
            time.sleep(WAIT_SECONDS)


def simplify(text):
    """Lower-case a text and collapse its spaces, so two texts can be compared."""
    return " ".join(text.lower().split())


def is_quoted_from(evidence, patient):
    """Check that the model's evidence really appears in the patient description."""
    evidence = simplify(evidence).strip(" .,;:'\"")
    return evidence != "" and evidence in simplify(patient)


def decide(patient, check):
    """Turn the model's answers about each criterion into one verdict, using fixed rules."""
    if not check["relevant"]:
        return {"verdict": "not_relevant", "failed_criteria": [], "unsupported": 0}

    failed = []
    unsupported = 0
    for item in check["criteria"]:
        answer = item["patient_has"]

        # A "yes" or "no" only counts if its evidence is in the description.
        if answer != "not_stated" and not is_quoted_from(item["evidence"], patient):
            unsupported += 1
            answer = "not_stated"

        if item["kind"] == "exclusion":
            # The patient has something the trial excludes.
            rules_out = answer == "yes"
        else:
            # The description contradicts a requirement. A requirement the
            # description never mentions does not rule the patient out.
            rules_out = answer == "no"

        if rules_out:
            failed.append(item["criterion"])

    verdict = "excluded" if failed else "eligible"
    return {"verdict": verdict, "failed_criteria": failed, "unsupported": unsupported}


def judge(patient, trial_text):
    """Ask the model about one patient and one trial, and return the verdict as a dictionary."""
    response = ask_model(patient, trial_text)
    if response.usage is not None:
        tokens_used["input"] += response.usage.total_input_tokens or 0
        tokens_used["output"] += response.usage.total_output_tokens or 0

    # No usable answer: the model declined, or ran out of room.
    if response.status != "completed" or not response.output_text:
        return {"verdict": "not_checked", "failed_criteria": [], "unsupported": 0, "check": None}

    # Check the reply really has the shape of a TrialCheck, then make it a dictionary.
    check = TrialCheck.model_validate_json(response.output_text).model_dump()
    verdict = decide(patient, check)
    # Keep the model's raw answers too, so decide() can be re-run without new requests.
    verdict["check"] = check
    return verdict


def load_verdicts():
    """Read the saved verdicts, or start with none."""
    if VERDICTS_FILE.exists():
        with open(VERDICTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_verdicts(verdicts):
    VERDICTS_FILE.parent.mkdir(exist_ok=True)
    with open(VERDICTS_FILE, "w", encoding="utf-8") as f:
        json.dump(verdicts, f, indent=1)


def saved_or_new_verdict(verdicts, key, patient, trial_text):
    """Return the saved verdict for this key, asking the model only if there is none."""
    if key not in verdicts:
        verdicts[key] = judge(patient, trial_text)
        # Save after every request, so stopping the script loses nothing.
        save_verdicts(verdicts)
    elif verdicts[key]["check"] is not None:
        # Work the verdict out again from the saved answers, so a change to
        # the rules in decide() takes effect without asking the model again.
        check = verdicts[key]["check"]
        verdicts[key] = decide(patient, check)
        verdicts[key]["check"] = check
    return verdicts[key]


def patient_text(summary, age, sex):
    """Put age and sex in front of the summary, because the model needs them too."""
    return f"Age {age:g}, {sex.lower()}. {summary}"


def main():
    parser = argparse.ArgumentParser(description="Check a patient against the criteria of their top trials.")
    parser.add_argument("summary", help="Free-text description of the patient")
    parser.add_argument("--age", type=float, required=True, help="Age in years")
    parser.add_argument("--sex", choices=["FEMALE", "MALE"], required=True)
    args = parser.parse_args()

    patient = patient_text(args.summary, args.age, args.sex)
    matches = match_trials(args.summary, args.age, args.sex, top_n=5)

    # The full trial text was stored in Chroma as each trial's "document".
    nct_ids = [match["nct_id"] for match in matches]
    stored = collection.get(ids=nct_ids)
    trial_texts = dict(zip(stored["ids"], stored["documents"]))

    verdicts = load_verdicts()
    for rank, match in enumerate(matches, start=1):
        nct_id = match["nct_id"]
        key = f"demo|{patient}|{nct_id}"
        print(f"Checking trial {rank} of {len(matches)}...", flush=True)
        verdict = saved_or_new_verdict(verdicts, key, patient, trial_texts[nct_id])
        print(f"{rank}. {nct_id}  {match['title']}")
        print(f"   {verdict['verdict'].upper()}")
        for criterion in verdict["failed_criteria"]:
            print(f"   - fails: {criterion}")


if __name__ == "__main__":
    main()
