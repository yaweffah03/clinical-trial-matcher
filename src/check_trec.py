"""Step 6b: three checks on the TREC 2021 data before using it.

Run from the project folder, after fetch_trec.py:
    python src/check_trec.py

  1. Does the trial file contain only trials the experts judged?
  2. Can the age and sex of each patient be pulled out of the free text?
  3. Which patient descriptions mention diabetes?
"""

import json
import re
import xml.etree.ElementTree as ET

from fetch_trec import CORPUS_FILE, QRELS_FILE, TOPICS_FILE

# An age is a number followed by a unit, for example "45-year-old", "32 yo",
# "70 y/o", "5 months old", "3-day-old" or the short form "74M".
AGE_PATTERN = re.compile(r"(\d+)[ -]?(year|yo|y/o|month|day|M\b|F\b)")

# Words that tell us the patient's sex. The first one in the text is used.
SEX_PATTERN = re.compile(
    r"\b(woman|female|girl|she|her|F|man|male|boy|gentleman|he|his|him|M)\b",
    re.IGNORECASE,
)
FEMALE_WORDS = {"woman", "female", "girl", "she", "her", "f"}

# "diabetes", "diabetic", or the abbreviation DM (as in "DM2" or "DM type 2").
DIABETES_PATTERN = re.compile(r"(?i:diabet\w*)|\bDM\w*")


def load_topics():
    """Read the patient descriptions into a dictionary: number -> text."""
    topics = {}
    for topic in ET.parse(TOPICS_FILE).getroot():
        # split() then join() turns line breaks and double spaces into single spaces.
        topics[topic.get("number")] = " ".join(topic.text.split())
    return topics


def patient_age(text):
    """Find the patient's age in years, or None if the text does not give one."""
    match = AGE_PATTERN.search(text)
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "month":
        return number / 12
    if unit == "day":
        return number / 365
    return number


def patient_sex(text):
    """Find the patient's sex as "FEMALE" or "MALE", or None if it is not given."""
    # The short form "74M" or "79F" gives the sex together with the age.
    age_match = AGE_PATTERN.search(text)
    if age_match is not None and age_match.group(2) in ("M", "F"):
        word = age_match.group(2)
    else:
        sex_match = SEX_PATTERN.search(text)
        if sex_match is None:
            return None
        word = sex_match.group(1)
    return "FEMALE" if word.lower() in FEMALE_WORDS else "MALE"


def check_judged_trials():
    """Compare the trials in the trial file with the trials the experts labelled."""
    corpus_ids = set()
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            corpus_ids.add(json.loads(line)["_id"])

    judged_ids = set()
    with open(QRELS_FILE, encoding="utf-8") as f:
        for line in f:
            patient, _, nct_id, label = line.split()
            judged_ids.add(nct_id)

    print("Check 1: does the trial file contain only judged trials?")
    print(f"  Trials in the trial file:          {len(corpus_ids)}")
    print(f"  Trials with at least one label:    {len(judged_ids)}")
    print(f"  In the file but never judged:      {len(corpus_ids - judged_ids)}")
    print(f"  Judged but missing from the file:  {len(judged_ids - corpus_ids)}")
    print()


def check_age_and_sex(topics):
    """Print the age and sex found for every patient."""
    print("Check 2: age and sex pulled from each patient description")
    print(f"  {'no.':>3}  {'age':>5}  {'sex':<6}  start of description")
    missing = 0
    for number, text in topics.items():
        age = patient_age(text)
        sex = patient_sex(text)
        if age is None or sex is None:
            missing += 1
        age_text = "?" if age is None else f"{age:.1f}"
        print(f"  {number:>3}  {age_text:>5}  {sex or '?':<6}  {text[:60]}")
    print(f"  Patients with no age or no sex found: {missing} of {len(topics)}")
    print()


def check_diabetes(topics):
    """Print every mention of diabetes with the words around it."""
    print("Check 3: mentions of diabetes (read each one: it may be a family")
    print("history, or a statement that the patient does NOT have diabetes)")
    patients = set()
    for number, text in topics.items():
        for match in DIABETES_PATTERN.finditer(text):
            patients.add(number)
            start = max(0, match.start() - 60)
            print(f"  {number:>3}  ...{text[start:match.end() + 40]}...")
    print(f"  Patients with at least one mention: {len(patients)} of {len(topics)}")


def main():
    topics = load_topics()
    check_judged_trials()
    check_age_and_sex(topics)
    check_diabetes(topics)


if __name__ == "__main__":
    main()
