"""Step 4: run the matcher on 10 test patients and compare results by age and sex.

Run from the project folder, after build_index.py:
    python src/evaluate.py

There are no expert labels saying which trials are truly right for each patient,
so this measures three things we can compute:
  - pool:       how many trials pass the age and sex filter
  - similarity: the average similarity of the top 20 matches
  - overlap:    how many of the top 20 stay the same when one detail changes
"""

from match import collection, eligibility_filter, match_trials

# Invented patients. The summaries leave out age and sex on purpose, so those
# two details only affect the results through the filter.
PATIENTS = [
    {"age": 15, "sex": "FEMALE", "summary": "Type 2 diabetes diagnosed one year ago, obesity, on metformin, family history of diabetes"},
    {"age": 24, "sex": "MALE", "summary": "Newly diagnosed type 2 diabetes, obesity, not yet on medication, wants lifestyle and diet support"},
    {"age": 35, "sex": "FEMALE", "summary": "Type 2 diabetes after previous gestational diabetes, polycystic ovary syndrome, on metformin"},
    {"age": 45, "sex": "MALE", "summary": "Type 2 diabetes with fatty liver disease and high triglycerides, on metformin and a statin"},
    {"age": 52, "sex": "FEMALE", "summary": "Type 2 diabetes poorly controlled on metformin and a sulfonylurea, HbA1c 9.5 percent, considering a GLP-1 medicine"},
    {"age": 58, "sex": "MALE", "summary": "Type 2 diabetes with painful peripheral neuropathy in both feet and a previous foot ulcer"},
    {"age": 63, "sex": "FEMALE", "summary": "Type 2 diabetes for 10 years with early chronic kidney disease and high blood pressure, on metformin"},
    {"age": 68, "sex": "MALE", "summary": "Type 2 diabetes on basal insulin, previous heart attack, heart failure with reduced ejection fraction"},
    {"age": 74, "sex": "FEMALE", "summary": "Type 2 diabetes with diabetic retinopathy and macular edema, treated with insulin"},
    {"age": 86, "sex": "MALE", "summary": "Long-standing type 2 diabetes, frailty, mild cognitive impairment, frequent low blood sugar on insulin"},
]

COMPARISON_AGES = [30, 50, 70, 85]


def pool_size(age, sex):
    """Count the trials that pass the age and sex filter."""
    return len(collection.get(where=eligibility_filter(age, sex))["ids"])


def mean_similarity(matches):
    """Average similarity score of a list of matches."""
    return sum(match["similarity"] for match in matches) / len(matches)


def overlap(matches_a, matches_b):
    """Count the trials that appear in both lists."""
    ids_a = {match["nct_id"] for match in matches_a}
    ids_b = {match["nct_id"] for match in matches_b}
    return len(ids_a & ids_b)


def other_sex(sex):
    return "MALE" if sex == "FEMALE" else "FEMALE"


def main():
    # Table 1: each patient as they are, then again with only the sex changed.
    print("Table 1: each patient, and what changes if only their sex is flipped")
    print(f"{'age':>3}  {'sex':<6}  {'pool':>4}  {'similarity':>10}  {'same top-20 after sex flip':>26}")
    for patient in PATIENTS:
        age, sex, summary = patient["age"], patient["sex"], patient["summary"]
        matches = match_trials(summary, age, sex)
        flipped = match_trials(summary, age, other_sex(sex))
        print(
            f"{age:>3}  {sex:<6}  {pool_size(age, sex):>4}  "
            f"{mean_similarity(matches):>10.3f}  {overlap(matches, flipped):>23}/20"
        )

    # Table 2: every summary re-run at the same four ages, so age is the only change.
    print()
    print("Table 2: all 10 summaries re-run at fixed ages (averages)")
    print(f"{'age':>3}  {'pool':>6}  {'similarity':>10}  {'same top-20 as at age 50':>24}")
    for test_age in COMPARISON_AGES:
        pools, similarities, overlaps = [], [], []
        for patient in PATIENTS:
            sex, summary = patient["sex"], patient["summary"]
            matches = match_trials(summary, test_age, sex)
            reference = match_trials(summary, 50, sex)
            pools.append(pool_size(test_age, sex))
            similarities.append(mean_similarity(matches))
            overlaps.append(overlap(matches, reference))
        count = len(PATIENTS)
        print(
            f"{test_age:>3}  {sum(pools) / count:>6.0f}  "
            f"{sum(similarities) / count:>10.3f}  {sum(overlaps) / count:>21.1f}/20"
        )


if __name__ == "__main__":
    main()
