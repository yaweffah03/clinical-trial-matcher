# clinical-trial-matcher

Matches a patient summary to recruiting type 2 diabetes trials from ClinicalTrials.gov using sentence embeddings and Chroma, then checks the top trials' eligibility criteria with a language model (Google Gemini). Inspired by NIH's TrialGPT. Learning project, not a clinical tool.

> **Not a clinical tool.** This is a learning project. Its eligibility check is often wrong, and it must not be used to make decisions about care.

## What it does

You give it a short patient description, an age and a sex. It returns the 20 recruiting type 2 diabetes trials whose text is most similar to the description, keeping only trials that accept that age and sex.

```
python src/match.py --age 62 --sex FEMALE "Type 2 diabetes for 10 years, on metformin, early kidney disease"

 1. 0.674  NCT07760883  Metformin Effects According to SLC16A11 Carrier Status
 2. 0.646  NCT05345327  SGLT2 Inhibitors As First Line Therapy to Prevent Renal Decline in Type 2 Diabetes
 3. 0.635  NCT07487467  Autologous SVF Therapy for Type 2 Diabetes
 ...
```

## How it works

| Step | Script | What it does |
|---|---|---|
| 1 | `src/fetch_trials.py` | Downloads recruiting trials with the condition "type 2 diabetes" from the ClinicalTrials.gov v2 API and saves them to `data/trials.json`. |
| 2 | `src/build_index.py` | Turns each trial's title, conditions and eligibility criteria into an embedding with `all-MiniLM-L6-v2` and stores it in Chroma, with minimum age, maximum age and sex as metadata. |
| 3 | `src/match.py` | Embeds the patient summary, filters trials by age and sex, and returns the 20 nearest by cosine similarity. |
| 4 | `src/evaluate.py` | Runs 10 invented patients and compares results by age and sex. |
| 4b | `src/evaluate_known_item.py` | Searches with each trial's summary and checks whether the matcher finds that trial. |
| 4c | `src/review.py` | Scores the top 5 trials per patient from the suitability labels in `results/review.csv`. |
| 5 | `src/fetch_trec.py`, `src/check_trec.py`, `src/fetch_trec_ages.py`, `src/build_trec_index.py`, `src/evaluate_trec.py` | Runs the same matching method on the TREC 2021 Clinical Trials benchmark and scores it against expert labels. |
| 6 | `src/check_criteria.py` | Asks a language model about each eligibility criterion of the top trials, then decides the verdict with fixed rules in code. |
| 6b | `src/evaluate_criteria.py` | Measures the criteria check against the suitability labels and the TREC expert labels. |

The age and sex filter is a hard rule: a trial outside the patient's age range or sex is removed. The matcher's ranking is by text similarity only. The criteria check in Step 6 is a separate, optional step that reads the criteria.

## How to run it

Tested with Python 3.14, sentence-transformers 6.3.0 and chromadb 1.5.9.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_trials.py
python src/build_index.py
python src/match.py --age 62 --sex FEMALE "Type 2 diabetes for 10 years, on metformin, early kidney disease"
python src/evaluate.py
python src/evaluate_known_item.py
python src/review.py
```

The TREC benchmark is optional. Run it after the steps above, because it reuses the model and filter from `match.py`. It downloads about 138 MB:

```
python src/fetch_trec.py
python src/check_trec.py
python src/fetch_trec_ages.py
python src/build_trec_index.py
python src/evaluate_trec.py
```

The criteria check is also optional. It needs a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey), set in the terminal and never written into a file. Tested with google-genai 3.3.0:

```
export GEMINI_API_KEY="your key"
python src/check_criteria.py --age 62 --sex FEMALE "Type 2 diabetes for 10 years, on metformin, early kidney disease"
python src/evaluate_criteria.py --tune
python src/evaluate_criteria.py
```

The model's answers used for the results below are saved in `results/criterion_checks_gemini-3.5-flash-lite.json`, so `evaluate_criteria.py` reproduces them without sending any requests. The first design's verdicts are kept in `results/criteria_verdicts_gemini-3.5-flash-lite.json` as a record; the code that produced them was replaced by the final design.

## Data

The results below use a snapshot of **886 trials** fetched on 11 November 2026. The registry changes daily, so a fresh fetch will give slightly different numbers. The labels in `results/review.csv` belong to this snapshot: after a fresh fetch, the top 5 trials for a patient may no longer be the ones that were labelled.

- 846 trials accept all sexes, 33 are female-only and 7 are male-only.
- 385 trials set no maximum age and 16 set no minimum age. These are stored as 120 and 0 so the filter can treat every trial the same way.

## Evaluation

There are no expert labels saying which diabetes trials are truly right for each test patient. The evaluation has five parts, each answering a different question. The first three use the diabetes trials. The fourth runs the same method on a public benchmark that does have expert labels. The fifth measures the criteria check on both.

1. **Behaviour by age and sex.** How do the results change when only the patient's age or sex changes?
2. **Known-item test.** Can the text search find a trial when the right answer is known?
3. **Suitability labels.** Of the top 5 trials for each test patient, how many could the patient plausibly join?
4. **TREC 2021 benchmark.** How does the same method score against expert labels, on other patients and trials?
5. **Criteria check.** Does a language model reading the criteria tell suitable trials from unsuitable ones, and does it improve the ranking?

### 1. Behaviour by age and sex

This part measures three things that can be computed without labels:

- **Pool:** how many trials pass the age and sex filter.
- **Similarity:** the average cosine similarity of the top 20 matches.
- **Overlap:** how many of the top 20 stay the same when exactly one detail changes.

These describe how the matcher behaves. They are not a measure of clinical relevance.

The 10 test patients are invented. Their summaries leave out age and sex, so those two details only affect results through the filter.

#### Each patient, and the effect of flipping only their sex

| Age | Sex | Pool | Similarity | Same top-20 after sex flip |
|---|---|---|---|---|
| 15 | Female | 62 | 0.509 | 20/20 |
| 24 | Male | 652 | 0.625 | 20/20 |
| 35 | Female | 723 | 0.537 | 11/20 |
| 45 | Male | 759 | 0.593 | 20/20 |
| 52 | Female | 779 | 0.683 | 20/20 |
| 58 | Male | 768 | 0.620 | 19/20 |
| 63 | Female | 772 | 0.609 | 20/20 |
| 68 | Male | 671 | 0.582 | 20/20 |
| 74 | Female | 600 | 0.593 | 20/20 |
| 86 | Male | 402 | 0.612 | 20/20 |

#### All 10 summaries re-run at fixed ages (averages)

| Age | Pool | Similarity | Same top-20 as at age 50 |
|---|---|---|---|
| 30 | 703 | 0.600 | 16.9/20 |
| 50 | 780 | 0.607 | 20.0/20 |
| 70 | 668 | 0.602 | 16.2/20 |
| 85 | 422 | 0.575 | 8.6/20 |

#### What this shows

- **Age changes the results a lot at the extremes.** At 85, the same patients have about half the candidate trials they have at 50, keep fewer than half of the same top 20, and get lower similarity scores. The 15-year-old has only 62 eligible trials and the lowest similarity of all.
- **Sex rarely changes the results.** For 8 of 10 patients, flipping sex changed nothing, because 95% of trials accept all sexes.
- **Sex matters when the condition is sex-specific.** The 35-year-old woman with previous gestational diabetes and polycystic ovary syndrome kept only 11 of her top 20 after the flip, because many of her best matches are female-only trials.

### 2. Known-item test

The index is built from each trial's title, conditions and eligibility criteria. The trial's summary is never embedded, so it is unseen text about a known trial. This test uses every summary as a search query, with no age or sex filter, and checks whether the matcher returns the trial the summary came from.

Recall@k is the share of trials found within the top k results.

| Group | Trials | Recall@1 | Recall@5 | Recall@20 |
|---|---|---|---|---|
| All trials | 886 | 65.6% | 83.4% | 92.9% |
| Accepts all sexes | 846 | 65.1% | 83.3% | 92.6% |
| Female only | 33 | 75.8% | 84.8% | 100.0% |
| Male only | 7 | 71.4% | 85.7% | 100.0% |
| Accepts under 18 | 71 | 62.0% | 83.1% | 93.0% |
| Adults under 65 only | 68 | 73.5% | 83.8% | 97.1% |
| Accepts 65 and over | 747 | 65.2% | 83.4% | 92.5% |

#### What this shows

- **The text search works.** Out of 886 trials, the right one is the first result about two times in three, and in the top 20 more than nine times in ten.
- **No clear difference by sex or age group.** Recall@5 is between 83% and 86% for every group. The single-sex groups are small (33 and 7 trials), so their higher recall@1 is not strong evidence.
- **This is an easier task than patient matching.** A trial's summary shares wording with its own title and criteria. A patient description does not. The test shows the search finds text on the same topic; it does not show the patient is eligible.

These figures change slightly from run to run, because Chroma's search is approximate. Across repeated runs the overall figures moved by about 0.2 percentage points. In the smaller groups one trial is worth more than a percentage point, so those rows moved by up to 3.

### 3. Suitability labels

For each of the 10 test patients, the top 5 trials (50 patient–trial pairs) were labelled 1 (suitable) or 0 (not suitable), using three checks in order:

1. Is the trial about a problem the patient has?
2. Do the main inclusion criteria fit what the summary says?
3. Does any exclusion criterion clearly apply?

A condition or past procedure the trial requires counts as absent if the summary does not mention it. Lab values, measurements and demographics that the summary does not mention are not held against the patient.

The pairs were labelled twice:

- **AI labels:** an AI assistant (Claude) read each trial's eligibility criteria and gave a label and a one-line reason.
- **Human labels:** the project author labelled all 50 from the trial pages on ClinicalTrials.gov.

Neither labeller is a clinician. The labels and reasons are in `results/review.csv`.

**Agreement.** The two sets of labels agreed on 48 of 50 pairs on the first pass. The author rechecked the two disputed pairs against the trial pages and corrected both, so the final labels agree on all 50. In both cases the trial's topic fitted the patient, but one inclusion criterion ruled the patient out.

Precision at 5 is the share of top-5 trials labelled suitable.

| Group | Trials | Suitable |
|---|---|---|
| All patients | 50 | 56% |
| Top trial only | 10 | 50% |
| Female | 25 | 56% |
| Male | 25 | 56% |
| Age under 40 | 15 | 67% |
| Age 40 to 64 | 20 | 65% |
| Age 65 and over | 15 | 33% |

#### What this shows

- **About half the top results are suitable.** 28 of 50 top-5 trials, and 5 of 10 first-ranked trials, are ones the patient could plausibly join.
- **No difference by sex.** Both sexes have 14 of 25 suitable.
- **The oldest patients do worst.** Patients aged 65 and over have 5 of 15 suitable, against 23 of 35 for younger patients. In this test set the older patients also have the most complications (heart failure, macular edema, cognitive impairment), so age and complications cannot be separated. Trials on those topics often exclude the complication itself.
- **Most failures are eligibility, not topic.** A typical example: a lifestyle programme ranked second for a newly diagnosed patient who wants lifestyle support, but the trial requires a diagnosis of at least 12 months.
- **The age filter has a gap.** A trial whose criteria say "adult patients" but whose registry entry sets no age limits passed the filter for the 15-year-old.

### 4. TREC 2021 benchmark

The TREC 2021 Clinical Trials track is a public test set: 75 patient descriptions and 35,832 patient–trial pairs labelled by expert assessors as eligible (2), excluded (1) or not relevant (0). It covers all conditions, not only diabetes.

The same method is used as for the diabetes trials: the same model, the same trial text (title, conditions and eligibility criteria), the same cosine ranking and the same age and sex filter. Three things had to be added:

- **Patient age and sex** are pulled from the free-text description with two patterns (for example "45-year-old man", "32 yo woman", "74M"). This worked for all 75 patients.
- **Trial age limits and sex** are not in the benchmark's trial file, so they are fetched from the ClinicalTrials.gov API as it is today.
- **The trials** come from the file prepared by the TrialGPT authors: 26,149 trials, every one of which was judged for at least one patient.

Each patient's top 10 trials are scored three ways:

- **NDCG@10:** ranking quality from 0 to 1, using the labels 2, 1 and 0 as gains. Trials near the top count more.
- **Eligible:** the share of the top 10 labelled eligible.
- **Excluded:** the share of the top 10 labelled excluded.

A trial that was not judged for a patient counts as 0.

| Group | Patients | NDCG@10 | Eligible | Excluded |
|---|---|---|---|---|
| No filter | 75 | 0.491 | 31% | 31% |
| Age and sex filter | 75 | 0.516 | 36% | 26% |
| Female (filtered) | 37 | 0.487 | 35% | 22% |
| Male (filtered) | 38 | 0.545 | 37% | 31% |
| Age under 18 (filtered) | 11 | 0.624 | 42% | 32% |
| Age 18 to 64 (filtered) | 56 | 0.494 | 36% | 23% |
| Age 65 and over (filtered) | 8 | 0.524 | 27% | 45% |

#### What this shows

- **The age and sex filter helps.** It raises the share of eligible trials in the top 10 from 31% to 36% and lowers the share of excluded trials from 31% to 26%.
- **Similarity is not eligibility, measured.** Even with the filter, about a quarter of the top 10 are trials on the right topic that experts say the patient is excluded from.
- **The oldest patients again get the most excluded trials.** For patients aged 65 and over, 45% of the top 10 are excluded and 27% eligible. This matches the suitability labels above, but rests on only 8 patients.
- **The filter makes some mistakes.** It removed 75 of the 5,569 trials that experts labelled eligible (1.3%). The cause was not investigated. Possible reasons are registry age limits that changed after 2021, or assessors being lenient about age.

#### What this does not show

- **It is not a check of the diabetes results.** Only two of the 75 patients have type 2 diabetes, and for neither is it the main problem. The benchmark tests the method, not the diabetes matches.
- **It is easier than a real search.** The trial file contains only judged trials. The original TREC task searched a snapshot of the whole registry taken on 27 April 2021, containing 375,580 trials, about 14 times as many, so these scores should not be compared directly with scores from that task.

### 5. Criteria check

The matcher never reads the criteria, which is why a quarter of its TREC top 10 are trials the patient is excluded from. Step 6 adds a check that does, using `gemini-3.5-flash-lite` on Google's free tier. The larger free Gemini models did not answer within two minutes on a free key, so the smallest one was used.

#### How it works

The model does not give the verdict. For each trial it answers:

- Is the trial about a condition the patient has?
- For every inclusion criterion, and for each exclusion criterion the patient has: what is the criterion about, does the description show the patient has it (yes, no or not stated), and which exact words show it?

Code then decides, with fixed rules:

- An exclusion criterion rules the patient out only if the answer is yes.
- An inclusion criterion rules the patient out only if the answer is no.
- Anything the description does not mention never rules anyone out.
- A yes or no only counts if its quoted evidence really appears in the patient description. Otherwise it is treated as not stated.

Splitting the work this way makes every verdict traceable to one criterion and one quote, and lets the rules change without asking the model again.

For TREC, the matcher's top 20 trials per patient are checked and re-ordered: trials about the patient's condition first, then by how many criteria they fail, fewest first, keeping the matcher's order for ties. A simpler order (eligible, then not relevant, then excluded) is shown for comparison.

#### Keeping the test fair

The data was split before any changes were made:

- **Tuning:** TREC patients 11 to 30 (400 patient–trial pairs). The instructions and rules were changed by looking only at these results.
- **Test:** the 50 labelled diabetes pairs and TREC patients 1 to 10 (250 pairs), scored once, after the method was fixed. The ranking by failed criteria was chosen on the tuning patients before the test results were seen.

The design went through two versions:

- **First design:** the model gave the verdict itself. It rejected more than half of the suitable trials, because it treated missing details as failures.
- **Final design:** the per-criterion answers and code-decided verdict described above. A first version of its rules also failed any inclusion criterion about a condition, treatment or procedure that the description never mentioned. On the tuning patients that rule rejected 86 of 113 expert-eligible trials and made the ranking worse, so it was removed.

#### Results on the test data

**Diabetes pairs (the author's suitability labels):**

| | First design | Final design |
|---|---|---|
| Check agrees with the label | 34 of 50 (68%) | 41 of 50 (82%) |
| Unsuitable trials caught | 21 of 22 | 17 of 22 |
| Suitable trials wrongly rejected | 15 of 28 | 4 of 28 |

**TREC patients 1 to 10 (expert labels), top 10 after re-ordering the top 20:**

| | NDCG@10 | Eligible | Excluded |
|---|---|---|---|
| Matcher only | 0.457 | 25% | 40% |
| First design | 0.490 | 30% | 30% |
| Final design, by verdict | 0.476 | 32% | 28% |
| Final design, by failed criteria | 0.523 | 34% | 37% |

The final design caught 61 of 74 expert-excluded trials and wrongly rejected 20 of 46 expert-eligible ones. On the tuning patients the ranking by failed criteria scored 0.617, against 0.478 for the matcher alone. The smaller gain on the test patients is the reason for keeping them separate.

#### What this shows

- **Reading the criteria helps.** On held-out data the check agrees with the suitability labels on 82% of pairs, and the re-ordered TREC top 10 has more eligible trials (34% against 25%) and a higher NDCG@10 (0.523 against 0.457).
- **How the verdict is used matters as much as the model.** With the same model answers, ranking by number of failed criteria beat a strict eligible-or-excluded order. Most expert-eligible trials the check rejects fail on a single criterion, so a strict order pushes them to the bottom.
- **The small model breaks its own instructions.** Three of the four suitable diabetes trials it rejected came from mistakes the instructions forbid: treating "previous gestational diabetes" as current, guessing that "early chronic kidney disease" means an eGFR of 60 or below, and assuming a "previous heart attack" was within the last 6 months. The fourth, counting mild cognitive impairment as a neurological disorder, is arguable.
- **The model's quotes are not always exact.** 83 answers on the test data quoted evidence that was not in the description, and were treated as not stated.
- **The share of excluded trials in the top 10 rises with the main ranking** (28% to 37%), though it stays below the matcher's 40%. Pushing unrelated trials down lets more trials on the right topic move up, including some the patient is excluded from.

These failures were found on the test data, so the method was not changed to fix them. Doing so would turn the test into tuning data.

## Limitations

- **Similarity is not eligibility.** The matcher ranks trials by how similar their text is to the patient summary. A trial that excludes kidney disease can rank highly for a patient with kidney disease, because both texts are about kidney disease. The matcher itself checks only age and sex; the criteria check in Step 6 is a separate, imperfect step.
- **Small, free language model.** The criteria check uses the smallest free Gemini model. It misreads some criteria and does not always follow its instructions. Its answers are not repeatable exactly: asking again can give a different answer, which is why the answers used for the results are saved.
- **Not comparable with TrialGPT's accuracy.** TrialGPT reports accuracy per criterion, against criterion-level expert annotations. This project measures whole-trial verdicts, which are harder: one misread criterion makes the verdict wrong. No criterion-level labels were made here.
- **Patient data is sent to Google.** The criteria check sends the patient description to Google's API (with storage turned off). The descriptions here are invented or from a public benchmark. Real patient data should never be sent this way.
- **Most trial text is cut off.** The embedding model reads at most 256 tokens. 509 of the 886 trial texts (57%) are longer, so later criteria, usually the exclusion criteria, are ignored.
- **No expert labels for the diabetes trials.** Expert labels are used only in the TREC benchmark, which covers other patients and trials. The suitability labels come from an AI assistant and the project author, not from clinicians. The author's labels were recorded in the same file as the AI labels, so the labelling cannot be shown to be blind. Patient summaries are one sentence long, so many criteria could not be checked.
- **Small test set.** There are only 10 invented patients and 50 labelled pairs. Group figures rest on as few as 15 pairs, and each pair moves the criteria check's agreement by 2 percentage points. They show tendencies, not statistically tested differences.
- **Age limits in free text are missed.** The filter uses the registry's age fields. A limit that appears only in the criteria text is not applied.
- **Broad condition search.** The API search returns any trial that lists type 2 diabetes among its conditions, including trials aimed at prevention or at people at risk.
- **Simplified age and sex.** Missing age limits are replaced with 0 and 120, and sex is limited to the registry's three values.
- **General-purpose model.** `all-MiniLM-L6-v2` was not trained on medical text.
- **One snapshot.** Results reflect the registry on one day.
- **Simple text patterns for TREC patients.** Age and sex are found with two patterns that fit these 75 descriptions. When a description never says "man" or "woman", sex comes from the first pronoun, which on other data could refer to a relative.
- **Mixed dates in the TREC run.** Trial text is from the 2021 benchmark, but age limits and sex are from the registry today.

## How this differs from TrialGPT

TrialGPT uses large language models in three stages: retrieving candidate trials, judging the patient against each eligibility criterion, and ranking trials from those judgements. It was evaluated on expert-annotated patient–trial pairs. This project does simpler versions of all three: retrieval with a small embedding model instead of a language model, criterion judgements with a small free model instead of GPT-4, and ranking by counting failed criteria instead of a learned score. Unlike TrialGPT, the model here never gives the verdict itself; fixed rules in code do.

## Credits

- Trial data: [ClinicalTrials.gov](https://clinicaltrials.gov), U.S. National Library of Medicine.
- Benchmark: [TREC 2021 Clinical Trials track](https://trec.nist.gov/data/trials2021.html), U.S. National Institute of Standards and Technology. The trial file is the one published by the TrialGPT authors.
- Inspiration: Jin Q, et al. "Matching patients to clinical trials with large language models." *Nature Communications* (2024).
