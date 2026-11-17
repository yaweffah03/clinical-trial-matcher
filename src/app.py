"""Step 7: a web page where anyone can try the matcher and the criteria check.

Run from the project folder, after build_index.py:
    streamlit run src/app.py

Streamlit turns this script into a web page. Every time the visitor clicks a
button, Streamlit runs the whole script again from the top and redraws the page.

The eligibility check is optional. It needs GEMINI_API_KEY, set in the terminal
(or in Streamlit's secrets when the page is hosted online). Without it, the page
still finds trials but cannot check them.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

import build_index

st.set_page_config(page_title="Clinical trial matcher", page_icon="🔎")

# A hosted copy starts without the trial database, so build it once.
# This must happen before match.py is imported, because match.py opens it.
if not Path(build_index.DB_FOLDER).exists():
    with st.spinner("Preparing the trial database. This only happens the first time..."):
        build_index.main()

from match import collection, match_trials  # noqa: E402  (needs the database above)

# The check needs an API key. Only load it if there is one, because
# check_criteria.py stops with an error when the key is missing.
CAN_CHECK = bool(os.environ.get("GEMINI_API_KEY"))
if CAN_CHECK:
    from check_criteria import is_quoted_from, load_verdicts, patient_text, saved_or_new_verdict

TRIALS_SHOWN = 5  # each checked trial is one request to the free API
CHECK_SECONDS = 90  # stop waiting for the AI after this long

EXAMPLE = "Type 2 diabetes for 10 years, on metformin, early kidney disease"
ANSWER_WORDS = {"yes": "Yes", "no": "No", "not_stated": "Not mentioned"}


def check_with_time_limit(verdicts, key, patient, trial_text):
    """Run the criteria check, but give up if the free API has not answered in time."""
    # The check runs in a second thread, so this one can stop waiting.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(saved_or_new_verdict, verdicts, key, patient, trial_text)
    try:
        return future.result(timeout=CHECK_SECONDS)
    except Exception:
        return None
    finally:
        pool.shutdown(wait=False)


def show_verdict(verdict, patient):
    """Explain the check's verdict for one trial in plain words."""
    if verdict is None:
        st.error("The AI did not answer in time. The free service may be busy, "
                 "or its daily limit may be used up. Please try again later.")
        return
    if verdict["verdict"] == "not_checked":
        st.warning("The AI could not check this trial.")
        return
    if verdict["verdict"] == "not_relevant":
        st.info("The AI thinks this trial is about a different condition.")
        return
    if verdict["verdict"] == "eligible":
        st.success("Nothing in the description rules this patient out.")
    else:
        st.error("Probably not eligible. The description conflicts with:")
        for criterion in verdict["failed_criteria"]:
            st.markdown(f"- {criterion}")

    # Show every answer the AI gave, so the visitor can see why.
    rows = []
    for item in verdict["check"]["criteria"]:
        answer = ANSWER_WORDS[item["patient_has"]]
        if item["patient_has"] != "not_stated" and not is_quoted_from(item["evidence"], patient):
            answer += " (ignored: quote not found in the description)"
        rows.append({
            "Criterion": item["criterion"],
            "Trial wants it": "Yes" if item["kind"] == "inclusion" else "No",
            "Patient has it": answer,
            "Words the AI pointed to": item["evidence"],
        })
    with st.expander("See every criterion the AI checked"):
        st.dataframe(rows, hide_index=True)


st.title("Clinical trial matcher")
st.write(
    "Describe a person with type 2 diabetes. This page finds recruiting clinical "
    "trials on ClinicalTrials.gov whose descriptions are closest to yours, and can "
    "ask an AI to read each trial's entry rules."
)
st.warning(
    "**This is a student project, not medical advice.** Results are often wrong. "
    "Do not enter real patient information: descriptions are sent to Google's AI service."
)

with st.form("patient"):
    summary = st.text_area("Describe the patient", value=EXAMPLE,
                           help="For example: how long they have had diabetes, medicines, other conditions.")
    age = st.number_input("Age", min_value=0, max_value=120, value=62)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    check = st.checkbox(
        "Also check each trial's entry rules with AI (takes about a minute)",
        value=CAN_CHECK,
        disabled=not CAN_CHECK,
        help=None if CAN_CHECK else "Not available: no API key is set for this page.",
    )
    submitted = st.form_submit_button("Find trials", type="primary")

if submitted:
    sex_code = sex.upper()
    matches = match_trials(summary, age, sex_code, top_n=TRIALS_SHOWN)
    st.subheader(f"The {len(matches)} closest trials")
    st.caption("Only trials that accept this age and sex are shown. "
               "They are ordered by how similar their text is to the description.")

    if check:
        patient = patient_text(summary, age, sex_code)
        verdicts = load_verdicts()
        stored = collection.get(ids=[match["nct_id"] for match in matches])
        trial_texts = dict(zip(stored["ids"], stored["documents"]))

    for rank, match in enumerate(matches, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {match['title']}**")
            st.markdown(f"[{match['nct_id']} on ClinicalTrials.gov]"
                        f"(https://clinicaltrials.gov/study/{match['nct_id']})")
            if check:
                key = f"demo|{patient}|{match['nct_id']}"
                with st.spinner(f"The AI is reading trial {rank} of {len(matches)}..."):
                    verdict = check_with_time_limit(verdicts, key, patient, trial_texts[match["nct_id"]])
                show_verdict(verdict, patient)
