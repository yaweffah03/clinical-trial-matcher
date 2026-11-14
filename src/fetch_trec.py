"""Step 6a: download the TREC 2021 Clinical Trials benchmark.

Run from the project folder:
    python src/fetch_trec.py

TREC 2021 is a public test set for matching patients to clinical trials.
It has three parts, saved to data/trec/:
  - topics2021.xml          75 patient descriptions (TREC calls them "topics")
  - qrels2021.txt           expert labels, one per line: patient, trial, label
                            (0 = not relevant, 1 = excluded, 2 = eligible)
  - trec_2021_corpus.jsonl  the trials, one per line (about 137 MB), as
                            prepared by the TrialGPT authors
"""

from pathlib import Path

import requests

TREC_FOLDER = Path("data/trec")
TOPICS_FILE = TREC_FOLDER / "topics2021.xml"
QRELS_FILE = TREC_FOLDER / "qrels2021.txt"
CORPUS_FILE = TREC_FOLDER / "trec_2021_corpus.jsonl"

DOWNLOADS = {
    TOPICS_FILE: "https://trec.nist.gov/data/trials/topics2021.xml",
    QRELS_FILE: "https://trec.nist.gov/data/trials/qrels2021.txt",
    CORPUS_FILE: "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl",
}


def download(url, path):
    """Download one file, writing it to disk a piece at a time."""
    # stream=True stops requests from loading the whole file into memory.
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


def main():
    TREC_FOLDER.mkdir(parents=True, exist_ok=True)
    for path, url in DOWNLOADS.items():
        if path.exists():
            print(f"Already have {path}")
            continue
        print(f"Downloading {path} ...")
        download(url, path)
    print("Done.")


if __name__ == "__main__":
    main()
