"""Fetch per-question MMLU details of one model from the archived Open LLM Leaderboard.

The leaderboard stored, for every evaluated model, one parquet file per MMLU subject with
the log-likelihood the model gave to each of the four answer letters and the index of the
right one. This script downloads those files for one evaluation run and writes one JSONL
line per question:

    {"id": "anatomy/17", "subject": "anatomy", "logits": [-1.2, -0.4, -2.9, -3.1], "label": 1}

A softmax over the four log-likelihoods is the model's probability for each letter.

The dataset pages state no licence, so the files are not redistributed with calibcheck;
this script is the way to get them. It needs pyarrow, which calibcheck does not depend on:

    uv run --with pyarrow python scripts/fetch_leaderboard_details.py \
        --repo open-llm-leaderboard-old/details_meta-llama__Llama-2-7b-hf \
        --run 2023-08-29T17:54:59.197645 --out data/llama-2-7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/datasets/{repo}/tree/main/{run}?limit=1000"
FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SUBJECT = re.compile(r"hendrycksTest-(?P<subject>[a-z_]+)\|5_")
AGENT = {"User-Agent": "calibcheck-fetch"}


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=AGENT)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def list_files(repo: str, run: str) -> dict[str, str]:
    """Map MMLU subject to the path of its five-shot details file in one run."""
    listing = json.loads(get(API.format(repo=repo, run=urllib.parse.quote(run))))
    found = {}
    for entry in listing:
        match = SUBJECT.search(entry["path"])
        if entry["type"] == "file" and match and entry["path"].endswith(".parquet"):
            found[match["subject"]] = entry["path"]
    return dict(sorted(found.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="dataset id on the Hugging Face Hub")
    parser.add_argument("--run", required=True, help="run folder (a timestamp) in the dataset")
    parser.add_argument("--out", required=True, help="JSONL file to write")
    parser.add_argument("--cache", help="folder for the downloaded parquet files")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out.parent / (out.stem + "-parquet")
    cache.mkdir(parents=True, exist_ok=True)
    files = list_files(args.repo, args.run)
    if not files:
        raise SystemExit(f"no MMLU details found in {args.repo} run {args.run}")

    n_items = 0
    with out.open("w", encoding="utf-8") as handle:
        for subject, path in files.items():
            local = cache / f"{subject}.parquet"
            if not local.exists():
                local.write_bytes(get(FILE.format(repo=args.repo, path=urllib.parse.quote(path))))
            table = pq.read_table(local, columns=["predictions", "gold", "hashes"])
            for i, row in enumerate(table.to_pylist()):
                record = {
                    "id": f"{subject}/{i}",
                    "subject": subject,
                    "logits": row["predictions"],
                    "label": row["gold"],
                    "question_hash": row["hashes"]["example"],
                }
                handle.write(json.dumps(record) + "\n")
                n_items += 1
    print(f"wrote {n_items} questions from {len(files)} subjects to {out}")


if __name__ == "__main__":
    main()
