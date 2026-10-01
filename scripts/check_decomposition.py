#!/usr/bin/env python3
"""Validate a TVRL binary5 prompt CSV; optionally fill absent ideal Yes targets."""
import argparse
import csv
import json
from pathlib import Path


def check_row(row, fill_yes=False):
    if not row.get("prompt", "").strip():
        raise ValueError("blank prompt")
    if not row.get("index", "").strip():
        raise ValueError("missing stable index")
    int(row["seed"])
    if not row.get("overall_question", "").strip():
        raise ValueError("missing overall_question")
    ids = [f"q{i}" for i in range(1, 6)]
    sub = json.loads(row["sub_questions"])
    if not isinstance(sub, list) or len(sub) != 5:
        raise ValueError("expected five sub_questions")
    for expected, item in zip(ids, sub):
        if item.get("id") != expected or item.get("answer_type") != "yes_no":
            raise ValueError("invalid subquestion id/type")
        if not item.get("question", "").strip().endswith("?"):
            raise ValueError("question must be nonempty and end with ?")
    questions = json.loads(row["questions"])
    if not isinstance(questions, list) or len(questions) != 1 or not isinstance(questions[0], str):
        raise ValueError("questions must contain one joint question string")
    for item in sub:
        if item["question"] not in questions[0] or item["id"] not in questions[0]:
            raise ValueError("joint question does not contain all subquestions")
    added = False
    if not row.get("ref_answers", "").strip() and fill_yes:
        target = {"sub_answers": [{"id": q, "answer": "Yes"} for q in ids]}
        row["ref_answers"] = json.dumps([target], ensure_ascii=False)
        added = True
    refs = json.loads(row.get("ref_answers", ""))
    if not isinstance(refs, list) or len(refs) != 1:
        raise ValueError("ref_answers must contain one structured target")
    target = json.loads(refs[0]) if isinstance(refs[0], str) else refs[0]
    answers = target["sub_answers"]
    if not isinstance(answers, list) or len(answers) != 5:
        raise ValueError("expected five answer slots")
    if [a.get("id") for a in answers] != ids or any(a.get("answer") != "Yes" for a in answers):
        raise ValueError("binary5 ideal target must be q1..q5, all Yes")
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fill-yes", action="store_true")
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    if args.fill_yes and args.output is None:
        parser.error("--fill-yes requires --output")
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new path")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        parser.error("empty input")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        parser.error(f"expected {args.expected_rows} rows, got {len(rows)}")
    seen, failures, filled = set(), [], 0
    for n, row in enumerate(rows, start=2):
        try:
            filled += int(check_row(row, args.fill_yes))
            if row["index"] in seen:
                raise ValueError("duplicate index")
            seen.add(row["index"])
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            failures.append(f"CSV record {n}: {exc}")
    if failures:
        parser.exit(1, "\n".join(failures[:10]) + f"\nFAILED: {len(failures)}/{len(rows)} rows\n")
    if args.output is not None:
        if "ref_answers" not in fields:
            fields.append("ref_answers")
        with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"OK: {len(rows)} rows, 5 questions/answer slots per row; filled {filled} targets")


if __name__ == "__main__":
    main()
