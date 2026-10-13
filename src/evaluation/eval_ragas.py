"""Run RAGAS faithfulness, answer relevancy, and context precision evaluation.

Usage: python -m src.evaluation.eval_ragas --input evaluation_dataset.json
The JSON input is a list of {question, answer, contexts, ground_truth} objects.
"""

import argparse
import json
from pathlib import Path


def evaluate(input_path: Path, allow_paid_evaluator: bool = False) -> list:
    if not allow_paid_evaluator:
        raise ValueError(
            "RAGAS may call a paid external evaluator. Explicit opt-in required."
        )
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    required = {"question", "answer", "contexts", "ground_truth"}
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {i} missing fields: {sorted(missing)}")
    result = ragas_evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return result.to_pandas().to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ragas-results.json"))
    parser.add_argument(
        "--allow-paid-evaluator",
        action="store_true",
        help="Explicitly authorize external evaluator calls; optional dependencies and credentials required",
    )
    args = parser.parse_args()
    records = evaluate(args.input, args.allow_paid_evaluator)
    args.output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(records)} evaluation rows to {args.output}")


if __name__ == "__main__":
    main()
