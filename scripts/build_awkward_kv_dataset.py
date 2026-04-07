from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PEOPLE = [f"person_{i:02d}" for i in range(32)]


def make_record(rng: random.Random, order_type: str, n_facts: int = 8) -> dict[str, str]:
    chosen = rng.sample(PEOPLE, n_facts)
    target_person = rng.choice(chosen)
    mapping = {person: str(rng.randrange(10)) for person in chosen}
    facts = [f"{person} -> {mapping[person]}" for person in chosen]
    rng.shuffle(facts)

    task = "Answer with a single digit only."
    question = f"What is the lock digit for {target_person}?"
    notes = "\n".join(facts)

    if order_type == "friendly":
        prompt = (
            f"Task: {task}\n\n"
            f"Notes:\n{notes}\n\n"
            f"Question: {question}\n"
            f"Answer: "
        )
    elif order_type == "awkward":
        prompt = (
            f"Task: {task}\n\n"
            f"Question first: {question}\n\n"
            f"You must read the notes before answering.\n"
            f"Notes:\n{notes}\n\n"
            f"Answer: "
        )
    else:
        raise ValueError(f"unknown order_type={order_type}")

    return {
        "order_type": order_type,
        "prompt": prompt,
        "target": mapping[target_person],
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--valid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    rng = random.Random(args.seed)

    train = [make_record(rng, "awkward") for _ in range(args.train_size)]
    valid_awkward = [make_record(rng, "awkward") for _ in range(args.valid_size)]
    valid_friendly = [make_record(rng, "friendly") for _ in range(args.valid_size)]

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid_awkward.jsonl", valid_awkward)
    write_jsonl(out / "valid_friendly.jsonl", valid_friendly)

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "seed": args.seed,
                "train_size": len(train),
                "valid_awkward_size": len(valid_awkward),
                "valid_friendly_size": len(valid_friendly),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

