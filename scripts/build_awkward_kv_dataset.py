from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PEOPLE = [f"person_{i:02d}" for i in range(32)]
TOKENS = [f"token_{i:02d}" for i in range(64)]


def make_record_simple(rng: random.Random, order_type: str, n_facts: int = 8) -> dict[str, str]:
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
        "variant": "simple_lookup",
        "prompt": prompt,
        "target": mapping[target_person],
    }


def make_record_multihop_sum(rng: random.Random, order_type: str, n_people: int = 12, n_distractors: int = 10) -> dict[str, str]:
    chosen_people = rng.sample(PEOPLE, n_people)
    chosen_tokens = rng.sample(TOKENS, n_people + n_distractors)
    target_people = rng.sample(chosen_people, 2)

    person_to_token = {person: chosen_tokens[i] for i, person in enumerate(chosen_people)}
    token_to_digit = {token: str(rng.randrange(10)) for token in chosen_tokens}

    fact_lines = [f"{person} uses {person_to_token[person]}" for person in chosen_people]
    fact_lines.extend(f"{token} opens with {token_to_digit[token]}" for token in chosen_tokens)
    rng.shuffle(fact_lines)

    a, b = target_people
    answer = str((int(token_to_digit[person_to_token[a]]) + int(token_to_digit[person_to_token[b]])) % 10)

    task = "Answer with a single digit only."
    question = f"What is the checksum digit for {a} and {b}? First find each person's token, then sum the two token digits modulo 10."
    notes = "\n".join(fact_lines)

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
            f"You must finish reading every note before answering.\n"
            f"Notes:\n{notes}\n\n"
            f"Answer: "
        )
    else:
        raise ValueError(f"unknown order_type={order_type}")

    return {
        "order_type": order_type,
        "variant": "multihop_sum",
        "prompt": prompt,
        "target": answer,
    }


def make_record(rng: random.Random, order_type: str, variant: str) -> dict[str, str]:
    if variant == "simple_lookup":
        return make_record_simple(rng, order_type)
    if variant == "multihop_sum":
        return make_record_multihop_sum(rng, order_type)
    raise ValueError(f"unknown variant={variant}")


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
    parser.add_argument("--variant", choices=["simple_lookup", "multihop_sum"], default="simple_lookup")
    args = parser.parse_args()

    out = Path(args.output_dir)
    rng = random.Random(args.seed)

    train = [make_record(rng, "awkward", args.variant) for _ in range(args.train_size)]
    valid_awkward = [make_record(rng, "awkward", args.variant) for _ in range(args.valid_size)]
    valid_friendly = [make_record(rng, "friendly", args.variant) for _ in range(args.valid_size)]

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid_awkward.jsonl", valid_awkward)
    write_jsonl(out / "valid_friendly.jsonl", valid_friendly)

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "seed": args.seed,
                "variant": args.variant,
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
