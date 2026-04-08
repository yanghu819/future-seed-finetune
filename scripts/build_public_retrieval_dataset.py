from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
import zipfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download


PUBLIC_TASKS: dict[str, dict[str, Any]] = {
    "hotpotqa": {
        "metric": "qa_f1",
        "recommended_max_length": 4096,
        "eval_source": {"repo_id": "THUDM/LongBench", "config": "hotpotqa", "split": "test"},
        "train_source": {"repo_id": "hotpotqa/hotpot_qa", "config": "fullwiki", "split": "train"},
    },
    "2wikimqa": {
        "metric": "qa_f1",
        "recommended_max_length": 4096,
        "eval_source": {"repo_id": "THUDM/LongBench", "config": "2wikimqa", "split": "test"},
        "train_source": {"repo_id": "framolfese/2WikiMultihopQA", "config": None, "split": "train"},
    },
    "musique": {
        "metric": "qa_f1",
        "recommended_max_length": 4096,
        "eval_source": {"repo_id": "THUDM/LongBench", "config": "musique", "split": "test"},
        "train_source": {"repo_id": "dgslibisey/MuSiQue", "config": None, "split": "train"},
    },
    "passage_retrieval_en": {
        "metric": "exact_match",
        "recommended_max_length": 4096,
        "eval_source": {"repo_id": "THUDM/LongBench", "config": "passage_retrieval_en", "split": "test"},
        "train_source": None,
    },
}

LONG_BENCH_DATA_ZIP_ENV = "LONG_BENCH_DATA_ZIP"
DEFAULT_LONG_BENCH_DATA_ZIP = Path("artifacts/transfer/longbench/data.zip")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def format_paragraphs_from_context(context: Any) -> str:
    if isinstance(context, str):
        return context
    rendered: list[str] = []
    if isinstance(context, dict):
        titles = context.get("title")
        sentences = context.get("sentences")
        paragraphs = context.get("paragraphs")
        if isinstance(titles, list) and isinstance(sentences, list):
            for title, sent_list in zip(titles, sentences):
                joined = " ".join(str(x) for x in sent_list)
                rendered.append(f"{title}: {joined}")
        elif isinstance(paragraphs, list):
            for item in paragraphs:
                rendered.append(format_paragraphs_from_context(item))
        else:
            for key, value in context.items():
                rendered.append(f"{key}: {value}")
        return "\n\n".join(x for x in rendered if x)
    if isinstance(context, list):
        for item in context:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("header") or ""
                text = item.get("paragraph_text") or item.get("paragraph") or item.get("text")
                if text is None and isinstance(item.get("sentences"), list):
                    text = " ".join(str(x) for x in item["sentences"])
                if text is None:
                    text = json.dumps(item, ensure_ascii=True)
                prefix = f"{title}: " if title else ""
                rendered.append(f"{prefix}{text}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                left, right = item
                if isinstance(right, list):
                    rendered.append(f"{left}: {' '.join(str(x) for x in right)}")
                else:
                    rendered.append(f"{left}: {right}")
            else:
                rendered.append(str(item))
        return "\n\n".join(x for x in rendered if x)
    return str(context)


def build_eval_prompt(task: str, row: dict[str, Any]) -> str:
    question = str(row.get("input", "")).strip()
    context = str(row.get("context", "")).strip()
    if task == "passage_retrieval_en":
        return (
            "Read the passages and answer with the exact passage identifier or title only.\n\n"
            f"Passages:\n{context}\n\n"
            f"Query:\n{question}\n\n"
            "Answer: "
        )
    return (
        "Read the context and answer with a short span copied from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer: "
    )


def build_train_prompt(task: str, question: str, context: str) -> str:
    if task == "passage_retrieval_en":
        return (
            "Read the passages and answer with the exact passage identifier or title only.\n\n"
            f"Passages:\n{context}\n\n"
            f"Query:\n{question}\n\n"
            "Answer: "
        )
    return (
        "Read the context and answer with a short span copied from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer: "
    )


def resolve_longbench_data_zip() -> Path:
    candidates: list[Path] = []
    env_value = os.environ.get(LONG_BENCH_DATA_ZIP_ENV)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(DEFAULT_LONG_BENCH_DATA_ZIP)
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    downloaded = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
        endpoint=os.environ.get("HF_ENDPOINT"),
        local_dir=str(DEFAULT_LONG_BENCH_DATA_ZIP.parent),
    )
    return Path(downloaded)


def longbench_rows_from_local_zip(task: str, limit: int) -> list[dict[str, Any]]:
    zip_path = resolve_longbench_data_zip()
    inner_path = f"data/{task}.jsonl"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_path) as handle:
            for idx, raw_line in enumerate(handle):
                row = json.loads(raw_line.decode("utf-8"))
                answers = row.get("answers") or []
                if not answers:
                    continue
                rows.append(
                    {
                        "id": str(row.get("_id", idx)),
                        "task": task,
                        "metric": PUBLIC_TASKS[task]["metric"],
                        "order_type": "longbench_eval",
                        "variant": task,
                        "prompt": build_eval_prompt(task, row),
                        "target": str(answers[0]),
                        "answers": [str(x) for x in answers],
                    }
                )
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def eval_rows_from_longbench(task: str, limit: int) -> list[dict[str, Any]]:
    return longbench_rows_from_local_zip(task, limit)


def train_rows_hotpotqa(limit: int) -> list[dict[str, Any]]:
    spec = PUBLIC_TASKS["hotpotqa"]["train_source"]
    ds = load_dataset(spec["repo_id"], spec["config"], split=spec["split"])
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()
        if not answer or not question:
            continue
        context = format_paragraphs_from_context(row.get("context"))
        rows.append(
            {
                "id": str(row.get("id", idx)),
                "task": "hotpotqa",
                "metric": "qa_f1",
                "order_type": "train_public",
                "variant": "hotpotqa",
                "prompt": build_train_prompt("hotpotqa", question, context),
                "target": answer,
                "answers": [answer],
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def train_rows_2wikimqa(limit: int) -> list[dict[str, Any]]:
    spec = PUBLIC_TASKS["2wikimqa"]["train_source"]
    ds = load_dataset(spec["repo_id"], split=spec["split"])
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()
        if not answer or not question:
            continue
        context = format_paragraphs_from_context(row.get("context"))
        rows.append(
            {
                "id": str(row.get("_id", idx)),
                "task": "2wikimqa",
                "metric": "qa_f1",
                "order_type": "train_public",
                "variant": "2wikimqa",
                "prompt": build_train_prompt("2wikimqa", question, context),
                "target": answer,
                "answers": [answer],
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def train_rows_musique(limit: int) -> list[dict[str, Any]]:
    spec = PUBLIC_TASKS["musique"]["train_source"]
    ds = load_dataset(spec["repo_id"], split=spec["split"])
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()
        if not answer or not question:
            continue
        paragraphs = row.get("paragraphs") or row.get("context")
        context = format_paragraphs_from_context(paragraphs)
        aliases = row.get("answer_aliases") or [answer]
        rows.append(
            {
                "id": str(row.get("id", idx)),
                "task": "musique",
                "metric": "qa_f1",
                "order_type": "train_public",
                "variant": "musique",
                "prompt": build_train_prompt("musique", question, context),
                "target": answer,
                "answers": [str(x) for x in aliases] or [answer],
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def build_train_rows(task: str, limit: int) -> list[dict[str, Any]]:
    if task == "hotpotqa":
        return train_rows_hotpotqa(limit)
    if task == "2wikimqa":
        return train_rows_2wikimqa(limit)
    if task == "musique":
        return train_rows_musique(limit)
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=sorted(PUBLIC_TASKS), required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=64)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = build_train_rows(args.task, args.train_limit)
    eval_rows = eval_rows_from_longbench(args.task, args.eval_limit)
    task_cfg = PUBLIC_TASKS[args.task]

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval_longbench.jsonl", eval_rows)

    metadata = {
        "task_id": args.task,
        "source": "longbench",
        "metric": task_cfg["metric"],
        "prompt_template": "public_retrieval_v1",
        "recommended_max_length": task_cfg["recommended_max_length"],
        "train_source": task_cfg["train_source"],
        "eval_splits": {"eval_longbench": len(eval_rows)},
        "train_examples": len(train_rows),
        "train_unavailable": len(train_rows) == 0,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"status": "ok", "output_dir": str(out), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
