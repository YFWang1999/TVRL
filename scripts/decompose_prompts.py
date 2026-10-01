#!/usr/bin/env python3
"""Decompose text-to-video prompts into five binary sub-questions."""

import argparse
import ast
import json
from multiprocessing import get_context
from pathlib import Path
import re
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT_CSV = "assets/train_prompts.csv"
DEFAULT_OUTPUT_CSV = "assets/train_prompts_binary5.csv"
DEFAULT_PROMPT_TEMPLATE = (
    "assets/prompt_decomposition_template.txt"
)
DEFAULT_MODEL_PATH = "ckpts/Qwen3.5-9B"
DECOMPOSITION_SYSTEM_PROMPT = (
    "You are a structured data generator for text-to-video evaluation. "
    "Return exactly one valid JSON object and nothing else."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tmp_root", type=str, default=None, help="Directory for per-rank shard outputs.")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs / worker processes to use.")
    parser.add_argument("--debug_single", action="store_true", help="Run a single worker for debugging.")
    parser.add_argument("--debug_gpu", type=int, default=0, help="GPU id used when --debug_single is enabled.")
    parser.add_argument("--num_subquestions", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_retries", type=int, default=2, help="Number of repair retries per sample.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable trust_remote_code when loading the decomposition model.",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def log(rank: int | str, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def resolve_tmp_root(output_csv: str, tmp_root: str | None) -> Path:
    if tmp_root:
        return Path(tmp_root)
    output_path = Path(output_csv)
    return output_path.parent / f"{output_path.stem}_shards"


def resolve_worker_tmp_root(output_csv: str, tmp_root: str | None, rank: int) -> Path:
    return resolve_tmp_root(output_csv, tmp_root) / f"rank_{rank:02d}"


def resolve_row_output_path(worker_tmp_root: Path, row_idx: int) -> Path:
    return worker_tmp_root / f"row_{row_idx:06d}.json"


def resolve_failure_output_path(worker_tmp_root: Path, row_idx: int) -> Path:
    return worker_tmp_root / "failed" / f"row_{row_idx:06d}.json"


def cleanup_shard_outputs(tmp_root: Path) -> None:
    for shard_path in tmp_root.glob("rank_*/row_*.json"):
        shard_path.unlink()
    for failure_path in tmp_root.glob("rank_*/failed/row_*.json"):
        failure_path.unlink()


def load_input_df(csv_path: str, limit: int | None) -> pd.DataFrame:
    input_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if limit is not None:
        input_df = input_df.head(limit).copy()
    return input_df.where(pd.notna(input_df), None)


def load_input_rows(csv_path: str, limit: int | None) -> list[dict[str, Any]]:
    return load_input_df(csv_path, limit).to_dict(orient="records")


def build_decomposition_prompt(template: str, prompt_text: str, num_subquestions: int) -> str:
    return (
        template.replace("<|PROMPT|>", prompt_text.strip()).replace(
            "<|NUM_SUBQUESTIONS|>", str(num_subquestions)
        )
    )


def build_repair_prompt(
    prompt_text: str,
    invalid_output: str,
    error_message: str,
    num_subquestions: int,
) -> str:
    return f"""The previous response for a prompt-decomposition task was invalid.

You must repair it into one valid JSON object.

Original text-to-video prompt:
{prompt_text}

Previous invalid output:
{invalid_output}

Validation error:
{error_message}

Requirements:
- Return valid JSON only.
- No markdown.
- No code fences.
- No explanation.
- Include exactly {num_subquestions} sub-questions.
- IDs must be q1 to q{num_subquestions} in order.
- Every sub-question must be a visually grounded yes/no question.
- Every answer_type must be exactly "yes_no".
- Every question must end with a question mark.

Return JSON with this schema:
{{
  "overall_question": "...",
  "sub_questions": [
    {{
      "id": "q1",
      "question": "...?",
      "answer_type": "yes_no"
    }}
  ]
}}"""


def clean_generation_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    start_index = None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start_index = idx
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index : idx + 1].strip())
                start_index = None

    candidates.append(text.strip())

    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def parse_json_payload(text: str) -> dict[str, Any]:
    text = clean_generation_text(text)

    candidates = extract_json_candidates(text)

    for candidate in reversed(candidates):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Failed to parse model output as a JSON object.")


def normalize_subquestion(raw_subquestion: dict[str, Any], index: int) -> dict[str, str]:
    question = str(raw_subquestion.get("question", "")).strip()
    if not question:
        raise ValueError(f"Sub-question {index} is missing a question field.")
    if not question.endswith("?"):
        question = f"{question}?"

    return {
        "id": f"q{index}",
        "question": question,
        "answer_type": "yes_no",
    }


def normalize_decomposition(payload: dict[str, Any], num_subquestions: int) -> tuple[str, list[dict[str, str]]]:
    overall_question = str(payload.get("overall_question", "")).strip()
    if not overall_question:
        raise ValueError("Missing overall_question in decomposition output.")
    if not overall_question.endswith("?"):
        overall_question = f"{overall_question}?"

    raw_subquestions = payload.get("sub_questions")
    if not isinstance(raw_subquestions, list):
        raise ValueError("sub_questions must be a list.")
    if len(raw_subquestions) != num_subquestions:
        raise ValueError(
            f"Expected exactly {num_subquestions} sub-questions, got {len(raw_subquestions)}."
        )

    normalized = []
    for idx, raw_subquestion in enumerate(raw_subquestions, start=1):
        if not isinstance(raw_subquestion, dict):
            raise ValueError(f"Sub-question {idx} is not a JSON object.")
        normalized.append(normalize_subquestion(raw_subquestion, idx))

    return overall_question, normalized


def build_binary_question_prompt_from_subquestions(
    sub_questions: list[dict[str, str]],
    include_answer_type: bool = True,
    include_system_prompt: bool = True,
) -> str:
    prompt_lines: list[str] = []

    if include_system_prompt:
        prompt_lines.extend(
            [
                "You are a careful video question-answering assistant.",
                "Use only visually grounded evidence from the video, and do not guess unseen details.",
                "",
            ]
        )

    prompt_lines.extend(
        [
            "Given an AI generated video, answer the following atomic visual questions using only evidence from the video.",
            "",
            "Rules:",
            "- Answer each question independently.",
            "- Use concise canonical phrases.",
            "- Do not add explanations.",
            '- For "yes_no" questions, answer with "Yes" or "No" only.',
            "- Return valid JSON only.",
            "",
            "Output format:",
            '{"sub_answers": [{"id": "q1", "answer": "..."}]}',
            "",
            "Sub-questions:",
        ]
    )

    for sub_question in sub_questions:
        payload = {
            "id": sub_question["id"],
            "question": sub_question["question"].strip(),
        }
        if include_answer_type:
            payload["answer_type"] = sub_question.get("answer_type", "yes_no")
        prompt_lines.append(f"- {json.dumps(payload, ensure_ascii=False)}")

    return "\n".join(prompt_lines)


class DecompositionFailedError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


def load_model(
    model_path: str,
    trust_remote_code: bool,
    device: str,
    local_rank: int,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        )
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def generate_decomposition(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    try:
        model_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        model_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(model_input, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    output_tokens = outputs[0][input_length:]
    return tokenizer.decode(output_tokens, skip_special_tokens=True).strip()


def sanitize_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in record.items():
        if isinstance(value, dict):
            sanitized[key] = {sub_key: sanitize_value(sub_value) for sub_key, sub_value in value.items()}
        elif isinstance(value, list):
            sanitized[key] = [sanitize_value(item) for item in value]
        else:
            sanitized[key] = sanitize_value(value)
    return sanitized


def normalize_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_existing_shard_compatible(output_path: Path, record: dict[str, Any], row_idx: int) -> bool:
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if existing.get("_source_row_idx") != row_idx:
        return False

    return normalize_prompt_value(existing.get("prompt")) == normalize_prompt_value(record.get("prompt"))


def save_failure_artifact(
    worker_tmp_root: Path,
    row_idx: int,
    sample_id: Any,
    record: dict[str, Any],
    exc: Exception,
) -> None:
    failure_path = resolve_failure_output_path(worker_tmp_root, row_idx)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = exc.attempts if isinstance(exc, DecompositionFailedError) else []
    payload = {
        "row_idx": row_idx,
        "sample_id": sample_id,
        "prompt": record.get("prompt"),
        "error": str(exc),
        "attempts": attempts,
    }
    failure_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_decomposition_with_retries(
    prompt_text: str,
    args: argparse.Namespace,
    template: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    request_prompt = build_decomposition_prompt(
        template=template,
        prompt_text=prompt_text,
        num_subquestions=args.num_subquestions,
    )
    last_error = "Unknown decomposition failure."

    for attempt_idx in range(args.num_retries + 1):
        generation_text = generate_decomposition(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt_text=request_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        attempt_payload = {
            "attempt": attempt_idx,
            "request_prompt": request_prompt,
            "raw_output": generation_text,
        }
        try:
            payload = parse_json_payload(generation_text)
            overall_question, sub_questions = normalize_decomposition(payload, args.num_subquestions)
            attempt_payload["status"] = "ok"
            attempts.append(attempt_payload)
            return overall_question, sub_questions, attempts
        except Exception as exc:
            last_error = str(exc)
            attempt_payload["status"] = "failed"
            attempt_payload["error"] = last_error
            attempts.append(attempt_payload)
            request_prompt = build_repair_prompt(
                prompt_text=prompt_text,
                invalid_output=generation_text,
                error_message=last_error,
                num_subquestions=args.num_subquestions,
            )

    raise DecompositionFailedError(last_error, attempts)


def process_record(
    record: dict[str, Any],
    row_idx: int,
    args: argparse.Namespace,
    template: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> dict[str, Any]:
    prompt_value = record.get("prompt")
    if prompt_value is None or pd.isna(prompt_value):
        raise ValueError("Encountered an empty prompt in the input CSV.")
    prompt_text = str(prompt_value).strip()
    if not prompt_text:
        raise ValueError("Encountered a blank prompt in the input CSV.")
    overall_question, sub_questions, _ = run_decomposition_with_retries(
        prompt_text=prompt_text,
        args=args,
        template=template,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    question_prompt = build_binary_question_prompt_from_subquestions(sub_questions)

    output_record = dict(record)
    output_record["overall_question"] = overall_question
    output_record["sub_questions"] = json.dumps(sub_questions, ensure_ascii=False)
    output_record["questions"] = json.dumps([question_prompt], ensure_ascii=False)
    output_record["_source_row_idx"] = row_idx
    return sanitize_record(output_record)


def worker_main(
    worker_id: int,
    world_size: int,
    args: argparse.Namespace,
    shard_rank: int | None = None,
) -> None:
    rank = worker_id if shard_rank is None else shard_rank
    device = f"cuda:{worker_id}" if torch.cuda.is_available() else "cpu"
    worker_tmp_root = resolve_worker_tmp_root(args.output_csv, args.tmp_root, rank)
    worker_tmp_root.mkdir(parents=True, exist_ok=True)

    log(rank, f"starting on device={device}, world_size={world_size}")
    log(rank, f"outputs will be written to {worker_tmp_root}")

    tokenizer, model = load_model(
        args.model_path,
        args.trust_remote_code,
        device,
        worker_id,
    )
    template = load_text(args.prompt_template)
    input_rows = load_input_rows(args.input_csv, args.limit)

    shard_indices = list(range(rank, len(input_rows), world_size))
    log(rank, f"assigned {len(shard_indices)} rows")

    progress = tqdm(shard_indices, desc=f"rank {rank}", disable=(len(shard_indices) == 0))
    success_count = 0
    skip_count = 0
    error_count = 0

    for row_idx in progress:
        record = input_rows[row_idx]
        output_path = resolve_row_output_path(worker_tmp_root, row_idx)
        if output_path.exists() and not args.overwrite and is_existing_shard_compatible(output_path, record, row_idx):
            skip_count += 1
            continue

        try:
            output_record = process_record(
                record=record,
                row_idx=row_idx,
                args=args,
                template=template,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            output_path.write_text(
                json.dumps(output_record, ensure_ascii=False),
                encoding="utf-8",
            )
            success_count += 1
        except Exception as exc:
            error_count += 1
            sample_id = record.get("index", row_idx)
            save_failure_artifact(worker_tmp_root, row_idx, sample_id, record, exc)
            log(rank, f"failed on row={row_idx}, sample={sample_id}: {exc}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(
        rank,
        f"done. success={success_count}, skipped={skip_count}, errors={error_count}, output_dir={worker_tmp_root}",
    )


def merge_worker_outputs(args: argparse.Namespace) -> None:
    tmp_root = resolve_tmp_root(args.output_csv, args.tmp_root)
    input_df = load_input_df(args.input_csv, args.limit)
    input_columns = list(input_df.columns)
    valid_row_count = len(input_df)
    output_columns = input_columns + [
        column_name
        for column_name in ["overall_question", "sub_questions", "questions"]
        if column_name not in input_columns
    ]

    shard_paths = sorted(tmp_root.glob("rank_*/row_*.json"))
    rows = []
    for shard_path in shard_paths:
        row = json.loads(shard_path.read_text(encoding="utf-8"))
        source_row_idx = row.get("_source_row_idx")
        if not isinstance(source_row_idx, int) or source_row_idx < 0 or source_row_idx >= valid_row_count:
            continue
        current_prompt = normalize_prompt_value(input_df.iloc[source_row_idx].get("prompt"))
        if normalize_prompt_value(row.get("prompt")) != current_prompt:
            continue
        rows.append(row)

    if rows:
        rows.sort(key=lambda row: row.get("_source_row_idx", 10**18))
        output_df = pd.DataFrame(rows)
        if "_source_row_idx" in output_df.columns:
            output_df = output_df.drop(columns=["_source_row_idx"])
        ordered_columns = output_columns + [column for column in output_df.columns if column not in output_columns]
        output_df = output_df[ordered_columns]
    else:
        output_df = pd.DataFrame(columns=output_columns)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if len(output_df) != len(input_df):
        print(
            f"Warning: merged {len(output_df)} rows, expected {len(input_df)} rows. "
            "Some samples may have failed during decomposition.",
            flush=True,
        )

    print(f"Saved {len(output_df)} rows to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    tmp_root = resolve_tmp_root(args.output_csv, args.tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        cleanup_shard_outputs(tmp_root)

    available = torch.cuda.device_count()

    if args.debug_single:
        if available > 0 and (args.debug_gpu < 0 or args.debug_gpu >= available):
            raise ValueError(f"debug_gpu should be in [0, {available - 1}], got {args.debug_gpu}")
        worker_id = args.debug_gpu if available > 0 else 0
        worker_main(worker_id, 1, args, shard_rank=0)
        merge_worker_outputs(args)
        return

    if available == 0:
        log("main", "No CUDA device found. Falling back to single-process CPU execution.")
        worker_main(0, 1, args, shard_rank=0)
        merge_worker_outputs(args)
        return

    world_size = available if args.num_gpus is None else min(args.num_gpus, available)
    if world_size <= 0:
        raise ValueError("num_gpus must be positive.")

    ctx = get_context("spawn")
    processes = []
    for worker_id in range(world_size):
        process = ctx.Process(target=worker_main, args=(worker_id, world_size, args))
        process.start()
        processes.append(process)

    failed = False
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError("One or more worker processes failed.")

    merge_worker_outputs(args)


if __name__ == "__main__":
    main()
