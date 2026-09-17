#!/usr/bin/env python3
"""Run externally supplied ds4 questions through AX's local generation APIs.

Independent strict grading; no reference grader or generated code is executed.
Each run uses a fresh output directory and retains requests and raw responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from ds4_questions import load_questions

SYSTEM = (
    "Solve the question carefully. Finish your visible response with one line "
    "of the form Answer: <answer>. Do not write anything after that line."
)
FORMATS = {
    "CHOICE": "one choice letter",
    "INTEGER": "one integer",
    "RATIONAL": "one integer or fraction",
    "EXACT_TEXT": "the exact answer text",
    "ORDERED_SEQUENCE": "answers in the requested order, separated by commas",
    "LINE_SET": "exact line numbers separated by commas; use 0 for safe code",
}


def build_request(case: dict, model: str, budget: int, answer_reserve: int = 0) -> dict:
    question = case["question"]
    if case["choices"]:
        question += "\n\nChoices:\n" + "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(case["choices"])
        )
    question += "\n\nFinal answer format: Answer: <" + FORMATS[case["kind"]] + ">"
    return {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": budget,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
        **({"ax_answer_reserve_tokens": answer_reserve} if answer_reserve else {}),
    }


def normalize(value: str, kind: str):
    value = value.strip()
    if kind == "CHOICE":
        if not re.fullmatch("[A-Ja-j]", value):
            raise ValueError("Expected one letter")
        return value.upper()
    if kind == "INTEGER":
        if not re.fullmatch(r"[+-]?\d+", value):
            raise ValueError("Expected integer")
        return int(value)
    if kind == "RATIONAL":
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+|\s*/\s*\d+)?", value):
            raise ValueError("Expected rational")
        return Fraction(value.replace(" ", ""))
    if kind == "LINE_SET":
        # A range denotes all its lines; no substring or overlap scoring.
        if not re.fullmatch(r"\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*", value):
            raise ValueError("Expected line set")
        numbers = set()
        for part in value.split(","):
            ends = [int(v.strip()) for v in part.split("-")]
            low, high = ends[0], ends[-1]
            if low > high or high > 10000:
                raise ValueError("Invalid line range")
            numbers.update(range(low, high + 1))
        if 0 in numbers and len(numbers) > 1:
            raise ValueError("Safe marker mixed with lines")
        return frozenset(numbers)
    if kind == "ORDERED_SEQUENCE":
        parts = tuple(" ".join(p.split()).casefold() for p in value.split(","))
        if not all(parts):
            raise ValueError("Empty sequence element")
        return parts
    if kind == "EXACT_TEXT":
        return " ".join(value.split()).casefold()
    raise ValueError("Unknown answer kind")


def grade(case: dict, response: dict) -> dict:
    try:
        (choice,) = response["choices"]
        reason = choice["finish_reason"]
        message = choice["message"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Non-text content")
    except (KeyError, TypeError, ValueError):
        return {"status": "protocol_error", "answer": None}
    if reason == "length":
        return {"status": "truncated", "answer": None}
    if reason != "stop":
        return {"status": "incomplete", "answer": None}
    # Never grade private reasoning or anything inside a think block.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    if "<think>" in content or "</think>" in content:
        return {"status": "invalid_format", "answer": None}
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or not re.fullmatch(r"Answer:\s*\S.*", lines[-1]):
        return {"status": "invalid_format", "answer": None}
    if sum(line.startswith("Answer:") for line in lines) != 1:
        return {"status": "invalid_format", "answer": None}
    answer = lines[-1].split(":", 1)[1].strip()
    try:
        actual = normalize(answer, case["kind"])
    except (ValueError, ZeroDivisionError):
        return {"status": "invalid_format", "answer": answer}
    expected = [normalize(key, case["kind"]) for key in [case["answer"], *case["aliases"]]]
    result: dict[str, object] = {
        "status": "correct" if actual in expected else "wrong",
        "answer": answer,
    }
    if case["kind"] == "LINE_SET" and isinstance(actual, frozenset):
        # Additive detection dimension, kept next to the strict status.
        # status stays exact-set equality; detection answers the weaker
        # "did the model find the vulnerable span" question, which single
        # range-keyed items otherwise score zero even when the reported
        # line falls inside the key span. Graded LINE_SET results always
        # carry these keys; non-LINE_SET kinds never do; summarize() and
        # line_set_detection() tally them by question kind.
        key_sets = [key for key in expected if isinstance(key, frozenset)]
        container = next((key for key in key_sets if actual.issubset(key)), None)
        # normalize() only returns non-empty line sets, so containment of
        # any key implies detection.
        result["detection"] = container is not None
        result["span_reported"] = len(actual)
        # Size of the key span the answer fell inside; None when the answer
        # escapes every key so ratios never mix denominators.
        result["span_expected"] = len(container) if container is not None else None
        # Fraction of the containing key span the answer actually named.
        # detection alone cannot tell "found one of two" from "found both",
        # which is how every campaign COMPSEC row scored detection=True yet
        # status=wrong: the models named true lines but never all of them.
        result["span_recall"] = len(actual) / len(container) if container is not None else None
    return result


def request_json(url: str, payload: dict | None, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def native_response(raw: dict, tokenizer, prompt: str) -> dict:
    """Decode token output, accounting for reasoning opened by the template."""
    text = tokenizer.decode(raw["output_tokens"], skip_special_tokens=False)
    if prompt.rfind("<think>") > prompt.rfind("</think>"):
        text = "<think>" + text
    # Only remove EOS, never arbitrary model prose or answer punctuation.
    eos = tokenizer.eos_token
    if eos and text.endswith(eos):
        text = text[: -len(eos)]
    reason = raw.get("finish_reason")
    if raw.get("status") != "finished":
        reason = "incomplete"
    return {
        "choices": [
            {
                "finish_reason": "length" if reason == "max_output_tokens" else reason,
                "message": {"content": text},
            }
        ]
    }


def build_answer_recovery_request(
    case: dict, original_messages: list[dict], prior_text: str, tokenizer, max_tokens: int
) -> tuple[dict, str]:
    """Build a bounded answer-only continuation for a reasoning overrun.

    Qwen can finish its private reasoning without producing the requested
    answer line, especially under greedy decoding. Re-presenting that trace as
    an assistant turn lets the model extract the answer without discarding the
    evidence already computed. This is a second, explicitly recorded QA phase;
    it is never silently mixed into the primary generation response.
    """
    followup = (
        "The previous attempt used its full reasoning budget. Return only the "
        "final answer now, exactly one line: Answer: <answer>. Do not explain."
    )
    messages = [
        original_messages[0],
        original_messages[1],
        {"role": "assistant", "content": prior_text},
        {"role": "user", "content": followup},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    payload = {
        "input_tokens": tokens,
        "max_output_tokens": max_tokens,
        "sampling": {
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "repetition_penalty": 1,
            "seed": 0,
            "ignore_eos": False,
        },
    }
    return payload, prompt


def summarize(rows: list[dict], total: int) -> dict:
    counts = Counter(row["grade"]["status"] for row in rows)
    groups = {}
    for field in ("source", "domain", "suite"):
        groups[field] = {}
        for label in sorted({row[field] for row in rows}):
            selected = [row for row in rows if row[field] == label]
            correct = sum(row["grade"]["status"] == "correct" for row in selected)
            entry = {
                "correct": correct,
                "total": len(selected),
                "accuracy": correct / len(selected),
            }
            detection = line_set_detection(selected)
            if detection is not None:
                entry["detection"] = detection
            groups[field][label] = entry

    def detection_counts(rows_subset: list[dict]) -> dict:
        detection = line_set_detection(rows_subset)
        if detection is None:
            return {}
        return {
            "detection_total": detection["total"],
            "detection_graded": detection["graded"],
            "detection_correct": detection["correct"],
            "detection_full_recall": detection["full_recall"],
            "detection_mean_recall": detection["mean_recall"],
        }

    summary = {
        "completed": len(rows),
        "planned": total,
        "status_counts": dict(counts),
        "accuracy_all_planned": counts["correct"] / total if total else None,
        "median_request_seconds": statistics.median(row["elapsed_seconds"] for row in rows)
        if rows
        else None,
        "breakdown": groups,
    }
    summary.update(detection_counts(rows))
    return summary


def line_set_detection(rows: list[dict]) -> dict | None:
    """Transparent LINE_SET detection and recall tally.

    total counts every LINE_SET row; graded counts rows whose grade carried
    the detection dimension (strictly graded answer); correct counts
    detection=True. Rows without a graded LINE_SET answer (truncated,
    invalid_format, protocol errors) stay in total so the graded rate can
    never silently inflate. Returns None when the population has no
    LINE_SET rows at all. Reads rows defensively: a malformed row counts
    toward total but can never abort a run.

    full_recall and mean_recall separate completeness from mere detection:
    detection=True only means every reported line sat inside a key span, so
    an answer naming one of four lines would otherwise be tallied as a
    detection success. Rows without a span denominator (undetected answers)
    are excluded from the mean rather than counted as zero.
    """
    line_rows = [row for row in rows if row.get("kind") == "LINE_SET"]
    if not line_rows:
        return None
    detections = []
    recalls = []
    for row in line_rows:
        grade_payload = row.get("grade") or {}
        detection = grade_payload.get("detection")
        if isinstance(detection, bool):
            detections.append(detection)
        recall = grade_payload.get("span_recall")
        if isinstance(recall, (int, float)) and not isinstance(recall, bool):
            recalls.append(float(recall))
    return {
        "total": len(line_rows),
        "graded": len(detections),
        "correct": sum(1 for detection in detections if detection),
        "full_recall": sum(1 for recall in recalls if recall >= 1.0),
        "mean_recall": sum(recalls) / len(recalls) if recalls else None,
    }


LOOP_SUSPECT_RUN = 20
REPETITION_SCAN_CHAR_LIMIT = 1_000_000


def scan_repetition(text: str | None) -> dict:
    """Repetition diagnostics for long primary outputs.

    Two granularities: blank-line-separated paragraphs, and stripped
    lines. Paragraph runs catch the one observed loop pathology (axq6
    OlympiadBench:1606, 354 repeats / 310 consecutive run against <=2 / 1
    on genuine long reasoning); line runs catch loops that repeat without
    blank lines and would otherwise merge into one paragraph. loop_suspect
    fires when either run exceeds LOOP_SUSPECT_RUN; it is diagnostic only
    and never affects grading. Never raises: a diagnostic must not fail a
    run.
    """
    if not text:
        return {}
    if len(text) > REPETITION_SCAN_CHAR_LIMIT:
        return {"repetition_scan_error": f"skipped: text over {REPETITION_SCAN_CHAR_LIMIT} chars"}
    try:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return {}
        max_repeat = max(Counter(paragraphs).values())
        max_run = run = 1
        for previous_p, current_p in zip(paragraphs, paragraphs[1:]):
            run = run + 1 if previous_p == current_p else 1
            max_run = max(max_run, run)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        max_line_run = line_run = 1
        for previous_line, current_line in zip(lines, lines[1:]):
            line_run = line_run + 1 if previous_line == current_line else 1
            max_line_run = max(max_line_run, line_run)
        return {
            "repetition_max_repeat": max_repeat,
            "repetition_max_run": max_run,
            "repetition_max_line_run": max_line_run,
            "loop_suspect": max(max_run, max_line_run) > LOOP_SUSPECT_RUN,
        }
    except Exception as exc:  # diagnostics must never fail a run
        return {"repetition_scan_error": f"{type(exc).__name__}: {exc}"}


def write_json(path: Path, value: dict):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="External ds4 checkout; import data only")
    parser.add_argument("--questions", type=Path, help="Previously imported question JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:31492")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--suite", choices=["all", "core", "hard", "hard-smoke"], default="all")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--answer-reserve-tokens",
        type=int,
        default=512,
        help="Reserve output tokens for the final answer (requires native support)",
    )
    parser.add_argument(
        "--answer-recovery-tokens",
        type=int,
        default=1024,
        help="Bounded answer-only continuation after a reasoning overrun",
    )
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a verified incomplete run")
    parser.add_argument(
        "--tokenizer-dir", type=Path, help="Use native token API and this official template"
    )
    args = parser.parse_args()
    if bool(args.source) == bool(args.questions):
        parser.error("Supply exactly one of --source or --questions")
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("Budgets must be positive")
    if not 0 <= args.answer_reserve_tokens < args.max_tokens:
        parser.error("Answer reserve must be nonnegative and below the output budget")
    if args.answer_recovery_tokens <= 0:
        parser.error("Answer recovery budget must be positive")
    bank = load_questions(args.source) if args.source else json.loads(args.questions.read_text())
    cases = [
        case
        for case in bank["cases"]
        if args.suite == "all"
        or case["suite"] == args.suite
        or args.suite == "hard-smoke"
        and case["hard_smoke"]
    ]
    if not cases:
        parser.error("No questions selected")
    # Fail before serving if any answer key cannot be graded.
    for case in cases:
        for value in [case["answer"], *case["aliases"]]:
            normalize(value, case["kind"])
    if args.resume:
        if args.import_only:
            parser.error("Cannot resume an import-only command")
        if json.loads((args.output / "questions.json").read_text()) != bank:
            raise ValueError("Resume question bank mismatch")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "questions.json", bank)
    if args.import_only:
        print(f"Imported {len(cases)} questions", flush=True)
        return
    tokenizer = None
    if args.tokenizer_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    cases.sort(key=lambda case: not case["hard_smoke"])  # stable, fixed smoke first
    manifest = {
        "schema": "ax.capability_qa.v1",
        "harness_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("run_ds4_qa.py", "ds4_questions.py")
        },
        "created_at": datetime.now(UTC).isoformat(),
        "source_hashes": bank["source_hashes"],
        "suite": args.suite,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "thinking": True,
        "answer_reserve_tokens": args.answer_reserve_tokens,
        "answer_recovery_tokens": args.answer_recovery_tokens,
        "answer_recovery": "continuation_v1" if tokenizer else None,
        "question_order": [case["key"] for case in cases],
        "system": SYSTEM,
        "api": "native_generate" if tokenizer else "chat_completions",
        "runtime_before": request_json(args.base_url + "/v1/runtime", None, 30),
        "models": request_json(args.base_url + "/v1/models", None, 30),
    }
    previous = []
    if args.resume:
        saved = json.loads((args.output / "manifest.json").read_text())
        saved.setdefault("answer_reserve_tokens", 0)
        for key in manifest:
            if key != "created_at" and saved.get(key) != manifest[key]:
                raise ValueError(f"Resume manifest mismatch: {key}")
        files = sorted(args.output.glob("[0-9][0-9][0-9].json"))
        if [path.name for path in files] != [f"{i:03d}.json" for i in range(len(files))]:
            raise ValueError("Resume artifacts have gaps")
        previous = [json.loads(path.read_text()) for path in files]
        if len(previous) > len(cases):
            raise ValueError("Too many resume rows")
        write_json(args.output / f"resume-{len(previous):03d}.json", manifest)
    else:
        write_json(args.output / "manifest.json", manifest)
    rows = []
    for index, case in enumerate(cases):
        payload = build_request(case, args.model, args.max_tokens, args.answer_reserve_tokens)
        started = time.monotonic()
        row = {key: case[key] for key in ("key", "source", "domain", "suite", "kind", "answer")}
        row["chat_request"] = payload
        prompt = None
        if tokenizer:
            prompt = tokenizer.apply_chat_template(
                payload["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            payload = {
                "input_tokens": tokens,
                "max_output_tokens": args.max_tokens,
                "sampling": {
                    "temperature": 0,
                    "top_k": 0,
                    "top_p": 1,
                    "repetition_penalty": 1,
                    "seed": 0,
                    "ignore_eos": False,
                },
            }
            if args.answer_reserve_tokens:
                payload["metadata"] = json.dumps(
                    {"ax_answer_reserve_tokens": args.answer_reserve_tokens}
                )
            row["rendered_prompt"] = prompt
            row["prompt_token_sha256"] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        row["request"] = payload
        row["request_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        if index < len(previous):
            saved = previous[index]
            if saved["key"] != case["key"] or saved["request"] != payload:
                raise ValueError("Resume request mismatch")
            decoded = saved.get("decoded_response", saved.get("response", {}))
            if saved["grade"] != grade(case, decoded):
                raise ValueError("Resume grade mismatch or failed transport")
            rows.append(saved)
            continue
        try:
            endpoint = "/v1/generate" if tokenizer else "/v1/chat/completions"
            response = request_json(args.base_url + endpoint, payload, args.timeout)
            row["response"] = response
            if tokenizer:
                row["decoded_response"] = native_response(response, tokenizer, prompt)
            row["grade"] = grade(case, row.get("decoded_response", response))
            row["primary_grade"] = row["grade"]
            if row["grade"]["status"] in {"truncated", "invalid_format"}:
                # Loop diagnostic on the primary text: a truncated stream or a
                # stop that never produced the answer line is where runaway
                # repetition surfaces. Native mode decodes raw tokens (same
                # decode the recovery path uses); chat mode reads the message.
                if tokenizer and isinstance(response, dict) and "output_tokens" in response:
                    primary_text = tokenizer.decode(
                        response["output_tokens"], skip_special_tokens=False
                    )
                elif isinstance(response, dict):
                    choices = response.get("choices") or [{}]
                    message = (
                        choices[0].get("message") or {} if isinstance(choices[0], dict) else {}
                    )
                    primary_text = message.get("content")
                else:
                    primary_text = None
                row.update(scan_repetition(primary_text))
            if tokenizer and row["grade"]["status"] == "truncated":
                recovery_started = time.monotonic()
                prior_text = tokenizer.decode(response["output_tokens"], skip_special_tokens=False)
                recovery_request, recovery_prompt = build_answer_recovery_request(
                    case,
                    row["chat_request"]["messages"],
                    prior_text,
                    tokenizer,
                    args.answer_recovery_tokens,
                )
                recovery_response = request_json(
                    args.base_url + "/v1/generate", recovery_request, args.timeout
                )
                recovery_decoded = native_response(recovery_response, tokenizer, recovery_prompt)
                recovery_grade = grade(case, recovery_decoded)
                row.update(
                    recovery_request=recovery_request,
                    recovery_request_sha256=hashlib.sha256(
                        json.dumps(recovery_request, sort_keys=True).encode()
                    ).hexdigest(),
                    recovery_response=recovery_response,
                    recovery_decoded_response=recovery_decoded,
                    recovery_grade=recovery_grade,
                    recovery_elapsed_seconds=time.monotonic() - recovery_started,
                )
                if recovery_grade["status"] in {"correct", "wrong", "invalid_format"}:
                    row["decoded_response"] = recovery_decoded
                    row["grade"] = recovery_grade
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["grade"] = {"status": "transport_error", "answer": None}
        row["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / f"{index:03d}.json", row)
        rows.append(row)
        write_json(args.output / "summary.json", summarize(rows, len(cases)))
        print(
            f"{index + 1}/{len(cases)} {case['key']} {row['grade']['status']} "
            f"{row['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if row["grade"]["status"] == "transport_error":
            # A timed-out request may still hold the GPU; never overlap a retry.
            raise SystemExit("Stopped after transport error; inspect server before resuming")
    write_json(
        args.output / "runtime_after.json", request_json(args.base_url + "/v1/runtime", None, 30)
    )


if __name__ == "__main__":
    main()
