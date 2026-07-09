#!/usr/bin/env python3
"""Gate the MLA prefix-restore kill-switch retirement decision."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
EVIDENCE_CHECKER_PATH = SCRIPT_DIR / "check_mla_prefix_restore_evidence.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_mla_prefix_restore_evidence", EVIDENCE_CHECKER_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
evidence_checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = evidence_checker
MODULE_SPEC.loader.exec_module(evidence_checker)

DEFAULT_ARTIFACTS: tuple[Path, ...] = ()
DEFAULT_REQUIRED_FAMILIES = ("glm", "deepseek")
DEFAULT_REQUIRED_MODES = ("warm_extend", "warm_repeat")
KEEP_GUARDRAIL = "keep_guardrail"
RETIRE_SWITCH = "retire_switch"


class MlaPrefixRestoreRetirementError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetirementEvidence:
    path: Path
    family: str
    mode: str
    model_id: str
    prompts_total: int
    warm_hit_count: int
    warm_reused_tokens: int


@dataclass(frozen=True)
class RetirementDecision:
    decision: str
    evidence: list[RetirementEvidence]
    missing_requirements: list[str]


def normalize_items(values: list[str] | None, defaults: tuple[str, ...]) -> list[str]:
    items = values if values else list(defaults)
    normalized = []
    for item in items:
        value = item.strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def classify_family(model_id: str, required_families: list[str]) -> str | None:
    lowered = model_id.lower()
    for family in required_families:
        if family in lowered:
            return family
    return None


def decide_mla_prefix_restore_retirement(
    artifacts: list[Path],
    *,
    required_families: list[str],
    required_modes: list[str],
    min_prompts: int = 5,
) -> RetirementDecision:
    evidence: list[RetirementEvidence] = []
    for artifact in artifacts:
        try:
            summary = evidence_checker.validate_artifact(
                artifact,
                min_prompts=min_prompts,
                require_default_path=True,
                model_substring="",
                expected_mode=None,
            )
        except evidence_checker.MlaPrefixRestoreEvidenceError as error:
            raise MlaPrefixRestoreRetirementError(str(error)) from error
        if summary.mode not in required_modes:
            raise MlaPrefixRestoreRetirementError(
                f"{artifact} has unsupported evidence mode {summary.mode!r}; "
                f"expected one of {', '.join(required_modes)}"
            )
        family = classify_family(summary.model_id, required_families)
        if family is None:
            raise MlaPrefixRestoreRetirementError(
                f"{artifact} model_id {summary.model_id!r} does not match required "
                f"families: {', '.join(required_families)}"
            )
        evidence.append(
            RetirementEvidence(
                path=artifact,
                family=family,
                mode=summary.mode,
                model_id=summary.model_id,
                prompts_total=summary.prompts_total,
                warm_hit_count=summary.warm_hit_count,
                warm_reused_tokens=summary.warm_reused_tokens,
            )
        )

    covered = {(item.family, item.mode) for item in evidence}
    missing = [
        f"{family}:{mode}"
        for family in required_families
        for mode in required_modes
        if (family, mode) not in covered
    ]
    decision = RETIRE_SWITCH if not missing else KEEP_GUARDRAIL
    return RetirementDecision(
        decision=decision,
        evidence=evidence,
        missing_requirements=missing,
    )


def check_mla_prefix_restore_retirement(
    artifacts: list[Path],
    *,
    expect_decision: str,
    required_families: list[str],
    required_modes: list[str],
    min_prompts: int = 5,
) -> RetirementDecision:
    decision = decide_mla_prefix_restore_retirement(
        artifacts,
        required_families=required_families,
        required_modes=required_modes,
        min_prompts=min_prompts,
    )
    if decision.decision != expect_decision:
        if expect_decision == KEEP_GUARDRAIL:
            raise MlaPrefixRestoreRetirementError(
                "MLA prefix-restore evidence now supports retiring the kill switch; "
                "update the PRD and remove AX_DISABLE_MLA_PREFIX_RESTORE deliberately"
            )
        raise MlaPrefixRestoreRetirementError(
            "MLA prefix-restore evidence is not sufficient to retire the kill switch: "
            f"missing {', '.join(decision.missing_requirements)}"
        )
    return decision


def summarize_evidence(evidence: list[RetirementEvidence]) -> str:
    if not evidence:
        return "0 artifacts"
    return "; ".join(
        (
            f"{item.family}:{item.mode} model={item.model_id} "
            f"prompts={item.prompts_total} hits={item.warm_hit_count} "
            f"reused_tokens={item.warm_reused_tokens}"
        )
        for item in evidence
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        dest="artifacts",
        help=(
            "MLA prefix-reuse equivalence artifact. If omitted, no current "
            "retirement evidence is assumed."
        ),
    )
    parser.add_argument(
        "--expect-decision",
        choices=[KEEP_GUARDRAIL, RETIRE_SWITCH],
        default=KEEP_GUARDRAIL,
    )
    parser.add_argument(
        "--required-family",
        action="append",
        dest="required_families",
        help="Required MLA model-family substring. Defaults to glm and deepseek.",
    )
    parser.add_argument(
        "--required-mode",
        action="append",
        dest="required_modes",
        help="Required evidence mode. Defaults to warm_extend and warm_repeat.",
    )
    parser.add_argument("--min-prompts", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = args.artifacts if args.artifacts else list(DEFAULT_ARTIFACTS)
    if not artifacts:
        print("[skip] no current MLA prefix-restore retirement artifacts declared")
        return 0
    required_families = normalize_items(
        args.required_families,
        DEFAULT_REQUIRED_FAMILIES,
    )
    required_modes = normalize_items(args.required_modes, DEFAULT_REQUIRED_MODES)
    try:
        decision = check_mla_prefix_restore_retirement(
            artifacts,
            expect_decision=args.expect_decision,
            required_families=required_families,
            required_modes=required_modes,
            min_prompts=args.min_prompts,
        )
    except MlaPrefixRestoreRetirementError as error:
        print(f"MLA prefix-restore retirement check failed: {error}", file=sys.stderr)
        return 1
    missing = (
        ", ".join(decision.missing_requirements)
        if decision.missing_requirements
        else "none"
    )
    print(
        "MLA prefix-restore retirement check passed: "
        f"decision={decision.decision}; "
        f"missing={missing}; "
        f"{summarize_evidence(decision.evidence)}"
    )
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
