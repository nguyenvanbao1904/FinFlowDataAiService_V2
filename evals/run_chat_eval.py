"""Run the chat agent against the golden dataset and report scores.

This calls DeepSeek for real — make sure DEEPSEEK_API_KEY is set and you
accept the LLM cost. Backend tools (Spring Boot) must also be reachable
or expected to fail; tool failures are surfaced in the report.

Usage:
    python -m evals.run_chat_eval
    python -m evals.run_chat_eval --case hpg_fair_value
    python -m evals.run_chat_eval --report out.md
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from app.models.chat import ChatOrchestrateRequest
from app.services.chat.orchestrator import ChatOrchestrator

_DATASET_PATH = Path(__file__).parent / "datasets" / "chat_golden.yaml"


# ── Eval input/output shapes ──────────────────────────────────────────


@dataclass
class ChatEvalInput:
    user_message: str
    user_id: str


@dataclass
class ChatEvalOutput:
    assistant_message: str
    tool_names: list[str]
    needs_clarification: bool


@dataclass
class ChatExpected:
    expected_tool_names: list[str]
    expected_assistant_contains: list[str]


# ── Custom evaluators ─────────────────────────────────────────────────


@dataclass
class ToolSelection(Evaluator[ChatEvalInput, ChatEvalOutput, ChatExpected]):
    """Score: fraction of expected tools that appeared in the agent run.

    A score of 1.0 means every expected tool was called. We don't penalize
    extra tools — the LLM may legitimately do more work than the minimum.
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[ChatEvalInput, ChatEvalOutput, ChatExpected],
    ) -> float:
        expected = set(ctx.expected_output.expected_tool_names) if ctx.expected_output else set()
        if not expected:
            return 1.0
        called = set(ctx.output.tool_names)
        hit = len(expected & called)
        return hit / len(expected)


@dataclass
class AssistantContains(Evaluator[ChatEvalInput, ChatEvalOutput, ChatExpected]):
    """Score: fraction of expected substrings present in assistant_message."""

    def evaluate(
        self,
        ctx: EvaluatorContext[ChatEvalInput, ChatEvalOutput, ChatExpected],
    ) -> float:
        needles = ctx.expected_output.expected_assistant_contains if ctx.expected_output else []
        if not needles:
            return 1.0
        msg = ctx.output.assistant_message
        return sum(1 for n in needles if n in msg) / len(needles)


# ── Dataset loader ────────────────────────────────────────────────────


def load_dataset() -> Dataset[ChatEvalInput, ChatEvalOutput, ChatExpected]:
    raw = yaml.safe_load(_DATASET_PATH.read_text(encoding="utf-8")) or []
    cases: list[Case[ChatEvalInput, ChatEvalOutput, ChatExpected]] = []
    for entry in raw:
        cases.append(Case(
            name=entry["name"],
            inputs=ChatEvalInput(
                user_message=entry["inputs"]["user_message"],
                user_id=entry["inputs"].get("user_id", "u-eval"),
            ),
            expected_output=ChatExpected(
                expected_tool_names=list(entry.get("expected_tool_names") or []),
                expected_assistant_contains=list(entry.get("expected_assistant_contains") or []),
            ),
        ))
    return Dataset(
        name="chat_golden",
        cases=cases,
        evaluators=[ToolSelection(), AssistantContains()],
    )


# ── Runner ────────────────────────────────────────────────────────────


async def _run_case(orchestrator: ChatOrchestrator, inputs: ChatEvalInput) -> ChatEvalOutput:
    request = ChatOrchestrateRequest(
        thread_id="eval-thread",
        user_id=inputs.user_id,
        user_message=inputs.user_message,
        last_messages=[],
    )
    response = await orchestrator.orchestrate(request)
    return ChatEvalOutput(
        assistant_message=response.assistant_message,
        tool_names=[tc["name"] for tc in response.tool_calls],
        needs_clarification=response.needs_clarification,
    )


async def main(case_filter: str | None = None, report_path: str | None = None) -> None:
    dataset = load_dataset()
    if case_filter:
        dataset = Dataset(
            name="chat_golden_filtered",
            cases=[c for c in dataset.cases if c.name == case_filter],
            evaluators=dataset.evaluators,
        )
        if not dataset.cases:
            raise SystemExit(f"No case matching '{case_filter}'")

    orchestrator = ChatOrchestrator()

    async def task(inputs: ChatEvalInput) -> ChatEvalOutput:
        return await _run_case(orchestrator, inputs)

    report = await dataset.evaluate(task)
    report.print(include_input=True, include_output=True, include_durations=True)

    if report_path:
        Path(report_path).write_text(str(report), encoding="utf-8")
        print(f"\nReport saved to {report_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run chat agent eval.")
    parser.add_argument("--case", help="Run a single case by name", default=None)
    parser.add_argument("--report", help="Save report to this path", default=None)
    args = parser.parse_args()
    asyncio.run(main(case_filter=args.case, report_path=args.report))


if __name__ == "__main__":
    cli()
