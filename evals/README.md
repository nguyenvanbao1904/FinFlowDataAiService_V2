# Evals — pydantic-evals scaffold for FinFlow chat agent

Evaluates the chat orchestrator across a curated set of representative
prompts (the "golden dataset"). Designed to catch regressions when:

- Changing the agent system prompt
- Switching DeepSeek model versions
- Modifying tool descriptions
- Upgrading pydantic-ai

## Layout

```
evals/
├── README.md              ← this file
├── datasets/
│   └── chat_golden.yaml   ← golden cases (prompt → expected tools/intent)
└── run_chat_eval.py       ← run evaluator + print report
```

## Running

```bash
# Set DEEPSEEK_API_KEY in .env first.
python -m evals.run_chat_eval

# Run a single case:
python -m evals.run_chat_eval --case investment_basic

# Save markdown report:
python -m evals.run_chat_eval --report out.md
```

## RAGAS Benchmark

Run the production annual-report RAG benchmark with the current best combo:

- Embedding: `voyage-3.5-lite`
- Rerank: `rerank-2.5-lite`
- Retrieval: Qdrant vector search + SQLite FTS/BM25
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall

```bash
# Default: 50 generated questions
evals/run_rag_ragas

# Run exactly 30 or 50 questions
evals/run_rag_ragas 30
evals/run_rag_ragas 50

# Reuse a fixed testset for reproducible comparisons
evals/run_rag_ragas 50 --testset artifacts/rag_eval/ragas/<run-id>/testset.json
```

Outputs are saved to `artifacts/rag_eval/ragas/<timestamp>/`:

- `summary.json`
- `scores.csv`
- `testset.json`
- `ragas_dataset.json`
- `rag_outputs.json`

## Adding a case

Edit `datasets/chat_golden.yaml`:

```yaml
- name: my_new_case
  inputs:
    user_message: "Your prompt here"
    user_id: "u-eval"
  expected_tool_names:
    - get_company_metrics
  expected_assistant_contains:
    - "VND"
```

Re-run; the dataset is the source of truth.

## What gets checked

For each case the evaluator runs the chat agent against DeepSeek (real LLM
call, no mocks) and verifies:

1. **Tool selection** — did the agent call the expected tools?
2. **Output content** — does the response contain expected substrings?
3. **No catastrophic failure** — usage_limits respected, no unhandled errors.

This is NOT a unit test (those live in `tests/`). Evals are expensive
(real LLM cost) and run before releases or when changing prompts.
