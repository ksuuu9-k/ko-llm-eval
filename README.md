# ko-llm-eval

Korean LLM service quality evaluation orchestration layer with multi-judge scoring.

`ko-llm-eval` is designed for evaluating real Korean LLM application outputs, not just base models.  
It evaluates the full service response unit:

- `prompt`
- `context`
- `answer`

and aggregates multiple judge opinions into:

- final score
- confidence
- agreement
- failure patterns

## Why This Exists

Many LLM evaluation setups have three common weaknesses:

1. They are English-first.
2. They rely too heavily on a single judge model.
3. They evaluate model outputs in isolation rather than service-level behavior.

That becomes especially limiting for Korean products, where quality often depends on:

- speech level consistency
- natural Korean phrasing
- context grounding
- instruction following

`ko-llm-eval` aims to make those signals explicit and composable.

## Core Ideas

### Service-Level Evaluation

The main evaluation unit is not just an answer string.  
It is the combination of:

- user prompt
- retrieved or system context
- final answer

### Multi-Judge Evaluation

A single answer can be evaluated by multiple independent judges.

### Aggregation

Judge outputs are merged into metric-level and overall results using:

- weighted mean
- confidence estimation
- agreement analysis

### Failure Analysis

The system does not stop at a score.  
It also extracts interpretable failure signals such as:

- `tone_inconsistency`
- `partial_hallucination`
- `instruction_violation`

## Judge Types

There are three judge roles in this project.

### `RuleJudge`

`RuleJudge` checks explicit, deterministic signals using code-based rules.  
It does not rely on LLM judgment.

Typical examples:

- mixed honorific and casual tone
- missing sentence endings
- overly short answers
- low lexical overlap with context

Why it matters:

- fast
- reproducible
- cheap
- good for clear failure signals

### `LLMJudge` (Semantic Judge)

Semantic judges focus on meaning-level quality:

- Is the answer relevant?
- Is it grounded in context?
- Does it follow the instruction?
- Does it make sense semantically?

This is the most general-purpose judge type.

### `KoreanRubricJudge` (Rubric Judge)

Rubric judges follow an explicit evaluation rubric, especially Korean quality criteria:

- tone consistency
- naturalness
- service-style appropriateness
- Korean-specific phrasing quality

This makes them useful for evaluating qualities that are easy for native speakers to notice but hard to encode as pure rules.

### Quick Comparison

| Judge Type | Main Question | Strength |
| --- | --- | --- |
| `RuleJudge` | Did the response break a known rule? | Stable and reproducible |
| Semantic Judge | Is the response meaningfully correct and relevant? | Broad quality judgment |
| Rubric Judge | Does the response satisfy our evaluation criteria? | Controlled, policy-driven evaluation |

In practice:

- semantic judges catch meaning problems
- rubric judges catch style and rubric-quality problems
- rule judges catch explicit, repeatable violations

## Current Features

- Common schemas for evaluation input and output
- `RuleJudge`
- `LLMJudge`
- `KoreanRubricJudge`
- Multi-provider judge clients
- Multi-judge registry from environment configuration
- Weighted score aggregation
- Confidence and agreement calculation
- Failure detection
- Single-run CLI
- JSONL batch execution
- JSONL result persistence
- Response format normalization and repair

## Supported Judge Providers

The project supports these provider types for LLM-backed judges:

- `openai`
- `anthropic`
- `gemini`
- `openai_compatible`
- `custom`

This means you can mix providers such as:

- OpenAI
- Claude
- Gemini
- Llama served through an OpenAI-compatible endpoint
- internal company judge APIs

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/ksuuu9-k/ko-llm-eval.git
```

For local development:

```bash
pip install -e .[dev]
```

Recommended Python version:

- Python 3.11+

## Quick Start

### CLI

```bash
ko-llm-eval version
ko-llm-eval --help
ko-llm-eval run examples/sample_request.json
ko-llm-eval batch --input examples/sample_batch.jsonl --output outputs/results.jsonl --failures-output outputs/failures.jsonl
```

### Python

```python
from ko_llm_eval.metrics.defaults import build_default_registry
from ko_llm_eval.orchestrator.evaluator import Evaluator
from ko_llm_eval.schemas import EvaluationInput

registry = build_default_registry()
evaluator = Evaluator(registry=registry)

payload = EvaluationInput(
    prompt="고객 환불 절차를 알려줘",
    context="환불은 구매 후 7일 이내 가능합니다.",
    answer="환불은 구매 후 7일 이내에 가능합니다.",
)

result = evaluator.evaluate(payload)
print(result.model_dump())
```

## Multi-Judge Environment Configuration

`build_default_registry()` can now load multiple semantic judges and multiple rubric judges from environment variables.

### Semantic Judge Pattern

```env
KO_LLM_EVAL_SEMANTIC_JUDGE_1_PROVIDER=openai
KO_LLM_EVAL_SEMANTIC_JUDGE_1_MODEL=gpt-4.1-mini
KO_LLM_EVAL_SEMANTIC_JUDGE_1_API_KEY=...
KO_LLM_EVAL_SEMANTIC_JUDGE_1_NAME=openai_semantic
KO_LLM_EVAL_SEMANTIC_JUDGE_1_WEIGHT=0.225
```

### Rubric Judge Pattern

```env
KO_LLM_EVAL_RUBRIC_JUDGE_1_PROVIDER=gemini
KO_LLM_EVAL_RUBRIC_JUDGE_1_MODEL=gemini-2.5-pro
KO_LLM_EVAL_RUBRIC_JUDGE_1_API_KEY=...
KO_LLM_EVAL_RUBRIC_JUDGE_1_NAME=gemini_rubric
KO_LLM_EVAL_RUBRIC_JUDGE_1_WEIGHT=0.175
```

### Available Per-Judge Fields

- `PROVIDER`
- `MODEL`
- `API_KEY`
- `NAME`
- `WEIGHT`
- `BASE_URL`
- `API_PATH`
- `TIMEOUT_SECONDS`
- `TEMPERATURE`
- `EXTRA_HEADERS`
- `EXTRA_BODY`

### Example Multi-Judge Setup

```env
KO_LLM_EVAL_SEMANTIC_JUDGE_1_PROVIDER=openai
KO_LLM_EVAL_SEMANTIC_JUDGE_1_MODEL=gpt-4.1-mini
KO_LLM_EVAL_SEMANTIC_JUDGE_1_API_KEY=...
KO_LLM_EVAL_SEMANTIC_JUDGE_1_NAME=openai_semantic

KO_LLM_EVAL_SEMANTIC_JUDGE_2_PROVIDER=anthropic
KO_LLM_EVAL_SEMANTIC_JUDGE_2_MODEL=claude-3-5-sonnet-latest
KO_LLM_EVAL_SEMANTIC_JUDGE_2_API_KEY=...
KO_LLM_EVAL_SEMANTIC_JUDGE_2_NAME=claude_semantic

KO_LLM_EVAL_RUBRIC_JUDGE_1_PROVIDER=gemini
KO_LLM_EVAL_RUBRIC_JUDGE_1_MODEL=gemini-2.5-pro
KO_LLM_EVAL_RUBRIC_JUDGE_1_API_KEY=...
KO_LLM_EVAL_RUBRIC_JUDGE_1_NAME=gemini_rubric
```

With that setup, `build_default_registry()` will compose:

- one `RuleJudge`
- all configured semantic judges
- all configured rubric judges

## Separate Semantic and Rubric Judge Loading

You can also load them separately.

```python
from ko_llm_eval.config import load_settings
from ko_llm_eval.metrics.defaults import (
    build_default_registry,
    build_rubric_judges,
    build_semantic_judges,
)

settings = load_settings()

semantic_judges = build_semantic_judges(settings, "tone")
rubric_judges = build_rubric_judges(settings, "tone")
registry = build_default_registry(settings)
```

## Response Format Guarantees

Regardless of provider, judge outputs are normalized into this internal format:

```json
{
  "score": 0.82,
  "confidence": 0.74,
  "reasoning": "short explanation",
  "tags": ["partial_grounding"]
}
```

The system also includes response-format hardening:

- strict JSON prompting
- code-fence extraction
- alias-key normalization
- 5-point and 10-point scale normalization
- string tag normalization
- one automatic repair retry for malformed responses

## Output Example

```json
{
  "overall_score": 0.84,
  "confidence": 0.87,
  "agreement": "high",
  "metrics": {
    "tone": {
      "aggregated_score": 0.91,
      "confidence": 0.92,
      "agreement": "high",
      "judges": [
        {
          "judge_name": "rule",
          "metric": "tone",
          "score": 0.95,
          "confidence": 0.9,
          "reasoning": "존댓말 톤이 비교적 일관적으로 유지되었습니다.",
          "tags": [],
          "raw": {}
        }
      ]
    }
  },
  "failures": []
}
```

## Project Structure

```text
src/ko_llm_eval/
  schemas/
  judges/
  metrics/
  orchestrator/
  aggregator/
  failure_detection/
  storage/
  cli/
```

## Documentation

- Example usage: [docs/example.md](docs/example.md)
- Korean rubric draft: [docs/rubric_ko.md](docs/rubric_ko.md)
- Development plan: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
- Environment example: [.env.example](.env.example)

## Testing

```bash
python -m pytest -p no:cacheprovider
```

## Repository Hygiene

Before pushing to GitHub, make sure you do not commit:

- real `.env`
- API keys
- local cache files
- build artifacts
- local test outputs

This repository includes a `.gitignore` to help with that.
