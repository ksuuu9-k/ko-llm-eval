# ko-llm-eval Examples

README로 돌아가기: [../README.md](../README.md)

## 1. 기본 단건 평가

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
    metadata={"domain": "customer_support"},
)

result = evaluator.evaluate(payload)
print(result.model_dump_json(indent=2))
```

## 2. semantic judge와 rubric judge를 따로 확인하기

```python
from ko_llm_eval.config import load_settings
from ko_llm_eval.metrics.defaults import (
    build_rubric_judges,
    build_semantic_judges,
)

settings = load_settings()

semantic_judges = build_semantic_judges(settings, "tone")
rubric_judges = build_rubric_judges(settings, "tone")

print([judge.name for judge in semantic_judges])
print([judge.name for judge in rubric_judges])
```

## 3. 여러 judge를 .env로 구성하기

예시:

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

이 상태에서:

```python
from ko_llm_eval.metrics.defaults import build_default_registry

registry = build_default_registry()

tone_judges = registry.get_judges("tone")
print([judge.name for judge in tone_judges])
```

예상 결과:

```python
["rule", "openai_semantic", "claude_semantic", "gemini_rubric"]
```

## 4. 배치 평가 예시

```python
from pathlib import Path

from ko_llm_eval.metrics.defaults import build_default_registry
from ko_llm_eval.orchestrator.batch import evaluate_batch, load_jsonl_inputs
from ko_llm_eval.orchestrator.evaluator import Evaluator
from ko_llm_eval.storage import JsonlResultWriter

inputs = load_jsonl_inputs(Path("examples/sample_batch.jsonl"))
evaluator = Evaluator(registry=build_default_registry())
writer = JsonlResultWriter(Path("outputs/results.jsonl"))

try:
    results = evaluate_batch(evaluator, inputs, writer=writer)
finally:
    writer.close()

print(f"processed={len(results)}")
```

## 5. CLI 예시

단건 실행:

```bash
ko-llm-eval run examples/sample_request.json
```

배치 실행:

```bash
ko-llm-eval batch --input examples/sample_batch.jsonl --output outputs/results.jsonl --failures-output outputs/failures.jsonl
```

## 6. provider 예시

### OpenAI

```env
KO_LLM_EVAL_SEMANTIC_JUDGE_1_PROVIDER=openai
KO_LLM_EVAL_SEMANTIC_JUDGE_1_MODEL=gpt-4.1-mini
KO_LLM_EVAL_SEMANTIC_JUDGE_1_API_KEY=...
```

### Claude

```env
KO_LLM_EVAL_SEMANTIC_JUDGE_1_PROVIDER=anthropic
KO_LLM_EVAL_SEMANTIC_JUDGE_1_MODEL=claude-3-5-sonnet-latest
KO_LLM_EVAL_SEMANTIC_JUDGE_1_API_KEY=...
KO_LLM_EVAL_SEMANTIC_JUDGE_1_BASE_URL=https://api.anthropic.com
```

### Gemini

```env
KO_LLM_EVAL_RUBRIC_JUDGE_1_PROVIDER=gemini
KO_LLM_EVAL_RUBRIC_JUDGE_1_MODEL=gemini-2.5-pro
KO_LLM_EVAL_RUBRIC_JUDGE_1_API_KEY=...
KO_LLM_EVAL_RUBRIC_JUDGE_1_BASE_URL=https://generativelanguage.googleapis.com
```

### Llama OpenAI 호환 서버

```env
KO_LLM_EVAL_SEMANTIC_JUDGE_1_PROVIDER=openai_compatible
KO_LLM_EVAL_SEMANTIC_JUDGE_1_MODEL=meta-llama/Llama-3.1-70B-Instruct
KO_LLM_EVAL_SEMANTIC_JUDGE_1_API_KEY=local_token
KO_LLM_EVAL_SEMANTIC_JUDGE_1_BASE_URL=http://localhost:8000/v1
```

### 내부 커스텀 모델

```env
KO_LLM_EVAL_RUBRIC_JUDGE_1_PROVIDER=custom
KO_LLM_EVAL_RUBRIC_JUDGE_1_MODEL=internal-rubric-v1
KO_LLM_EVAL_RUBRIC_JUDGE_1_API_KEY=internal_token
KO_LLM_EVAL_RUBRIC_JUDGE_1_BASE_URL=https://llm-gateway.internal
KO_LLM_EVAL_RUBRIC_JUDGE_1_API_PATH=/judge
KO_LLM_EVAL_RUBRIC_JUDGE_1_EXTRA_HEADERS={"x-tenant":"eval-team"}
KO_LLM_EVAL_RUBRIC_JUDGE_1_EXTRA_BODY={"team":"quality"}
```

## 7. custom provider 응답 형식

`custom` provider는 아래 형태 중 하나를 반환하면 됩니다.

직접 반환:

```json
{
  "score": 0.83,
  "confidence": 0.72,
  "reasoning": "Grounded and mostly fluent.",
  "tags": ["partial_grounding"]
}
```

중첩 반환:

```json
{
  "result": {
    "score": 0.83,
    "confidence": 0.72,
    "reasoning": "Grounded and mostly fluent.",
    "tags": ["partial_grounding"]
  }
}
```

## 8. LLMJudge를 직접 만들기

현재는 client를 직접 만들어 넣는 방식이 기본입니다.

```python
from ko_llm_eval.config import JudgeModelConfig
from ko_llm_eval.judges import LLMJudge, build_llm_client

client = build_llm_client(
    JudgeModelConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="your_api_key",
        base_url="https://api.openai.com/v1",
        name="openai_semantic",
    )
)

judge = LLMJudge(client=client, name="openai_semantic")
```

## 9. 포맷 이탈 복구

judge 응답이 아래처럼 포맷을 벗어나더라도 내부적으로 복구를 시도합니다.

- markdown code fence
- alias 키
- 문자열 숫자
- 5점/10점 척도
- 문자열 tags
- malformed 응답 1회 repair

## 10. 참고 문서

- README: [../README.md](../README.md)
- Korean rubric draft: [rubric_ko.md](rubric_ko.md)
- Development plan: [../DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md)
