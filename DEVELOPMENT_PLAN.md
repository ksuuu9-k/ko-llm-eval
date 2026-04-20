# ko-llm-eval 개발 계획

## 1. 개발 목표

`ko-llm-eval`의 목표는 한국어 LLM 서비스 품질을 평가하기 위한 멀티 저지 기반 오케스트레이션 레이어를 구축하는 것이다. 이 시스템은 단일 모델의 점수 산출이 아니라, 여러 judge가 각각의 관점에서 응답을 평가하고 이를 통합해 신뢰 가능한 최종 점수와 실패 유형을 제공하는 데 초점을 둔다.

핵심 개발 방향은 다음과 같다.

- 모델 자체보다 서비스 응답 품질을 평가한다.
- 한국어 특화 metric을 기본 내장한다.
- 평가 결과의 점수뿐 아니라 신뢰도와 disagreement를 함께 제공한다.
- MVP는 빠르게 검증하고, 이후 calibration과 dataset 축적으로 확장한다.

## 2. MVP 범위

1주 MVP에서는 아래 범위만 구현한다.

- Judge 3종
- `LLMJudge`
- `RuleJudge`
- `KoreanRubricJudge`
- Metric 3종
- `tone`
- `fluency`
- `grounding`
- Aggregation
- judge별 가중 평균
- 분산 기반 confidence 계산
- 간단한 disagreement 등급화
- Failure detection
- `tone_inconsistency`
- `instruction_violation`
- `partial_hallucination`
- 실행 인터페이스
- Python library API
- 로컬 CLI
- 결과 JSON 출력

MVP에서는 아래 항목은 제외한다.

- 웹 대시보드
- 실시간 모니터링
- human eval calibration UI
- 장기 로그 분석 시스템
- 복수 도메인 judge 고도화

## 3. 사용자 시나리오

### 시나리오 A: 서비스 응답 단건 평가

사용자는 `prompt`, `context`, `answer`를 입력한다. 시스템은 각 metric에 대해 여러 judge를 실행하고, aggregated score와 confidence를 반환한다.

### 시나리오 B: 배치 평가

사용자는 JSONL 또는 CSV 형태의 평가 입력셋을 넣는다. 시스템은 각 row를 평가하고 결과를 파일로 저장한다.

### 시나리오 C: 실패 원인 분석

낮은 점수의 응답에 대해 시스템은 실패 태그와 judge별 사유를 함께 반환한다.

## 4. 아키텍처 제안

MVP 기준의 권장 아키텍처는 다음과 같다.

1. `schemas`
   평가 입력/출력 데이터 모델 정의

2. `judges`
   각 judge 구현

3. `metrics`
   metric 정의와 judge 매핑

4. `orchestrator`
   metric별 judge 실행 및 결과 수집

5. `aggregator`
   점수 통합, confidence 계산, agreement 산출

6. `failure_detection`
   failure tag 추출

7. `storage`
   결과 저장 인터페이스

8. `cli`
   실행 엔트리포인트

## 5. 권장 디렉터리 구조

```text
ko-llm-eval/
  src/
    ko_llm_eval/
      __init__.py
      config.py
      schemas/
        evaluation.py
      judges/
        base.py
        llm_judge.py
        rule_judge.py
        korean_rubric_judge.py
      metrics/
        registry.py
        tone.py
        fluency.py
        grounding.py
      orchestrator/
        evaluator.py
      aggregator/
        weighted_mean.py
        confidence.py
        agreement.py
      failure_detection/
        detector.py
      storage/
        base.py
        jsonl.py
      cli/
        main.py
  tests/
    unit/
    integration/
    fixtures/
  examples/
    sample_request.json
    sample_batch.jsonl
  docs/
    rubric_ko.md
    api_examples.md
  pyproject.toml
  README.md
```

## 6. 핵심 컴포넌트 설계

### 6.1 평가 입력 스키마

입력 단위는 서비스 평가 관점에서 아래 필드를 기본으로 한다.

```python
class EvaluationInput(BaseModel):
    prompt: str
    context: str | None = None
    answer: str
    metadata: dict = {}
```

추후 도메인별 확장을 위해 `metadata`를 열어 둔다.

### 6.2 Judge 인터페이스

judge는 특정 metric에 대해 점수와 근거를 반환해야 한다.

```python
class BaseJudge(Protocol):
    name: str

    def evaluate(
        self,
        metric: str,
        evaluation_input: EvaluationInput,
    ) -> JudgeResult:
        ...
```

반환값 예시는 아래와 같다.

```python
class JudgeResult(BaseModel):
    judge_name: str
    metric: str
    score: float
    confidence: float | None = None
    reasoning: str | None = None
    tags: list[str] = []
    raw: dict = {}
```

### 6.3 Judge별 역할

#### `LLMJudge`

- semantic fidelity, grounding, fluency 같이 규칙화가 어려운 항목 평가
- rubric prompt 기반 점수 산출
- 출력은 가능한 한 구조화된 JSON으로 강제

#### `RuleJudge`

- 형식 준수, 말투 혼용, 길이 제한, 금칙어 등 명시 규칙 판정
- deterministic 구현
- 빠르고 재현성 높은 judge로 사용

#### `KoreanRubricJudge`

- 한국어 자연스러움, 문체 일관성, 조사/어미로 인한 의미 왜곡 감지
- 초기에는 LLM 기반 rubric judge로 구현
- 추후 사람 평가 기반 calibration 대상

### 6.4 Metric Registry

metric과 judge를 느슨하게 결합하기 위해 registry 방식을 사용한다.

예시:

- `tone` -> `RuleJudge`, `KoreanRubricJudge`
- `fluency` -> `LLMJudge`, `KoreanRubricJudge`
- `grounding` -> `LLMJudge`, 선택적 `RuleJudge`

이 구조를 쓰면 phase 2에서 metric 추가가 쉽다.

### 6.5 Orchestrator

오케스트레이터는 다음 역할을 담당한다.

- 평가 입력 검증
- metric별 judge 목록 조회
- judge 실행
- 결과 수집
- timeout 및 예외 처리
- aggregation 호출
- 최종 출력 생성

MVP에서는 동기 실행으로 시작하되, 인터페이스는 추후 병렬 실행이 가능하도록 설계한다.

권장 방식:

- 내부 API는 `async` 친화적으로 설계
- 초기 구현은 단순 순차 실행
- phase 2에서 thread pool 또는 async 병렬화

### 6.6 Aggregator

MVP Aggregator는 아래 3단계로 단순하게 시작한다.

1. metric별 judge score를 weight와 함께 평균
2. judge score 분산으로 confidence 계산
3. judge 간 편차로 agreement 산출

예시 규칙:

- `aggregated_score = sum(score * weight) / sum(weight)`
- score variance가 낮을수록 confidence 상승
- score range 또는 표준편차 기반으로 `high`, `medium`, `low` agreement 분류

초기 weight 예시:

- `LLMJudge`: 0.45
- `RuleJudge`: 0.20
- `KoreanRubricJudge`: 0.35

이 가중치는 고정값으로 시작하고, phase 2에서 calibration 데이터 기반으로 조정한다.

### 6.7 Failure Detection

Failure detection은 rule 기반 태그 추출로 먼저 구현한다.

예시:

- grounding score가 낮고 hallucination 관련 reasoning이 있으면 `partial_hallucination`
- tone metric에서 judge 간 지적이 있으면 `tone_inconsistency`
- format 관련 rule 실패 시 `instruction_violation`

MVP에서는 explainability를 위해 black-box 모델보다 규칙 기반 합성 로직이 더 적합하다.

## 7. API / CLI 설계

### Python API 예시

```python
result = evaluator.evaluate(
    prompt="고객 환불 절차를 알려줘",
    context="환불은 구매 후 7일 이내 가능",
    answer="환불은 구매 후 7일 이내에 가능합니다.",
)
```

### CLI 예시

```bash
ko-llm-eval run --input examples/sample_request.json
ko-llm-eval batch --input examples/sample_batch.jsonl --output outputs/result.jsonl
```

### 출력 예시

```json
{
  "overall_score": 0.82,
  "confidence": 0.71,
  "agreement": "medium",
  "metrics": {
    "tone": {
      "aggregated_score": 0.91,
      "judges": [
        {"name": "gpt", "score": 0.95},
        {"name": "rule", "score": 0.89}
      ]
    }
  },
  "failures": [
    "partial_hallucination"
  ]
}
```

## 8. 기술 스택 제안

MVP 기준 추천 스택:

- Python 3.11+
- `pydantic` for schema validation
- `typer` 또는 `click` for CLI
- `pytest` for tests
- `httpx` for LLM API client
- `tenacity` for retry
- `orjson` 또는 표준 `json` for serialization

선택 사항:

- `pandas` for batch input handling
- `rich` for CLI 가독성

프레임워크는 무겁게 가져가지 않고, 라이브러리 중심 구조를 권장한다.

## 9. 1주 개발 일정

### Day 1: 프로젝트 부트스트랩

- Python 패키지 구조 생성
- 기본 스키마 정의
- CLI 엔트리포인트 생성
- Judge 인터페이스 및 Result 모델 정의
- README 초안 작성

산출물:

- 실행 가능한 프로젝트 뼈대
- 샘플 입력 파일

### Day 2: RuleJudge 구현

- tone consistency rule 설계
- 출력 형식/길이 검증 rule 추가
- 기본 failure tag 연결
- unit test 작성

산출물:

- deterministic rule judge 1차 완성

### Day 3: LLMJudge 구현

- judge prompt 템플릿 작성
- structured output parser 구현
- retry / timeout / fallback 처리
- fluency, grounding 평가 연결

산출물:

- 외부 LLM을 호출하는 judge 완성

### Day 4: KoreanRubricJudge 구현

- 한국어 특화 rubric 정의
- tone / fluency rubric prompt 정교화
- 예시 기반 테스트 케이스 추가

산출물:

- 한국어 특화 judge 완성

### Day 5: Orchestrator + Aggregator

- metric registry 구현
- judge 실행 흐름 연결
- weighted mean 구현
- confidence / agreement 계산 구현

산출물:

- end-to-end 단건 평가 가능

### Day 6: Failure detection + Batch 실행

- failure tag 추출기 구현
- JSONL batch runner 추가
- 결과 저장 로직 구현

산출물:

- 배치 평가 가능
- failure case 분석 가능

### Day 7: 안정화

- integration test 추가
- README / docs 정리
- 샘플 결과 검증
- 다음 phase backlog 정리

산출물:

- MVP 릴리스 가능 상태

## 10. 테스트 전략

테스트는 아래 3층으로 구성한다.

### Unit Test

- rule 함수 단위 검증
- aggregator 계산 검증
- failure detection 규칙 검증

### Integration Test

- 하나의 입력이 전체 pipeline을 통과하는지 검증
- mock judge를 사용해 deterministic 테스트 구성

### Golden Set Test

- 한국어 예시 데이터셋 20~50개로 baseline 결과 고정
- 변경 시 regression 여부 확인

초기 핵심은 LLM 응답 자체보다 오케스트레이션 로직의 재현성을 테스트하는 것이다.

## 11. 데이터와 루브릭 준비

MVP 성패는 코드보다 rubric 품질에 크게 좌우된다. 따라서 개발과 병행해서 아래 문서를 별도로 준비해야 한다.

- `docs/rubric_ko.md`
- tone 평가 기준
- fluency 평가 기준
- grounding 평가 기준
- 점수 구간 해석
- 좋은 예시 / 나쁜 예시

추천 방식:

- 1~5점 또는 0~1 점수 체계를 명확히 정의
- judge prompt에 rubric 전문을 넣지 말고, 핵심 기준만 구조화해서 삽입
- 동일 rubric으로 human eval도 가능한 형태로 작성

## 12. 운영 관점 비기능 요구사항

MVP에서도 아래 항목은 최소 반영이 필요하다.

- timeout 처리
- judge 실패 시 graceful degradation
- 개별 judge 결과 raw log 저장
- API key 설정 분리
- 재실행 가능성 보장

권장 설정 파일:

- `.env`
- `config.yaml` 또는 `pyproject.toml` 기반 설정

## 13. 리스크와 대응

### 리스크 1: judge 결과 흔들림

대응:

- structured output 강제
- temperature 낮게 설정
- rubric 간결화
- 동일 입력 재평가 테스트 추가

### 리스크 2: 한국어 metric 기준 불명확

대응:

- rubric 문서 먼저 작성
- 좋은/나쁜 예시를 최소 10개 이상 준비
- 사람 평가 기준과 wording 맞추기

### 리스크 3: wrapper 수준으로 보일 위험

대응:

- aggregation, disagreement, failure analysis를 핵심 가치로 강조
- 단순 judge 호출기가 아니라 평가 해석 레이어로 설계

### 리스크 4: grounding 평가 품질 부족

대응:

- context 인용 여부 검사 룰 추가
- hallucination 사례셋 구축
- human review 대상 샘플 축적

## 14. Phase 2 로드맵

MVP 이후 4~6주 확장 계획은 다음과 같다.

1. benchmark dataset 구축
2. human eval과 score calibration
3. domain judge 추가
4. 병렬 실행 및 비용 최적화
5. 결과 대시보드 구축
6. 실서비스 API 형태로 외부 제공

## 15. 바로 실행할 첫 작업

이 계획 기준으로 실제 구현에 들어갈 때 가장 먼저 할 작업은 아래 순서가 적절하다.

1. 프로젝트 패키지 골격 생성
2. 평가 입출력 스키마 정의
3. Judge 인터페이스와 mock judge 구현
4. Orchestrator와 Aggregator 뼈대 연결
5. RuleJudge 먼저 완성
6. LLMJudge와 KoreanRubricJudge 연결
7. 테스트와 샘플셋 추가

## 16. 성공 기준

MVP 성공 기준은 아래처럼 정의한다.

- 단건 평가와 배치 평가가 모두 동작한다.
- 3개 metric에 대해 judge별 결과와 최종 score가 JSON으로 반환된다.
- disagreement와 confidence가 함께 계산된다.
- 대표 failure tag가 자동 추출된다.
- 최소 20개 이상의 한국어 샘플셋에서 end-to-end 테스트가 가능하다.

## 17. 한 줄 요약

`ko-llm-eval`의 MVP는 "한국어 특화 rubric + multi-judge + aggregation"을 최소 단위로 빠르게 구현해, 단순 점수기가 아닌 신뢰 가능한 서비스 품질 평가 레이어를 만드는 데 집중해야 한다.
