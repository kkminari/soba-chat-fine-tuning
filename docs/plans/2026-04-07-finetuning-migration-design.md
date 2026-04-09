# 오픈소스 파인튜닝 모델 교체 설계

> 작성일: 2026-04-07
> 목적: 챗봇 대화 생성 API를 Claude Sonnet에서 Qwen3-14B QLoRA 파인튜닝 모델로 교체

---

## 1. 교체 범위

### 교체 대상 (대화 생성 — 6개 메서드)

| 메서드 | 현재 | 변경 후 |
|--------|------|---------|
| `generate_intro()` | Claude Sonnet | Qwen3-14B FT |
| `generate_first_question()` | Claude Sonnet | Qwen3-14B FT |
| `generate_response()` | Claude Sonnet | Qwen3-14B FT |
| `generate_retry_message()` | Claude Sonnet | Qwen3-14B FT |
| `generate_ending_message()` | Claude Sonnet | Qwen3-14B FT |
| `generate_display_title()` | Claude Sonnet | Qwen3-14B FT |

### 유지 (분석/로직 — Claude Haiku)

| 메서드 | 유지 이유 |
|--------|----------|
| `analyze_response()` | 응답 분석은 논리적 판단 → API가 적합 |
| `determine_next_question_order()` | 질문 순서 결정은 별도 로직 |

### 변경하지 않는 것

- 모바일/웹 프론트엔드 (API 인터페이스 동일)
- 세션 관리 로직, DB 구조, 인증 시스템
- `analysis_service.py` (규칙 기반 사전 검증)

---

## 2. 파인튜닝 스택

| 항목 | 선택 |
|------|------| 
| 베이스 모델 | Qwen/Qwen3-14B |
| 파인튜닝 방식 | QLoRA (4bit NF4) |
| 인프라 | RunPod (A100 80GB) |
| 학습 데이터 | GPT-4o API로 합성 생성 |
| 서빙 | 추후 결정 (RunPod Serverless / vLLM / HF Endpoints) |

---

## 3. 설계 핵심: 관심사 분리

파인튜닝 모델은 **대화 톤과 질문 리프레이징**에만 집중.
질문 순서 결정, 응답 유효성 분석은 Claude Haiku에 유지.

모델 입력에 8개 질문 전체를 넣지 않고, 현재 질문 + 사용자 응답 + 다음 질문만 전달:

```
주제: {topic}
현재 질문: {current_question}
사용자 응답: {user_answer}
다음 질문: {next_question}
```

이점:
- max_seq_length 512로 충분
- 태스크 난이도 낮음 → 적은 데이터로 학습 가능
- 품질 안정성 높음

---

## 4. 학습 데이터 설계

### 통합 모델 (6개 태스크를 하나의 모델로)

system prompt의 태스크 타입으로 구분.

### 시드 데이터 (v2 — 2026-04-09 업데이트)

기존 데이터 증강 파이프라인으로 생성된 249건의 고품질 주제+질문 세트를 시드로 사용:
- 술 도메인: 144건 (원본 33 + 8차 진화 증강 111)
- 화장품 도메인: 105건 (원본 42 + 증강 63)
- 프레임워크: TPOLEACI (Time, Person, Occasion, Location, Emotion, Action, Cognition, Image)

### 데이터 규모 (v2 — 2026-04-09 업데이트)

| 태스크 | 시드 | 변형 | 건수 | 모델 | 비고 |
|--------|------|------|------|------|------|
| 코멘트 + 질문 리프레이징 | 249×8 | ×3 | ~5,976 | GPT-4o | 핵심 태스크 |
| 인트로 | 249 | ×2 | ~498 | GPT-4o-mini | 단순 |
| 첫 질문 전환 | 249 | ×2 | ~498 | GPT-4o-mini | 단순 |
| 재시도 메시지 | 249 | ×2 | ~498 | GPT-4o-mini | 패턴 제한적 |
| 종료 메시지 | 249 | ×2 | ~498 | GPT-4o-mini | 단순 |
| 제목 생성 | 249 | ×2 | ~498 | GPT-4o-mini | 단순 변환 |
| **합계** | | | **~8,466** | | **필터링 후 ~7,800건** |

예상 비용: ~$17 | 예상 시간: ~10분 (비동기)

### 데이터 포맷 (ChatML)

```json
{
  "messages": [
    {"role": "system", "content": "<태스크별 시스템 프롬프트>"},
    {"role": "user", "content": "<입력>"},
    {"role": "assistant", "content": "<기대 출력>"}
  ],
  "task_type": "response"
}
```

### 핵심 태스크 예시 (코멘트 + 리프레이징)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "당신은 친근한 설문 챗봇입니다. 사용자 응답에 공감 코멘트를 달고, 다음 질문으로 자연스럽게 전환하세요.\n\n출력 형식:\n{\"comment\": \"...\", \"next_question\": \"...\"}"
    },
    {
      "role": "user",
      "content": "주제: 맥주 소비 트렌드\n현재 질문: 맥주 선택 시 가장 중요한 요소는?\n사용자 응답: 가격이랑 맛 둘 다 중요한데 솔직히 가격을 더 보게 돼요\n다음 질문: 국산 맥주와 수입 맥주 중 선호하는 것은?"
    },
    {
      "role": "assistant",
      "content": "{\"comment\": \"가격도 중요한 요소죠! 요즘 맥주 가격이 많이 올라서 더 신경 쓰이실 것 같아요 💰\", \"next_question\": \"그러면 혹시 국산 맥주와 수입 맥주 중에서는 어떤 쪽을 더 선호하시나요? 😊\"}"
    }
  ],
  "task_type": "response"
}
```

### 다양성 확보 변수

| 변수 | 다양화 방법 |
|------|------------|
| 설문 주제 | 20~30개 (맥주, 화장품, 여행, 카페 등) |
| 사용자 응답 스타일 | 짧은/긴, 구어체/문어체, 이모지 유무 |
| 대화 진행 단계 | 초반/중반/후반 |
| 감정 톤 | 긍정/중립/부정/무관심 |

---

## 5. 합성 데이터 생성 파이프라인

### 프로젝트 구조

```
finetuning/
├── data_generation/
│   ├── generate_data.py       # GPT-4o 호출 메인 스크립트
│   ├── prompts/               # 태스크별 생성 프롬프트
│   │   ├── response.py
│   │   ├── intro.py
│   │   ├── first_question.py
│   │   ├── retry.py
│   │   ├── ending.py
│   │   └── title.py
│   ├── topics.json            # 설문 주제 + 질문 시드 데이터
│   └── validate_data.py       # 생성 데이터 품질 검증
├── configs/
│   └── training_config.yaml
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── data/
│   ├── raw/
│   └── processed/
├── .env                       # OPENAI_API_KEY, HF_TOKEN, WANDB_API_KEY
└── requirements.txt
```

### 생성 흐름

```
topics.json (주제 + 질문 세트)
  → GPT-4o API (태스크별 프롬프트 × 변수 조합)
  → raw/ 저장
  → validate_data.py (JSON 파싱, 중복, 길이 필터)
  → processed/ 최종 데이터
  → HuggingFace Datasets 업로드 (선택)
```

---

## 6. 백엔드 통합 설계

### 새 파일 구조

```
backend/app/services/
├── claude_service.py    # 기존 유지 (Haiku 분석 + 폴백용)
├── qwen_service.py      # 새로 생성 (파인튜닝 모델 호출)
└── llm_service.py       # 새로 생성 (라우팅 인터페이스)
```

### LLMService (통합 인터페이스)

```python
class LLMService:
    def __init__(self):
        self.qwen = QwenService()      # 대화 생성
        self.claude = ClaudeService()   # 응답 분석 (Haiku)
    
    # 대화 생성 → Qwen FT
    async def generate_intro(self, topic): ...
    async def generate_response(self, context, answer, next_q): ...
    # ... (6개 메서드)
    
    # 분석 → Claude Haiku 유지
    async def analyze_response(self, ...):
        return await self.claude.analyze_response(...)
    async def determine_next_question_order(self, ...):
        return await self.claude.determine_next_question_order(...)
```

### QwenService (모델 호출)

```python
class QwenService:
    def __init__(self):
        self.api_url = settings.qwen_api_url
    
    async def generate(self, system_prompt, user_input, task_type):
        response = await httpx.post(self.api_url, json={
            "model": "soba-chatbot-qwen3-14b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        })
        return response.json()
```

### config.py 추가

```python
qwen_api_url: str = ""
use_finetuned_model: bool = True  # False → Claude 폴백
```

### 변경 파일 목록

| 파일 | 변경 |
|------|------|
| `config.py` | Qwen URL, 스위치 플래그 추가 |
| `session_service.py` | ClaudeService → LLMService 교체 |
| `requirements.txt` | 변경 없음 |
| 프론트엔드 | **변경 없음** |

---

## 7. 파인튜닝 설정 (training_config.yaml)

기존 FINETUNING_SETUP_GUIDE.md 기반:

```yaml
model:
  name: "Qwen/Qwen3-14B"
  max_seq_length: 512

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  bf16: true
```

---

## 8. 실행 로드맵

### Phase 1: 데이터 준비 (1~2일)

- finetuning/ 프로젝트 구조 생성
- topics.json 작성 (questions.db 기존 데이터 활용 + 추가 주제)
- 태스크별 GPT-4o 생성 프롬프트 작성
- generate_data.py 구현 및 실행
- validate_data.py 품질 검증
- ~1,850건 데이터셋 확정

### Phase 2: 파인튜닝 (1일)

- training_config.yaml, data_loader.py 설정
- RunPod A100 80GB에서 학습 (~2~3시간)
- WandB loss 모니터링
- evaluate.py 품질 평가 (JSON 파싱률 95%+ 목표)
- 선택: 7B와 14B 비교 실험

### Phase 3: 서빙 + 백엔드 통합 (1일)

- 서빙 환경 결정 및 배포
- qwen_service.py, llm_service.py 구현
- session_service.py 의존성 교체
- 로컬 테스트

### Phase 4: 검증 (1일)

- Claude vs Qwen FT A/B 비교 (20개 시나리오)
- 전체 세션 플로우 테스트 (8개 질문 완주)
- 엣지 케이스 확인 (짧은 응답, 부정적 톤, JSON 안정성)
- 품질 미달 시 데이터 보강 → Phase 2 반복

### 예상 비용

| 항목 | 비용 |
|------|------|
| GPT-4o 데이터 생성 | ~$5 |
| RunPod A100 학습 (3시간) | ~$10 |
| 서빙 (월간) | 서빙 방식에 따라 상이 |
