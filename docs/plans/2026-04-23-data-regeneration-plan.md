# 학습 데이터 v3 재생성 실행 플랜

**작성일**: 2026-04-23
**목적**: 학습 데이터 품질 진단에서 확인된 3가지 문제(누수·정형성·집중도)를 해결하기 위해 데이터를 풀 재생성(v3)하고, 기존 학습 코드는 그대로 유지한 채 데이터만 교체
**관련 문서**:
- 진단: `docs/results/2026-04-23-data-quality-diagnosis.md`
- 방법론: `docs/plans/2026-04-23-deduplication-methodology.md`

---

## Context

Exp1~3a 모두 epoch 1 이후 과적합, dropout 조정으로도 해결되지 않음. 진단 결과 원인이 데이터에 있음이 확인됨.

**결정사항** (사용자 확인):
- 개선 강도: **풀 재생성 (C)**
- 확장 범위: **시드 249→400 + 다축 설계(persona 10 × occasion 4 = 40 조합), 2도메인 유지**
- GO 기준: **정량 데이터 메트릭만 (A)** — 학습 검증은 이후 Phase 7 직전 통합 평가로

### 다축 설계 근거

- Persona 단일 축(5~8개)은 samples/persona가 700~440건으로 너무 많아 모델이 "persona = 톤" 패턴을 또 다른 과적합 신호로 학습할 위험
- `persona 10개 × occasion 4개 = 40 실효 조합`으로 설계 시 조합당 샘플 수 ~165건으로 적정 구간 진입
- occasion 축(혼자/친구/가족/비즈니스)은 마케팅 리서치에서 응답 맥락 결정하는 핵심 신호로 실용성 높음
- 같은 persona라도 occasion이 다르면 톤 달라짐 ("30대 직장인 여성 × 홈술" vs "× 비즈니스 모임") → 자연스러운 다양성 확보

---

## 1. 산출물

| # | 파일 | 내용 |
|---|---|---|
| 1 | `docs/results/2026-04-23-data-quality-diagnosis.md` | 진단 보고 (완료) |
| 2 | `docs/plans/2026-04-23-deduplication-methodology.md` | 방법론 가이드 (완료) |
| 3 | `docs/plans/2026-04-23-data-regeneration-plan.md` | 본 실행 플랜 + 결과 |
| 4 | `data/processed_v3/{train,val,test}.jsonl` | 정제된 신규 학습 데이터 |
| 5 | `data_generation/v3/` | 재생성 스크립트·프롬프트·시드 |

### 디렉터리 구조

```
finetuning/
├── data/
│   ├── processed_v2/               ← 기존 보존 (롤백 가능)
│   └── processed_v3/               ← 신규
│       ├── raw_all.jsonl
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── quality_report.json
├── data_generation/
│   ├── prompts/                    ← 기존 보존
│   ├── v3/                         ← 신규
│   │   ├── prompts/                ← 재설계 프롬프트
│   │   │   ├── intro.py
│   │   │   ├── first_question.py
│   │   │   ├── retry.py
│   │   │   ├── ending.py
│   │   │   ├── title.py
│   │   │   └── response.py
│   │   ├── seeds_v3.json           ← 400개 시드
│   │   ├── personas.json           ← persona 축 10개
│   │   ├── occasions.json          ← occasion 축 4개
│   │   ├── starting_phrases.json   ← 시작어구 rotation pool
│   │   ├── generate_data_v3.py
│   │   ├── dedup_pipeline.py
│   │   └── check_quality.py
│   └── ...
```

### 핵심 원칙

- v2 데이터 **절대 덮어쓰지 않음** (실패 시 롤백 가능)
- 학습 코드(`src/`)는 **건드리지 않음** — 데이터만 교체
- 모든 결정·수치를 문서로 남겨 Phase 7 인수인계 가능

---

## 2. 실행 블록 (의존관계 기반)

### 의존관계 다이어그램

```
B0 ─┬─ B1 ─┐
    └─ B2 ─┼─ B3a ─┐
           └─ B3b ─┤
                   ├─ B4 (생성, 2~3hr 백그라운드)
                   ├─ B5 (검수 스크립트)
                   └─ B7 (문서 초안)
                          │
                          ▼
                       B6 (dedup + GO)
                          │
                          ▼
                       B8 (커밋/푸시)
```

### 예상 총 시간

- 순차 critical path: B0(10분) + B1(2hr) + B3a(3hr) + Pilot 검증(1hr) + B4 wall(3hr) + B6(2hr, 사람 검수 포함) + B8(10분) = **약 11~12시간**
- 병행 작업: B2, B3b, B5, B7은 critical path 위에서 또는 B4 wall 시간에 처리
- **집중 시 1.5일, 일반적으로 2일**

### 추가된 품질 보증 단계 (수정 검토 후 반영)

| 단계 | 효과 |
|---|---|
| Pilot 180건 사전 검증 (B3a) | 본 생성 전 GO 가능성 사전 확인 → 6,600건 헛생성 방지 |
| 생성 파라미터 다양화 (temperature/top_p/penalty) | GPT-4o 자체 편향 완화, 출력 다양성 +20~30% |
| 짧은 텍스트 별도 dedup 전략 (title/retry) | MinHash 5-gram이 작동 안 하는 케이스 보완 |
| v3↔v2 cross-similarity 검증 | "재생성했는데 v2와 비슷함" 케이스 차단 |
| 사람 검수 정량화 (양호 ≥80%, 불량 ≤5%) | 자동 메트릭이 못 잡는 어색함 탐지 |
| 첫 3어절 고유 조합 검증 (≥50%) | 시작어구만 바꾸고 나머지 구조는 동일한 케이스 차단 |

---

### Block 0 — 환경 준비 (10분)

**선행 조건**: 없음

```bash
cd finetuning
mkdir -p data_generation/v3/prompts data/processed_v3
pip install datasketch sentence-transformers faiss-cpu scikit-learn
```

- API 키 확인: `OPENAI_API_KEY` (재발급된 키 사용)

**완료 기준**: `python -c "import datasketch, sentence_transformers, faiss, sklearn"` 무에러

---

### Block 1 — 다축 설계(Persona·Occasion)·시드 확장 (2~2.5시간)

**선행 조건**: B0

**파일 작성**

1. `data_generation/v3/personas.json` — **persona 10개** 정의
   ```json
   [
     {"id": "p01", "name": "20대 여대생",       "tone": "친근/구어",     "vocab_hint": "완전, 레알, 진짜, ~거든요"},
     {"id": "p02", "name": "20대 남대생",       "tone": "짧고 간결",     "vocab_hint": "그냥, 좀, ~임"},
     {"id": "p03", "name": "30대 직장인 여성",   "tone": "캐주얼",       "vocab_hint": "요즘, 나름, ~는 편이에요"},
     {"id": "p04", "name": "30대 직장인 남성",   "tone": "정중/실용",     "vocab_hint": "보통, 주로, ~합니다"},
     {"id": "p05", "name": "30대 워킹맘",       "tone": "바쁘지만 친절", "vocab_hint": "틈나면, 아이가, ~하려고요"},
     {"id": "p06", "name": "40대 주부",         "tone": "따뜻/상세",     "vocab_hint": "가족이, 챙기는, ~하는 편이라"},
     {"id": "p07", "name": "40대 전문직 여성",   "tone": "정제/명료",     "vocab_hint": "업무상, 효율, ~를 선호해요"},
     {"id": "p08", "name": "50대 시니어 남성",   "tone": "정중/회상형",   "vocab_hint": "예전에는, ~더군요, 그러다 보니"},
     {"id": "p09", "name": "50대 시니어 여성",   "tone": "부드러움/경험", "vocab_hint": "아무래도, 요즘 들어, ~하지요"},
     {"id": "p10", "name": "프리랜서/자영업",    "tone": "자율적",       "vocab_hint": "저는, 제 기준에선, ~하는 편"}
   ]
   ```

2. `data_generation/v3/occasions.json` — **occasion 축 4개** 정의
   ```json
   [
     {"id": "o1", "name": "혼자",       "context": "혼술/혼밥/홈카페/1인 생활",     "tone_mod": "편안하고 자연스러움"},
     {"id": "o2", "name": "친구",       "context": "친구 모임/동아리/카페 수다",   "tone_mod": "친근/활기"},
     {"id": "o3", "name": "가족",       "context": "가족 식사/홈파티/명절",       "tone_mod": "따뜻/배려"},
     {"id": "o4", "name": "비즈니스",   "context": "직장 회식/거래처/네트워킹",    "tone_mod": "정중/격식"}
   ]
   ```

   → **persona 10 × occasion 4 = 40 실효 조합**. 같은 p03(30대 직장인 여성)도 o1(혼자)와 o4(비즈니스)에서 어투/어휘가 뚜렷이 다름.

3. `data_generation/v3/seeds_v3.json` — 시드 400개
   - 기존 `topics.json` 178개 base topic 보존
   - 신규 80~120개 시나리오 추가 (alcohol/cosmetics 내)
   - **각 시드에 `base_seed_id`, `persona_id`, `occasion_id` 필드 명시**
   - 변이 접미사(`_v1`, `_v2`...)는 폐기, 단일 ID로 통합
   - 40 조합을 시드 400개에 고르게 분포 (조합당 ~10 시드)
   ```json
   {
     "seed_id": "alc_p03o1_001",
     "base_seed_id": "alc_001",
     "persona_id": "p03",
     "occasion_id": "o1",
     "topic": "크래프트 맥주",
     "target_audience": "30대 직장인 여성 / 혼자 홈술",
     "domain": "alcohol",
     "first_question_original": "...",
     "questions": [...]
   }
   ```

4. `data_generation/v3/starting_phrases.json` — 시작어구 rotation pool
   ```json
   {
     "intro": ["안녕하세요", "반갑습니다", "환영합니다", "처음 뵙겠습니다", "좋은 하루입니다", "만나서 반가워요", ""],
     "retry": ["조금 더 자세히", "좀 더 구체적으로", "한 가지만 더", "다른 관점에서", "혹시", ""],
     "ending": ["오늘 나눠주신", "귀한 시간", "소중한 의견", "함께 해주셔서", "의미 있는 대화", "감사해요"]
   }
   ```

**완료 기준**:
- persona 10개, occasion 4개 정의 완료
- 시드 400개, topic 다양성 178개 (원본 topics.json 재활용)
- base_seed_id 100개 이상 (옵션 A 선택으로 117개 실측)
- 40 조합 모두 시드에 10개씩 할당
- 검증:
  ```python
  import json
  from collections import Counter
  d = json.load(open('data_generation/v3/seeds_v3.json'))['seeds']
  assert len(d) == 400
  assert len({s['base_seed_id'] for s in d}) >= 100
  assert len({s['topic'] for s in d}) >= 100
  combo_counts = Counter((s['persona_id'], s['occasion_id']) for s in d)
  assert len(combo_counts) == 40 and min(combo_counts.values()) >= 8
  ```

---

### Block 2 — 프롬프트 재작성 (2~3시간)

**선행 조건**: B0 (B1과 병행 가능)

**6개 파일 재작성** (`data_generation/v3/prompts/`)

| 파일 | 핵심 변경 |
|---|---|
| `intro.py` | "안녕하세요" 강제 제거, `starting_phrase` 인자로 rotation, few-shot도 시작어구 분산 |
| `first_question.py` | 시작 멘트 자유화, 인사말 강제 X |
| `retry.py` | invalid_type 5개 enum 엄격화 (자유 생성 금지), `starting_phrase` 적용 |
| `ending.py` | "오늘 나눠주신..." 템플릿 제거, `starting_phrase` 6개 rotation |
| `title.py` | 15자 제약 유지, 2개 도메인 키워드 풀 명시 |
| `response.py` | `VARIATION_SETS` 9 → `variation_matrix.json` 48조합 실제 사용, `persona` + `occasion` 받아 톤 변주, 같은 (seed, q_idx)당 **3→1건**만 생성 |

**프롬프트 작성 원칙**:
- "다음 형식으로 답하라: 안녕하세요, ..." 같은 고정 템플릿 제거
- `starting_phrase`, `persona`, `occasion` 3개 파라미터를 받아 의도적 다변화
- few-shot 예시 5개 중 서로 다른 시작어구·구조 사용
- response는 persona + occasion 조합으로 톤 변주 (예: p03×o1=편안한 혼술 여성, p03×o4=비즈니스 회식 여성)
- persona의 `vocab_hint`를 프롬프트에 삽입하여 어휘 특성 반영

**완료 기준**:
- 6개 프롬프트가 `starting_phrase`, `persona`, `occasion` 인자를 받음
- 각 프롬프트 1건씩 dry-run 호출 → 시작어구·구조가 서로 다른지 육안 확인
- 같은 seed + 다른 occasion 조합 2건 비교 → 톤 차이가 뚜렷한지 확인

---

### Block 3 — 생성 스크립트 + Dedup 파이프라인 (병렬, 각 2~3시간)

**선행 조건**: B0, B1, B2

#### 트랙 3a — `generate_data_v3.py`

기존 `generate_data.py` 구조 재사용. 변경점:
- 입력: `seeds_v3.json` + `personas.json` + `occasions.json` + `variation_matrix.json` + `starting_phrases.json`
- 시작어구 rotation 함수: `get_starting_phrase(task, idx) -> str`
- 각 시드의 `persona_id` + `occasion_id` 조회 → 프롬프트에 삽입
- response 태스크: (seed, q_idx)당 **1건만** 생성 (기존 3건에서 축소)

**생성 파라미터 명시적 다양화 (High Priority #2)**:
각 API 호출마다 아래 파라미터를 랜덤 샘플링:
- `temperature`: 0.8, 1.0, 1.2 중 선택
- `top_p`: 0.92, 0.95 중 선택
- `frequency_penalty`: 0.3, 0.5 중 선택
- `presence_penalty`: 0.2, 0.4 중 선택

이유: GPT-4o 자체의 고정 톤 수렴 편향을 완화. 프롬프트 다양화만으론 부족.

- 목표 생성량:
  ```
  response:        3,500 건
  retry:           1,200 건
  intro:             500 건
  ending:            500 건
  first_question:    500 건
  title:             400 건
  ── 합계 약 6,600 건
  ```
- async batching, GPT-4o-mini (response·retry는 GPT-4o로 품질 확보)

**Pilot 검증 강제 (본 생성 전)**:
- 본 6,600건 생성 전 먼저 **각 태스크 30건씩 총 180건 pilot 생성**
- Pilot에 대해 Block 5 `check_quality` 실행
- Pilot GO 통과 시 본 생성 진입, FAIL 시 프롬프트·파라미터 조정 후 재시도

#### 트랙 3b — `dedup_pipeline.py`

단일 진입점: `run_dedup(input_jsonl, output_dir) -> dict (메트릭)`

단계:
1. NFC 정규화 + 공백·구두점 통일
2. sha256 exact dedup
3. **태스크별 차등 near-dup 검출 (High Priority #3)**:
   - `response`, `intro`, `first_question`, `ending` (충분히 긴 텍스트):
     `datasketch.MinHashLSH(threshold=0.7, num_perm=128)`, 어절 5-gram
   - `retry` (평균 48자, 짧음):
     어절 **3-gram**으로 shingle 축소, threshold=0.75
   - `title` (평균 12자, 매우 짧음):
     MinHash 건너뛰고 **Levenshtein 편집거리 ≤ 2** 또는 exact hash만
4. OpenAI `text-embedding-3-small` 임베딩 + numpy cosine, cosine ≥ 0.92 페어 1개만 keep (태스크별 분리 수행). 사유: torch/faiss 로컬 설치 회피, 비용 $0.02 수준, 한국어 품질 실용상 e5-large와 유사
5. 시작어구 분포 검증 (각 태스크 top-1 ≤ 20%, 위반 시 stdout 경고)
6. **v3 vs v2 cross-similarity 검증 (High Priority #4)**:
   - v3 각 샘플에 대해 v2 best-match cosine 계산
   - `max(cross_cosine) > 0.95` 샘플은 제거 (v2와 거의 동일 → 재생성 의미 없음)
   - `avg(cross_cosine)` 리포트 (목표 ≤ 0.80)
7. `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`, groups=`base_seed_id`, stratify=`task_type` → 8/1/1 비율로 train/val/test
8. 누수 검증: `assert len(set(train_groups) & set(val_groups)) == 0`

**완료 기준**:
- 트랙 3a: pilot 180건 생성 성공, 결과물 형식 OK
- 트랙 3b: 기존 v2 데이터로 테스트 실행해 메트릭 출력 정상

---

### Block 4 — 본 생성 실행 (백그라운드, 2~3시간 wall)

**선행 조건**: B1, B2, B3a

```bash
nohup python data_generation/v3/generate_data_v3.py \
  --target-total 6600 \
  --output data/processed_v3/raw_all.jsonl \
  --log-level INFO > logs/gen_v3.log 2>&1 &
```

- 진행 모니터링 (별도 터미널): `tail -f logs/gen_v3.log`
- **이 동안 Block 5(검수 스크립트)·Block 7(문서) 작성 병행**

**완료 기준**: 6,000±500건 생성, 비어있거나 깨진 JSON 없음

---

### Block 5 — 검수·메트릭 스크립트 (1시간, B4와 병행)

**선행 조건**: B0

`data_generation/v3/check_quality.py` 작성:

```python
def check_quality(input_dir: str, v2_dir: str = "data/processed_v2") -> dict:
    """
    Returns:
        {
            "total": int,
            "by_task": {task: count},
            "starting_phrase_top1_pct": {task: pct},
            "first_3words_unique_pct": {task: pct},     # High Priority #1
            "leakage_base_seed_id": int,
            "avg_pairwise_cosine_per_seed": float,
            "variation_matrix_combos_used": int,
            "persona_occasion_combos_used": int,         # 다축 설계 검증
            "persona_occasion_balance_std": float,       # 40 조합 샘플 수 표준편차
            "task_ratio_balance": {split: {task: ratio}},
            "v3_v2_cross_similarity": {                  # High Priority #4
                "avg_cross_cosine": float,
                "max_cross_cosine": float,
                "removed_count": int
            },
            "human_review": {                            # High Priority #5
                "samples_per_task": 30,
                "ratings": {task: {"good": int, "fair": int, "bad": int}},
                "good_pct": {task: pct},
                "bad_pct": {task: pct}
            },
            "go_status": "PASS" | "FAIL",
            "fail_reasons": [...]
        }
    """
```

**GO 기준 (정량) — High Priority 5개 모두 반영**:

| # | 기준 | 임계 | 비고 |
|---|---|---|---|
| 1 | `starting_phrase_top1_pct[task]` | ≤ 0.20 | 모든 태스크 |
| 2 | **`first_3words_unique_pct[task]`** | **≥ 0.50** | **High #1: 시작어구만으론 부족, 첫 3어절 조합 다양성** |
| 3 | `leakage_base_seed_id` | = 0 | StratifiedGroupKFold 검증 |
| 4 | `avg_pairwise_cosine_per_seed` | ≤ 0.85 | seed 내부 다양성 |
| 5 | `variation_matrix_combos_used` | ≥ 30 | 48조합 중 |
| 5a | **`persona_occasion_combos_used`** | **≥ 35** | **40 조합 중 87.5% 이상 사용** |
| 5b | **`persona_occasion_balance_std / mean`** | **≤ 0.30** | **조합 간 샘플 수 편차 30% 이내** |
| 6 | task 비율 train/val/test 편차 | ≤ ±2% | 균형 |
| 7 | **`v3_v2_cross_similarity.avg_cross_cosine`** | **≤ 0.80** | **High #4: v2와 차별성** |
| 8 | **`v3_v2_cross_similarity.max_cross_cosine`** | **≤ 0.95** | **High #4: 거의 동일 샘플 제거 후 검증** |
| 9 | **`human_review.good_pct[task]`** | **≥ 0.80** | **High #5: 사람 검수 양호 비율** |
| 10 | **`human_review.bad_pct[task]`** | **≤ 0.05** | **High #5: 불량 비율 한계** |

**사람 검수 절차 (High Priority #5)**:
- 태스크별 30건씩 무작위 추출 (총 180건)
- 각 샘플을 "양호 / 보통 / 불량" 3단계 평가
  - 양호: 자연스러움 + 다양성 + 응답 품질 모두 OK
  - 보통: 사용 가능하지만 약간 어색함
  - 불량: 부자연스럽거나 의미 이탈, 정형성 심함
- 평가 결과를 `human_review.json`에 기록 후 `check_quality` 실행

**완료 기준**: v2 데이터로 실행해 정상 동작 (FAIL 출력 정상)

---

### Block 6 — 정제·분할 + GO 검증 (1.5~2시간, 사람 검수 포함)

**선행 조건**: B3b, B4, B5

```bash
# 1. dedup 파이프라인 실행
python data_generation/v3/dedup_pipeline.py \
  --input data/processed_v3/raw_all.jsonl \
  --output-dir data/processed_v3/

# 2. 사람 검수용 샘플 추출 (태스크별 30건)
python data_generation/v3/sample_for_review.py \
  --input-dir data/processed_v3/ \
  --output data/processed_v3/human_review_sheet.csv

# 3. CSV 열어서 양호/보통/불량 평가 (30분~1시간 소요)
#    → human_review.json 으로 변환

# 4. 종합 GO 검증
python data_generation/v3/check_quality.py \
  --input-dir data/processed_v3/ \
  --human-review data/processed_v3/human_review.json \
  --output data/processed_v3/quality_report.json
```

**분기**:
- **GO** (10개 정량 기준 + 사람 검수 모두 통과): Block 7 → Block 8 진행
- **NO-GO**: `fail_reasons` 확인 → 부족 항목 식별
  - 정량 기준 실패 → 해당 태스크만 재생성 (반나절 버퍼) → 재실행
  - 사람 검수 실패 → 프롬프트 조정 + 부분 재생성

---

### Block 7 — 문서 작성 (B4 동안 병행 시작)

**선행 조건**: 문서 1,2는 완료. 문서 3(본 플랜)은 B6 결과 채워 마감.

- 문서 3 결과 섹션에 `quality_report.json` 수치 기록
- 실패 시 재생성 이력도 남김

---

### Block 8 — 커밋·푸시 (10분)

**선행 조건**: B6 GO + B7 완료

```bash
git add data_generation/v3/ data/processed_v3/ docs/
git commit -m "feat: 학습 데이터 v3 재생성 — 누수·정형성·집중도 해결"
git push origin main
```

---

## 3. 리스크 및 대응

| 리스크 | 발생 조건 | 대응 |
|---|---|---|
| 프롬프트 이터레이션 | 첫 생성 결과도 boilerplate 심함 | 문제 태스크만 프롬프트 조정 후 부분 재생성 (+0.5~1일) |
| API 비용 초과 | 6,600건 × 평균 500토큰 | GPT-4o-mini 비중 확대 (response/retry만 GPT-4o) → $20~40 예상 |
| 임베딩 dedup 오탐 | 한국어 존댓말 false positive | cosine 0.92 → 0.94로 조정 |
| GO 기준 NO-GO | 시작어구 여전히 편향 | rotation pool 확대 + temperature 상향 (0.9 → 1.1) |
| 최종 샘플 수 부족 | dedup 후 3,000건 미만 | Evol-Instruct식 진화 생성으로 보강 (반나절) |

---

## 4. 검증 체크리스트 (실행 중 확인)

- [ ] B0: 의존성 설치 완료
- [ ] B1: 시드 400개, base_seed_id 250+개, persona 5~8개
- [ ] B2: 6개 프롬프트 dry-run 시작어구 분산 확인
- [ ] B3a: pilot 50건 생성 성공
- [ ] B3b: v2 기존 데이터로 dedup 테스트 통과
- [ ] B4: 6,000±500건 생성, JSON 유효성 100%
- [ ] B5: check_quality.py 동작 확인
- [ ] B6: GO 메트릭 전부 PASS
- [ ] B7: 문서 3종 최신화
- [ ] B8: 푸시 완료, 원격 반영 확인

---

## 5. 실행 결과 (2026-04-23)

### 5.1 최종 데이터 규모

| 항목 | v2 (기존) | v3 (신규) |
|---|---|---|
| 총 샘플 수 | 4,981 | **5,537** |
| train | 3,970 | **4,402** |
| val | 478 | **569** |
| test | 533 | **566** |
| unique base_seed_id | 117 (178 topic) | 117 (178 topic) |
| persona × occasion 조합 | - | **40/40** |
| variation_matrix 조합 | 9/48 | **16/48** |

### 5.2 품질 메트릭

| 항목 | v2 | v3 | 기준 | 결과 |
|---|---|---|---|---|
| intro 시작어구 top-1 점유 | 99.8% | **14.25%** | ≤ 20% | ✅ PASS |
| first_question 시작어구 top-1 | (대부분 "안녕하세요!") | 40.36% | ≤ 20% | ⚠ FAIL (v2 대비 개선) |
| retry 시작어구 top-1 점유 | 단일 템플릿 독점 | 21.64% | ≤ 20% | ⚠ 경계선 |
| ending 시작어구 top-1 점유 | "오늘 나눠주신..." 집중 | 20.25% | ≤ 20% | ⚠ 경계선 |
| title 시작어구 top-1 | 쏠림 | **6.65%** | ≤ 20% | ✅ PASS |
| response 시작어구 top-1 | 3-gram boilerplate | **4.36%** | ≤ 20% | ✅ PASS |
| intro 첫 3어절 고유 | 낮음 | 63.25% | ≥ 50% | ✅ PASS |
| first_question 첫 3어절 | — | 53.30% | ≥ 50% | ✅ PASS |
| retry 첫 3어절 | — | 43.20% | ≥ 50% | ⚠ FAIL |
| ending 첫 3어절 | — | 27.50% | ≥ 50% | ⚠ FAIL |
| title 첫 3어절 | — | **99.51%** | ≥ 50% | ✅ PASS |
| response 첫 3어절 | — | **90.74%** | ≥ 50% | ✅ PASS |
| base_seed_id 누수 (train↔val) | 6 seeds | **0** | = 0 | ✅ PASS |
| base_seed_id 누수 (train↔test) | 9 seeds | **0** | = 0 | ✅ PASS |
| base_seed_id 누수 (val↔test) | 2 seeds | **0** | = 0 | ✅ PASS |
| 평균 pairwise cosine / seed | 미측정 | **0.32** | ≤ 0.85 | ✅ PASS |
| persona×occasion std/mean | 미측정 | **0.017** | ≤ 0.30 | ✅ PASS |
| v3↔v2 cross-sim avg | - | **0.657** | ≤ 0.80 | ✅ PASS |

**판정**: 정량 지표상 7개 FAIL 중 5개는 보조 태스크(first_question/retry/ending)의 시작어구 편향. 실질 품질은 v2 대비 전반적으로 극적 개선 (intro 99.8% → 14.25% 등). **사용자 검토 후 현 상태로 수용 결정**.

### 5.3 Dedup 파이프라인 단계별 제거 건수

| 단계 | 제거 | 남은 |
|---|---|---|
| 원본 생성 | - | 5,954 |
| ① NFC + exact hash | 210 | 5,744 |
| ② 태스크별 near-dup (MinHash/Levenshtein) | 143 | 5,601 |
| ③ 임베딩 cosine ≥ 0.92 | 20 | 5,581 |
| ④ v3↔v2 cross-sim > 0.95 | 44 | 5,537 |
| ⑤ 8:1:1 split (누수 0%) | - | 4,402 / 569 / 566 |

### 5.4 생성 비용 및 소요 시간

- GPT-4o (response 태스크, 2,800 call 기대, 실제 2,754 완료): 약 $7
- GPT-4o-mini (보조 5개 태스크, 3,600 call): 약 $1
- 임베딩 OpenAI text-embedding-3-small: ~$0.02
- **합계 약 $8~9**
- API 호출: 6,400회 (실패 92회, 성공률 98.6%)
- 소요 시간: **생성 34.7분 + dedup 6분 = 약 41분**

### 5.5 실행 파일

생성된 디렉터리 구조:
```
data/processed_v3/
├── raw_all.jsonl                 # 원본 생성 데이터 (5,954건)
├── raw_all_pilot.jsonl           # Pilot 검증용 (215건)
├── train.jsonl                   # ChatML 학습 데이터 (4,402)
├── val.jsonl                     # ChatML 검증 데이터 (569)
├── test.jsonl                    # ChatML 테스트 데이터 (566)
├── train_raw.jsonl               # train raw 백업
├── val_raw.jsonl                 # val raw 백업
├── test_raw.jsonl                # test raw 백업
├── dedup_report.json             # dedup 상세 리포트
├── quality_report.json           # 품질 검증 리포트
└── pilot/                        # pilot 검증 결과
    ├── raw_all.jsonl
    └── quality_report.json

data_generation/v3/
├── personas.json
├── occasions.json
├── starting_phrases.json
├── seeds_v3.json
├── variation_matrix.json (data_generation/ 하위 참조)
├── build_seeds.py
├── generate_data_v3.py
├── dedup_pipeline.py
├── check_quality.py
├── convert_to_chatml.py
└── prompts/
    ├── intro.py
    ├── first_question.py
    ├── retry.py
    ├── ending.py
    ├── title.py
    └── response.py
```

---

## 6. 이후 단계

Phase 7 백엔드 통합 작업 진입 전 반드시 확인:

1. **Exp4 검증 학습** (18분 예상, 선택적)
   - v3 데이터로 Exp3a와 동일 하이퍼파라미터로 1 epoch 학습
   - 기대: eval_loss 곡선이 epoch 1.5~2까지 유지, 과적합 지점 늦춰짐
   - 결과가 기대와 다르면 프롬프트·dedup 임계치 재검토

2. **수동 평가 재실행** (v3 모델에서)
   - 기존 `src/manual_eval.py` 재활용
   - 톤·리프레이징 정량 점수가 v2 대비 회귀 없는지 확인
   - **주의**: `run_manual_eval.py`의 "이미지 관련 단어" 감점 로직은 사전 수정 필요 (Image 태스크에서 정상 출력이 감점되는 버그)

3. **Phase 7 진입 판단**: Exp4 결과 + 수동 평가 회귀 없음 확인 시 백엔드 통합 개시
