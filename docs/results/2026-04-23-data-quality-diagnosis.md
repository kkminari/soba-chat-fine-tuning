# 학습 데이터 품질 진단 보고서

**작성일**: 2026-04-23
**대상**: SOBA 챗봇 파인튜닝 데이터 (`data/processed_v2/`)
**배경**: Exp1~3a 모두 epoch 1 이후 과적합 발생, dropout 조정으로도 해결되지 않음 → 모델 용량 문제가 아닌 데이터 자체의 문제일 가능성 높음

---

## 1. 핵심 결론

학습 데이터는 절대 규모(3,970건)는 작지 않으나, 다음 세 가지 특성이 결합되어 **"1 epoch 최적 + 이후 과적합"** 현상을 필연적으로 만들고 있습니다.

1. **보조 태스크의 극도 정형성** — 시작어구 1~2개가 절반 이상 점유하여 모델이 1 epoch 안에 완전 암기
2. **유효 다양성 부족** — 시드당 19샘플 집중, 같은 (seed, q_idx)에서 파생된 변형이 의미·어순 거의 동일
3. **Split 누수 (실측)** — base topic 기준 val 26%, test 32%가 train과 중복되어 eval_loss가 실제보다 낮게 보이는 착시

→ 결과적으로 모델은 1 epoch 내에 학습 가능한 정보를 모두 흡수하고, 이후 epoch에서는 암기(memorization)만 진행되어 train_loss는 떨어지지만 eval_loss는 상승. **모델 용량 과다보다 데이터의 유효 정보량이 1 epoch 분량인 것이 정확한 진단**.

---

## 2. 데이터 규모 및 분포

### 2.1 태스크별 샘플 수 (raw → processed → processed_v2)

| 태스크 | raw (merged) | v1 train/val/test | v2 train/val/test | v2 total |
|---|---|---|---|---|
| response | 2,430 | 1,902 / 216 / 259 | 1,584 / 184 / 218 | 1,986 |
| retry | 1,241 | 920 / 115 / 122 | 866 / 112 / 118 | 1,096 |
| intro | 498 | 394 / 46 / 51 | 394 / 46 / 51 | 491 |
| ending | 497 | 398 / 48 / 51 | 398 / 48 / 51 | 497 |
| first_question | 484 | 387 / 47 / 50 | 387 / 47 / 50 | 484 |
| title | 464 | 341 / 41 / 45 | 341 / 41 / 45 | 427 |
| **합계** | 5,614 | **4,342 / 513 / 578** | **3,970 / 478 / 533** | **4,981** |

- v1 → v2 정제로 452건(8.3%) 감소. 주로 response(-418)와 retry(-59).
- 핵심 태스크(response)가 전체의 40%, retry가 22% — retry 비중이 매우 높은데 이 태스크가 가장 정형화되어 있어 과적합 기여도가 큼.

### 2.2 시드·주제 다양성

- `topics.json`에 **249개 seed_id**, 그러나 **base topic 문자열은 178개**.
- 시드 변이 존재 (예: `alc_005`, `alc_039`, `alc_039_v1`, `alc_005_v1_v2_v3_v4`, `alc_005_v1_v2_v3_v4_v5_v6` 모두 "크래프트 맥주" 주제). 최대 5개 seed가 하나의 base topic 공유.
- 도메인은 **alcohol 144건, cosmetics 105건 2개뿐** → 도메인 일반화 신호 약함.

### 2.3 샘플 길이 분포 (chars, v2 train)

| 태스크 | min | p50 | p75 | max | mean |
|---|---|---|---|---|---|
| response (JSON) | 65 | 91 | 102 | 149 | 93.2 |
| intro | 61 | 85 | 92 | 124 | 86.0 |
| ending | 64 | 90 | 97 | 130 | 90.3 |
| first_question | 23 | 63 | 73 | 127 | 64.6 |
| retry | 20 | 48 | 55 | 80 | 47.2 |
| title | 6 | 12 | 13 | 15 | 11.5 |

- 응답이 매우 짧음(전체 평균 ~30토큰). 짧은 응답은 학습 신호가 약하고 loss가 빠르게 수렴하여 과적합 유발.

---

## 3. 다양성 / 중복성 분석

### 3.1 정확 중복률 (assistant 텍스트 기준)

| 태스크 | N | unique | exact-dup % |
|---|---|---|---|
| response (comment+next_q) | 2,430 | 2,430 | 0.0% |
| intro | 498 | 496 | 0.4% |
| first_question | 484 | 484 | 0.0% |
| ending | 497 | 497 | 0.0% |
| retry | 1,241 | 1,191 | **4.0%** |
| title | 464 | 438 | **5.6%** |

정확 일치 중복만 보면 건강해 보이지만, **진짜 문제는 시작어구·n-gram의 집중**.

### 3.2 시작 5어절(starting phrase) 쏠림 — 과적합 주범

**retry (1,241건 중 상위 5개 시작어구가 169건, 13.6% 점유)**
```
[68x] 조금 더 구체적으로 말씀해 주시면
[33x] 조금 더 자세히 말씀해 주시면
[31x] 좀 더 구체적으로 말씀해 주시면
[21x] 조금 더 자세한 답변을 주시면
[16x] 좀 더 자세한 의견을 주시면
```
사실상 **1가지 문장 틀**을 1,100개 학습시키는 것과 같음.

**ending (497건 중 상위 10개 시작어구가 144건, 29% 점유)**
```
[28x] 오늘 나눠주신 소중한 이야기 정말
[21x] 오늘 나눠주신 이야기 정말 감사드립니다!
[20x] 오늘 나눈 이야기 정말 감사드립니다!
[18x] 오늘 나눠주신 이야기에 정말 감사드립니다!
```
첫 문장 기준으로 보면 497건 중 고유 첫 문장 266개(53.5%) — 절반이 반복 구조.

**intro (498건 중 491건이 "안녕하세요"로 시작, 99.8%)**
즉 intro 태스크의 첫 어절은 시스템 프롬프트만 봐도 100% 예측 가능. 모델은 1 epoch에 완전 습득.

**response.comment 상위 3-gram (8,640회 등장 / 8,109고유)**
```
[19x] 그럴 수 있죠.
[16x] 수 있죠. 이해합니다.
[14x] 다를 수 있죠.
[14x] 그럴 수도 있죠.
```
프롬프트에 "짧고 간결, 한 문장당 30자 이내"가 강제되어 있어 GPT-4o가 자연스럽게 같은 템플릿으로 수렴.

### 3.3 title 태스크 — 사실상 반복

```
[3x] 40대 강사 메이크업 조사
[2x] Z세대 술 문화 조사
[2x] 워킹맘 음주 패턴
[2x] 프리랜서 음주 패턴
```
15자 이내 제약 + 2개 도메인 → 실질적 어휘 공간 협소.

### 3.4 response 샘플의 유사 변이 3개 묶음

response는 "1회 API 호출 = 3가지 변형 배치"로 생성됨. 같은 (seed_id, q_idx)에서 파생된 3개 샘플이 다음처럼 거의 같은 구조:

예시 (seed=alc_003, q_idx=4, 질문="저렴한 맥주를 선택할 때와 비싼 맥주를 선택할 때 각각 어떤 기분이 드시나요?"):
```
var1 next_q: 편의점에서 맥주 고르실 때 가격표는 어떻게 확인하세요? 용량 대비 가격도 따져보세요?
var2 next_q: 편의점에서 맥주 살 때, 가격표는 어떻게 체크하세요? 용량 대비 가격도 보시나요?
var3 next_q: 맥주 고르실 때, 편의점에서 가격표는 어떻게 보세요? 용량 대비 가격도 계산해보시나요?
```
정확 일치는 아니지만 의미·어순·구문이 거의 동일 — **모델이 하나를 배우면 나머지 둘도 이미 맞춤**.

(seed, q_idx) 쌍 중 569쌍이 3개 샘플 배치, 733쌍은 1회만 등장.

### 3.5 시드별 샘플 집중도

- v2 train의 유효 시드 수: **209개**, 평균 19.0 samples/seed.
- 한 시드에서 같은 질문 내러티브로 최대 ~20건이 파생 → **유효 독립 샘플 수는 209 × (1~2) = ~400 수준으로 추정**. 겉보기 N=3,970이지만 정보량은 그 1/10.

---

## 4. 데이터 생성 방식의 근본 문제

### 4.1 보조 태스크 프롬프트 — "단일 스타일 강제"

| 태스크 | 프롬프트 출력 요구 | 결과 |
|---|---|---|
| intro | "구조: 따뜻한 인사 → 주제 소개 → 소요시간 → 준비 확인" | 498건 전부 "안녕하세요"로 시작 |
| ending | "감사 + 나눠준 이야기의 가치 + 마무리" | 상위 10 시작어구가 29% 점유 |
| retry | "비난하지 않고 부드럽게, 1~2문장" + 5가지 invalid_type | "조금 더 구체적으로…" 68회 등 템플릿 고착 |
| first_question | "짧은 시작 멘트 + 첫 질문 리프레이징" | 시작 대부분 "안녕하세요!" (clean_data가 제거하지만 학습은 완료) |
| title | "15자 이내, 2가지 스타일" | 어휘 공간 극소 |
| response | "존댓말, 이모지 금지, 한 문장당 30자 이내" + VARIATION_SETS 9조합만 순환 | 3-gram boilerplate 집중 |

**핵심 문제**: 프롬프트가 다양성을 유도하기보다 "스타일 제약을 강하게 거는" 방향. GPT-4o는 제약 안에서 최단 거리의 안전한 표현으로 수렴.

### 4.2 variation_matrix.json vs 실제 사용

- `variation_matrix.json`: response_styles(4) × conversation_stages(3) × user_sentiments(4) = **48조합** 정의
- 실제 `prompts/response.py`의 `VARIATION_SETS`: **3세트 × 3변형 = 9조합만** 순환 사용
- **variation_matrix는 사실상 사용되지 않음** → 설계 의도의 1/5 수준 다양성만 발현

### 4.3 seed 변이 augmentation의 역효과

- seed_id 접미사 변이(`_v1`, `_v1_v2_v3`)로 같은 base topic에서 최대 5개 변이 시드.
- 변이 시드들은 질문 셋이 대부분 겹침 → **train/val/test를 seed_id 기준으로 나눠도 base topic + 유사 질문이 split 간 전파**.

---

## 5. 데이터 분할 누수 (핵심 문제)

### 5.1 seed_id 누수

v2 기준 실제 분할:
- train: 209 unique seeds (3,970 samples)
- val: 24 unique seeds (478 samples)
- test: 32 unique seeds (533 samples)

**누수된 seed_id (train ∩ val 6개, train ∩ test 9개, val ∩ test 2개)**:
```
train ∩ val:
  alc_031_v1_v2_v3_v4_v5_v6_v7_v8  train=8 val=20
  alc_047_v1_v2_v3_v4_v5_v6        train=6 val=21
  cos_002_v1_v2_v3                 train=8 val=23
  cos_050_v1_v2_v3_v4              train=8 val=16
  cos_052_v1                       train=4 val=18
  cos_053_v1_v2_v3_v4              train=6 val=12
```
→ v2 train의 135개 샘플, val의 110개 샘플(23% of val), test의 106개 샘플(20% of test)이 반대 쪽 split에 동일 seed 기반 친척 샘플을 가짐.

태스크별 분포를 보면 패턴이 더 선명:
```
seed=cos_002_v1_v2_v3
  ending         train=2  val=2   test=0
  first_question train=2  val=2   test=0
  intro          train=2  val=2   test=0
  response       train=0  val=11  test=0   ← response는 완전 val
  retry          train=0  val=5   test=0
  title          train=2  val=1   test=0
```
→ **분할이 per-seed가 아닌 per-(seed, task_type)로 된 것처럼 보임**. response는 잘 분리되었으나 intro/ending/first_question/title은 같은 시드가 양쪽에 들어감.

### 5.2 Base topic 누수 — 실제로는 더 심각

seed 변이 때문에 base topic 기준으로는 훨씬 큰 누수:

| Split pair | 공유 base topic 수 | val/test 기준 비율 |
|---|---|---|
| train ∩ val | 6 | **26.1% of val topics (6/23)** |
| train ∩ test | 8 | **32.0% of test topics (8/25)** |
| val ∩ test | 2 | — |

공유 topic 예시: '리더십 이미지 메이크업', '시니어 사회적 음주', '안티에이징 제품', '여성 1인 가구 안전 홈술', '전문직 여성용 화장품', '여대생 안전 음주 문화'.

→ val/test의 1/4~1/3은 "모델이 train에서 같은 주제의 다른 시드 변이를 보고 학습한 문제". **이것이 eval_loss가 1 epoch 만에 0.64~0.68까지 빠르게 떨어지는 이유** — 진짜로 새로운 평가가 아니라 부분적으로 암기된 평가에 가까움.

### 5.3 near-duplicate 샘플

v2 val 200건 무작위 추출 → 6건이 train 내 0.9+ 유사 샘플 보유 (3%). 예:
```
[NEAR] val:  안녕하세요! 오늘은 고급 화장품에 대해 이야기 나눠볼까 해요. 가벼운 대화로 5~10분이면 충분해요. 준비되셨나요?
       tr :  안녕하세요! 오늘은 크래프트 맥주에 대해 이야기 나눠볼까 해요. 가벼운 대화로 5~10분이면 충분해요. 준비되셨나요?

[NEAR] val:  조금 더 자세한 답변을 주시면 감사하겠습니다. 어떤 화장품을 사용하시는지 궁금합니다.
       tr :  조금만 더 자세하게 답변해 주시면 감사하겠습니다. 어떤 화장품을 사용하는지 궁금합니다.
```
user-text까지 동일(exact match)하는 경우도 train ∩ val에서 10건, train ∩ test에서 9건.

---

## 6. 가설 우선순위 및 과적합 메커니즘

### 가설 1 (High) — 보조 태스크의 정형성이 유효 학습량을 급감시킴

| 근거 |
|---|
| retry 시작어구 상위 5개가 169건(13.6%) 점유 |
| intro 첫 어절 99.8%가 "안녕하세요" |
| ending 상위 10개 시작구가 29% 점유 |
| title은 15자 이내, 고유 어휘 공간 협소 |

**메커니즘**: 보조 태스크 4개(intro/ending/retry/first_question)는 total 1,645 train samples인데 실제로 학습할 "규칙"은 각 태스크당 1~2개 템플릿뿐. QLoRA 128M trainable 파라미터가 수 분 내로 완전 학습 → epoch 1 안에 이미 포화. epoch 2부터는 내용 변이를 암기하려 들지만 val 쪽에도 같은 템플릿이 있어 eval_loss는 그대로 유지되다 결국 memorization overhead로 상승.

### 가설 2 (High) — Split 누수로 eval_loss가 실제보다 낮게 보이고, 누수 없는 부분이 과적합 신호를 먼저 드러냄

| 근거 |
|---|
| v2 val의 26.1%, test의 32.0%가 train과 base topic 공유 |
| seed_id 자체도 train↔val 6개, train↔test 9개 중복 |
| 특히 intro/ending/first_question/title은 seed 간 분리 안 됨 (per-task 분할) |
| val 200샘플 중 6건이 train 근사중복 보유 |

**메커니즘**: 1 epoch이 끝나면 누수 없는 순수 val(~74%)과 누수된 val(~26%)가 혼합된 eval_loss = 0.64는 "낮게 나오는 착시". epoch 2 이후 모델은 누수 없는 부분을 일반화하지 못해 그 부분의 loss가 튀어오르고, 동시에 train 쪽은 암기해서 train_loss만 떨어짐 — 정확히 Exp1/Exp2/Exp3에서 관찰된 곡선.

### 가설 3 (Med-High) — 시드당 샘플 집중으로 유효 독립 샘플 수가 1/3~1/2

| 근거 |
|---|
| 3,970 train / 209 unique seeds = 19 samples/seed 집중 |
| (seed_id, question_index) 쌍 중 569쌍이 3개 샘플 배치 |
| 같은 (seed, q_idx) 3샘플이 의미·어순 거의 동일 |
| 2개 도메인(alcohol 144 + cosmetics 105)만 |

**메커니즘**: SFT의 관점에서 "샘플 수"보다 "샘플 간 독립성"이 loss 수렴 속도를 결정. 실효 N이 ~400~1,000 수준이면 trainable 128M 파라미터 모델은 몇백 스텝이면 전부 암기 가능 → 1 epoch(~543 step) 지점에서 이미 포화.

### 가설 4 (Med) — 응답이 너무 짧고 JSON 포맷이 너무 정형

| 근거 |
|---|
| title p50=12자, retry p50=48자, response p50=91자 |
| response는 100% 유효 JSON 스키마 |
| 평균 assistant 토큰 수 ~30 |

**메커니즘**: 짧은 응답은 token-level loss 기여가 적고, JSON 구조·공감어구·질문 어미 같은 고빈도 토큰이 loss를 지배 → 몇 배치 안에 loss가 0.7대로 수렴. 실제 학습되어야 할 "공감 내용의 의미"는 loss 기여도가 낮아 학습이 덜 됨.

### 가설 5 (Med) — VARIATION_SETS/variation_matrix 불일치로 인한 실질 다양성 축소

| 근거 |
|---|
| variation_matrix.json에 48조합 정의 |
| 실제 response.py는 9조합만 순환 사용 |
| intro/ending 2톤만, title 2스타일만 |

---

## 7. "1 epoch 최적" 현상과 데이터의 연결

관찰된 사실 (Exp1: Qwen3-14B, Exp2: 8B, Exp3a: 8B+dropout 0.10):
- Epoch 1 eval_loss 0.637~0.678, epoch 2부터 상승
- Step 500 (epoch 0.92)에서 Exp3a의 eval_loss 최저 0.6753
- dropout을 0.10으로 올려도 과적합 지점은 거의 같은 시점에서 발생

**이 패턴은 모델 용량 문제보다 데이터 특성으로 더 잘 설명됨**:

1. 보조 태스크(retry/intro/ending/first_question/title) ≈ 1,645 train samples × 평균 5~10 토큰 학습 신호 → 100~200 step 내 포화
2. response 2,430 × 40 tokens = 97,200 token 학습 신호이나 (seed, q_idx) 당 3개 유사 샘플이라 실효 정보량은 ~1/3
3. **효과적 학습 가능 신호는 대략 400~500 step 분량**. epoch 1 (543 step)에서 그 분량을 이미 다 본 셈 → 정확히 관찰된 곡선

dropout을 올려도 곡선이 거의 같은 이유: dropout은 과적합 속도를 늦출 수는 있어도, **학습 데이터 자체에 내재한 반복 구조를 해결하지 못함**. 같은 템플릿이 1,000번 반복되면 dropout이 있어도 학습됨.

---

## 8. 추가 발견 사항

- **retry invalid_type 라벨이 32종으로 분절됨** (`too_short`, `only_consonants`, `consonants_only`, `just_consonants`, `consonant_vowel_only` 등 동일 의미 중복). GPT-4o가 프롬프트의 5가지 enum을 무시하고 자유 생성한 결과.
- **validator 미흡**: `convert_to_chatml.py`는 assistant 내용 empty 체크만, 중복/유사도 체크는 없음.
- **chatml.jsonl에는 seed_id가 있으나 train/val/test로 분할될 때 제거됨** — 추후 분할 재검증이 어려운 상태.

---

## 9. 권장 조치 방향 (요약)

데이터 자체의 문제 해결 우선순위:

1. **Split 재분할**: base topic 단위로(seed 변이 합쳐서) train/val/test 재구성. 같은 base topic은 반드시 한 split으로.
2. **보조 태스크 정형 완화**: intro 첫 어절 강제 제거, retry 템플릿 다변화 (감탄사/장소 고정 언급/반응형 문장 등 3~5종), ending 3문장 구조 해제.
3. **Response 중복 축소**: (seed, q_idx) 당 3개 배치 대신 1~2개로 축소하고, 남은 샘플 수만큼 시드 수 확대.
4. **variation_matrix 실제 사용**: response.py의 `VARIATION_SETS` 9조합 → 48조합 전체 랜덤 샘플링으로 재생성.
5. **Near-duplicate filter 추가**: `clean_data.py`에 MinHash 또는 cosine 유사도 0.9+ 쌍 제거 단계 추가.

→ 구체적 실행 계획은 `docs/plans/2026-04-23-data-regeneration-plan.md` 참조.
→ 중복 제거 방법론 가이드는 `docs/plans/2026-04-23-deduplication-methodology.md` 참조.
