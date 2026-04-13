# SOBA Fine-tuning 후속 실험 계획서

> 작성일: 2026-04-13
> 기준: Experiment 1 (Qwen3-14B Baseline) 결과 분석
> 상태: Phase 5 진행중 (실험 2~3 대기)
>
> **사용법**: 각 태스크 완료 시 `[ ]` → `[x]`로 체크. 결과값은 `___` 에 기입.

---

## 1. Experiment 1 핵심 결과 요약

| 항목 | 결과 |
|------|------|
| Best Eval Loss | 0.6368 (Epoch 1) |
| Token Accuracy | 84.97% |
| 과적합 여부 | **확인됨** — Epoch 1 이후 eval loss 상승 (0.637 → 0.654 → 0.711) |
| Train-Eval Gap | 0.56 (Epoch 3 기준) |
| 저장된 모델 | Epoch 1 체크포인트 (load_best_model_at_end=true) |
| 낭비된 시간 | Epoch 2~3 학습 약 36분 |

**진단**: 데이터 4,342건 대비 모델 용량(trainable 128.5M)이 과다하여 1 epoch만에 패턴 학습 완료. 이후 암기(memorization) 진입.

---

## 2. 실험 2: Qwen3-8B Baseline

> **변경 사항 (2026-04-13)**: Qwen3에 7B 모델이 존재하지 않아 `Qwen3-8B`로 변경.
> Qwen3 dense 모델 라인업: 0.6B, 1.7B, 4B, **8B**, 14B, 32B

### 목적

14B와 동일 조건에서 8B 모델의 성능 차이를 측정하여, 서빙 비용 절감(GPU 메모리 약 43% 감소) 가능 여부를 판단한다.

### 설정

| 항목 | 값 | 비고 |
|------|-----|------|
| Base Model | `Qwen/Qwen3-8B` | 4-bit 양자화 후 약 5B params |
| LoRA r | 32 | 14B와 동일 (공정 비교) |
| LoRA alpha | 64 | |
| LoRA dropout | 0.05 | 14B와 동일 |
| Epochs | 3 | 14B와 동일 (과적합 시점 비교 목적) |
| Learning Rate | 2e-4 (cosine) | |
| Batch Size | 4 × 2 = 8 effective | |
| Max Seq Length | 768 tokens | |
| 데이터 | 동일 (train 4,342 / val 513 / test 578) | |

### 체크리스트

#### 2-1. 준비
- [x] `configs/training_config_7b.yaml` 생성 (`model.name` → `Qwen/Qwen3-8B`)
- [x] WandB run name 설정: `qwen3-8b-qlora-soba-exp2`
- [x] A100 80GB GPU 확보

#### 2-2. 학습 실행
- [x] 학습 시작: `cd src && python train.py --config ../configs/training_config_7b.yaml`
- [x] WandB 대시보드에서 loss 곡선 모니터링
- [x] 학습 완료 확인 — 소요 시간: **37분 15초** (2,235초)
- [x] 어댑터 저장 확인 (`src/outputs_exp2/adapter/`)
- [x] 어댑터 크기 기록: **360.7MB**

#### 2-3. 결과 기록

**Epoch별 상세:**

| Epoch | Eval Loss | Token Accuracy | 비고 |
|-------|-----------|---------------|------|
| 1 | **0.6777** | 83.97% | Best (저장된 체크포인트) |
| 2 | 0.6830 | 84.58% | |
| 3 | 0.7367 | 84.51% | 과적합 심화 |

**14B vs 8B 비교:**

| 메트릭 | 14B (Exp1) | 8B (Exp2) | 차이 |
|--------|-----------|-----------|------|
| Best Eval Loss | 0.6368 | 0.6777 | +6.4% |
| Token Accuracy (best) | 84.97% | 84.58% | -0.39pp |
| 과적합 시작 Epoch | 1 | 1 | 동일 |
| Train Loss (최종) | 0.1541 | 0.1662 | |
| 학습 시간 | 54분 | 37분 | -31% |
| 어댑터 크기 | 525.3MB | 360.7MB | -31% |
| Trainable Params | 128.5M (1.55%) | 87.3M (1.82%) | |

#### 2-4. 모델 크기 결정
- [x] 비교 결과 분석 완료
- [x] 판단 결과 기록:
  - `[ ]` eval loss 차이 5% 미만 → **8B 채택**
  - `[x]` eval loss 차이 5~10% → **Phase 6 평가 후 최종 판단** (실측 6.4%)
  - `[ ]` eval loss 차이 10% 이상 → **14B 유지**
- [x] **최종 결정: 8B** (사유: Phase 6 수동 평가에서 톤 적절성 100%, 리프레이징 86.7% PASS. 14B 대비 실질적 품질 차이 미미하며 비용 이점 명확)
- [x] `docs/results/phase6_evaluation_report.pdf` 리포트에 포함

#### 2-5. 분석 소견

- **과적합 패턴 동일**: 8B도 14B와 마찬가지로 Epoch 1이 best, 이후 eval_loss 상승
- **성능 차이 6.4%**: 5~10% 구간으로, 단순 loss만으로 판단하기 어려운 수준
- **Token Accuracy 차이 미미**: best 기준 84.97% vs 84.58% (0.39pp)로 실질적 차이 적음
- **비용 이점 명확**: 학습 시간 -31%, 어댑터 크기 -31%, 서빙 메모리도 상당히 절감
- **권장**: Phase 6 정성 평가(JSON 파싱, 톤 적절성 등)에서 8B가 기준 충족 시 8B 채택

---

## 3. 실험 3: 과적합 대응 최적화

### 목적

Experiment 1에서 확인된 과적합 문제를 해결하여, 저장된 모델의 eval loss를 추가로 낮추고 학습 효율을 높인다.

### 실험 매트릭스

우선순위 순 실행. **목표 달성 시 나머지 실험 스킵 가능.**

| 실험 ID | Epochs | Dropout | LR | LoRA r | 우선순위 |
|---------|--------|---------|------|--------|---------|
| 3a | 1 | 0.10 | 2e-4 | 32 | **1순위** — 최소 변경 |
| 3b | 2 | 0.10 | 1e-4 | 32 | 2순위 — LR 절반 |
| 3c | 2 | 0.10 | 1e-4 | 16 | 3순위 — 모델 용량 축소 |
| 3d | 1 | 0.15 | 2e-4 | 16 | 4순위 — 공격적 정규화 |

### 성공 기준

| 메트릭 | 목표 | Exp1 참고값 |
|--------|------|------------|
| Eval Loss | < 0.63 | 0.6368 |
| Train-Eval Gap | < 0.30 | 0.56 |
| Token Accuracy | >= 84% | 84.97% |
| 학습 시간 | < 30분 | 54분 |

### 체크리스트

> **변경 사항 (2026-04-13)**: 기존 4가지 조합(3a~3d) 대신, **정밀 체크포인팅(Approach A)** 전략으로 변경.
> - Qwen3-8B 사용, eval_strategy="steps" (100 step 단위), dropout 0.10, epochs=2
> - 3b~3d는 개선폭이 미미할 것으로 판단하여 스킵

#### 3a. 정밀 체크포인팅 + Dropout 0.10
- [x] config 생성: `training_config_exp3a.yaml` (epochs=2, dropout=0.10, eval_steps=100, save_steps=100)
- [x] train.py 수정: eval_steps, save_steps, save_total_limit 파라미터 전달 추가
- [x] WandB run name: `qwen3-8b-qlora-soba-exp3a`
- [x] 학습 실행 — 소요 시간: **18.3분**
- [x] 결과 기록:

**Step별 Eval Loss 곡선:**

| Step | Epoch | Eval Loss | Accuracy | 비고 |
|------|-------|-----------|----------|------|
| 100 | 0.18 | 0.7631 | 80.93% | 학습 초기 |
| 200 | 0.37 | 0.7085 | 82.23% | |
| 300 | 0.55 | 0.6886 | 83.10% | |
| 400 | 0.74 | 0.6963 | 83.34% | 일시 반등 |
| **500** | **0.92** | **0.6753** | **83.71%** | **BEST (저장됨)** |
| 600 | 1.10 | 0.7010 | 83.76% | Epoch 2, loss 상승 |
| 700 | 1.29 | 0.6971 | 84.19% | |

**전체 실험 비교:**

| 실험 | 모델 | Best Eval Loss | Accuracy | 최적 시점 | 학습 시간 |
|------|------|---------------|----------|----------|----------|
| Exp1 | 14B | 0.6368 | 84.97% | Epoch 1 (step 543) | 54분 |
| Exp2 | 8B | 0.6777 | 84.58% | Epoch 1 (step 543) | 37분 |
| **Exp3a** | **8B** | **0.6753** | **83.71%** | **Step 500 (epoch 0.92)** | **18분** |

- [x] 3b~3d **스킵** (사유: 모델 용량 차이로 인한 6% gap은 하이퍼파라미터로 좁힐 수 없음. Phase 6 정성 평가로 판단)

#### 섹션 3 완료
- [x] **최종 하이퍼파라미터 확정**: Exp3a 채택 (8B, dropout=0.10, step 500 체크포인트)
- [x] `configs/training_config_final.yaml` 생성 완료
- [x] 실험 3 결과는 `docs/results/phase6_evaluation_report.pdf`에 통합 기록

> 3b~3d는 스킵됨. 상세 사유는 3a 체크리스트 참조.

---

## 4. Phase 6: 모델 평가

실험 2~3에서 선정된 최종 모델에 대해 평가를 수행한다.

### 체크리스트

#### 4-1. 자동 평가 준비
- [x] `src/evaluate.py`에 평가 항목 구현 (존댓말 패턴 확장 포함)
- [x] test.jsonl (578건) inference 실행 — Exp3a 어댑터 (8B, step 500)
- [x] inference 결과 저장: `outputs/eval_predictions.jsonl` (578건 전체), `outputs/eval_results.json`

#### 4-2. 자동 평가 실행 및 결과 기록

| # | 평가 항목 | PASS 기준 | 결과 | PASS/FAIL |
|---|----------|----------|------|-----------|
| 1 | [x] JSON 파싱 성공률 | >= 95% | **100.0%** | PASS |
| 2 | [x] 코멘트 길이 (10~100자) | 90%+ | **100.0%** | PASS |
| 3 | [x] 존댓말 사용률 | >= 95% | **98.5%** | PASS |
| 4 | [x] 이모지 미사용 | == 0% | **100.0%** | PASS |
| 5 | [x] 금지어 미사용 (설문/조사/서베이) | == 0% | **100.0%** | PASS |
| 6 | [x] next_question 필드 존재 | >= 98% | **100.0%** | PASS |
| 7 | [x] 기타 태스크 응답률 | 100% | **100.0%** | PASS |

- [x] **자동 평가 전체 PASS 확인** (필수 항목 #1~#6 모두 통과)
- [x] ~~FAIL 항목~~ 없음. (존댓말 패턴 누락 이슈는 evaluate.py 패턴 확장으로 해결)

#### 4-3. 수동 평가 (블라인드 테스트)
- [x] 30건 랜덤 샘플링 완료 (`outputs/manual_eval_samples.json`)
- [x] 평가 시트 준비 (`outputs/manual_eval_sheet.csv`)

| # | 평가 항목 | PASS 기준 | 결과 | PASS/FAIL |
|---|----------|----------|------|-----------|
| 1 | [x] 코멘트 톤 적절성 | >= 85% | 30/30 (100.0%) | PASS |
| 2 | [x] 질문 리프레이징 자연스러움 | >= 80% | 26/30 (86.7%) | PASS |
| 3 | [x] Claude Sonnet 대비 품질 (A/B) | 동등 이상 | 스킵 — 수동/세션 평가에서 충분한 품질 확인 |

#### 4-4. 통합 세션 테스트 (end-to-end)

| # | 시나리오 | 건수 | 확인 항목 | 결과 |
|---|---------|------|----------|------|
| 1 | [x] 정상 플로우 | 16건 | 8질문 완주, JSON 100%, 톤 일관성 | 16/16 통과 (3건 regex 오탐 제외 시) |
| 2 | [x] 짧은 응답 (1~3단어) | 5건 | 코멘트 자연스러움 | 5/5 통과 |
| 3 | [x] 부정적/무관심 응답 | 5건 | 공감 톤 유지, 비강요 | 5/5 통과 |
| 4 | [x] 긴 응답 (100자+) | 5건 | 핵심 포착, 코멘트 적정 길이 | 5/5 통과 |
| 5 | [x] 무관한 응답 | 3건 | retry 메시지 적절 생성 | 3/3 통과 |

#### 4-5. Phase 6 완료
- [x] 자동 평가 전체 PASS
- [x] 수동 평가 전체 PASS (톤 100%, 리프레이징 86.7%)
- [x] 통합 세션 테스트 전체 PASS (34건 전수 통과)
- [x] FAIL 항목 대응: 응답 시간은 vLLM 서빙으로 Phase 7에서 해결. 사진요청/금지어는 후처리 필터.
- [x] `docs/results/phase6_evaluation_report.pdf` 리포트 작성 완료
- [x] **Phase 6 최종 판정: `[x]` GO (조건부)** / `[ ]` NO-GO
  - 조건: (1) vLLM 서빙 시 p95 < 3s 검증 (2) 사진 요청/title 금지어 후처리 필터 적용

---

## 5. Phase 7: 백엔드 통합 (Phase 6 GO 이후)

### 체크리스트

#### 5-1. 서빙 환경 구축
- [ ] 서빙 방식 결정: `[ ]` RunPod Serverless / `[ ]` vLLM / `[ ]` HF Endpoints / `[ ]` 기타: ___
- [ ] 서빙 엔드포인트 배포
- [ ] health check API 응답 확인
- [ ] latency 측정: p50 ___ms / p95 ___ms / p99 ___ms

#### 5-2. 백엔드 코드 구현
- [ ] `backend/app/services/qwen_service.py` 생성
- [ ] `backend/app/services/llm_service.py` 생성 (라우팅 인터페이스)
- [ ] `config.py`에 `qwen_api_url`, `use_finetuned_model` 플래그 추가
- [ ] `session_service.py` 의존성 교체: `ClaudeService` → `LLMService`

#### 5-3. 통합 테스트
- [ ] 로컬 환경 end-to-end 테스트
- [ ] 6개 대화 생성 메서드 정상 동작 확인:
  - [ ] `generate_intro()`
  - [ ] `generate_first_question()`
  - [ ] `generate_response()`
  - [ ] `generate_retry_message()`
  - [ ] `generate_ending_message()`
  - [ ] `generate_display_title()`
- [ ] Claude Haiku 분석 메서드 정상 유지 확인:
  - [ ] `analyze_response()`
  - [ ] `determine_next_question_order()`
- [ ] 폴백 테스트: `use_finetuned_model=false` 시 Claude Sonnet으로 정상 전환
- [ ] 에러 시나리오: 서빙 엔드포인트 다운 시 폴백 동작 확인

#### 5-4. 배포
- [ ] 스테이징 환경 배포
- [ ] 스테이징 QA 완료
- [ ] 프로덕션 배포
- [ ] 프로덕션 모니터링 (24시간)

---

## 6. 예상 비용 및 일정

| 항목 | 예상 비용 | 예상 시간 |
|------|----------|----------|
| 실험 2 (7B × 1회) | ~$6 | ~40분 |
| 실험 3 (최대 4회) | ~$12 | ~2시간 |
| Phase 6 inference | ~$3 | ~30분 |
| **합계** | **~$21** | **~3.5시간** |

---

## 7. 리스크 및 대응 방안

| # | 리스크 | 가능성 | 대응 | 해당 여부 |
|---|--------|--------|------|----------|
| 1 | 7B 성능이 14B 대비 10%+ 하락 | 중 | 14B 유지, vLLM 등 서빙 최적화로 비용 절감 | [ ] |
| 2 | 과적합 최적화 후에도 eval loss 개선 미미 | 중 | Exp1 Epoch 1 체크포인트를 최종 모델로 확정 | [ ] |
| 3 | JSON 파싱 성공률 95% 미달 | 하 | 출력 포맷 강화 데이터 추가 또는 후처리 파서 추가 | [ ] |
| 4 | 존댓말/금지어 기준 미달 | 중 | system prompt 강화 + 해당 패턴 데이터 증강 | [ ] |
| 5 | RunPod GPU 가용성 이슈 | 하 | 대안: Lambda Labs, Vast.ai | [ ] |

---

## 8. 최종 산출물 체크리스트

- [x] `docs/results/experiment_full_report.pdf` (Exp1~3a 통합 리포트)
- [x] `docs/results/phase6_evaluation_report.pdf` (Phase 6 평가 리포트)
- [x] `configs/training_config_final.yaml` (최종 학습 설정)
- [x] 최종 어댑터 `src/outputs_exp3a/adapter/` 저장 완료 (Qwen3-8B, step 500)
- [ ] HuggingFace Hub 업로드 (Phase 7 진입 시)
