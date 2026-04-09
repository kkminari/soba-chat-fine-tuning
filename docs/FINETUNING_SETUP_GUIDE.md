# QLoRA 파인튜닝 세팅 가이드

> 이 문서는 `senti-fine-tuning-test` 프로젝트의 파인튜닝 파이프라인을 정리한 것입니다.
> 새 프로젝트에서 이 문서를 참고하여 동일한 구조로 파인튜닝을 세팅할 수 있습니다.

---

## 1. 프로젝트 구조

```
project-root/
├── configs/
│   └── training_config.yaml    # 모든 하이퍼파라미터 중앙 관리
├── src/
│   ├── data_loader.py          # 데이터 로딩, 전처리, 프롬프트 템플릿
│   ├── train.py                # QLoRA 학습 메인 스크립트
│   ├── evaluate.py             # Base vs Fine-tuned 비교 평가
│   ├── inference.py            # 단일 텍스트 추론
│   └── cross_validate.py       # K-Fold 교차 검증 (소규모 데이터용)응
├── notebooks/
│   └── eda.ipynb               # 데이터 탐색
├── reports/                    # 실험 보고서
├── .env                        # API 키 (git 미포함)
├── .gitignore
└── requirements.txt
```

---

## 2. 환경 설정

### 2-1. 필수 패키지 (requirements.txt)

```
torch>=2.1.0
transformers>=4.44.0
peft>=0.12.0
trl>=0.9.0
bitsandbytes>=0.43.0
datasets>=2.20.0
accelerate>=0.33.0
wandb>=0.17.0
scikit-learn>=1.5.0
sentencepiece>=0.2.0
protobuf>=5.27.0
python-dotenv>=1.0.0
pandas>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### 2-2. API 키 (.env)

```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

### 2-3. .gitignore

```
__pycache__/
*.pyc
.env
wandb/
outputs/
checkpoints/
*.pt
*.bin
*.safetensors
.DS_Store
.ipynb_checkpoints/
```

---

## 3. 설정 파일 (training_config.yaml)

아래는 전체 설정 템플릿입니다. 새 프로젝트에서 복사 후 `[변경]` 표시된 항목만 수정하면 됩니다.

```yaml
# ============================================================
# 모델 설정
# ============================================================
model:
  name: "Qwen/Qwen3-14B"            # [변경] 사용할 베이스 모델
  max_seq_length: 512                # [변경] 프롬프트+응답 최대 토큰 수
                                     #   최초 512로 시작 → 토큰 통계 확인 후 조정
                                     #   줄이면 패딩 감소 → batch_size 증가 가능

# ============================================================
# 양자화 (Quantization) 설정
# ============================================================
# 14B 모델 기준: FP16 ~28GB → 4bit ~8GB로 축소
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"         # NormalFloat4 (FP4보다 정확)
  bnb_4bit_compute_dtype: "bfloat16" # 연산은 bfloat16 (4bit는 저장용)
  bnb_4bit_use_double_quant: true    # 양자화 상수 재양자화 → 추가 ~0.4GB 절약

# ============================================================
# LoRA 설정
# ============================================================
# 원리: W_new = W_frozen + (alpha/r) * A * B
lora:
  r: 16                  # [변경] rank — 데이터 규모에 따라 조정
                         #   소규모(<500건): 8~16
                         #   중규모(500~5000건): 16~32
                         #   대규모(5000건+): 32~64
  lora_alpha: 32         # 보통 r의 2배 (스케일 = alpha/r)
  lora_dropout: 0.1      # [변경] 과적합 방지
                         #   데이터 적으면 0.1~0.15, 많으면 0.05
  target_modules:        # [변경] 모델 아키텍처에 따라 다름
    - q_proj             # Attention Query
    - k_proj             # Attention Key
    - v_proj             # Attention Value
    - o_proj             # Attention Output
    - gate_proj          # MLP Gate (분류 태스크에 중요)
    - up_proj            # MLP Up
    - down_proj          # MLP Down
  bias: "none"
  task_type: "CAUSAL_LM"

# ============================================================
# 학습 하이퍼파라미터
# ============================================================
training:
  num_train_epochs: 4              # [변경] 소규모: 3~5, 대규모: 1~3
  per_device_train_batch_size: 4   # [변경] GPU 메모리에 따라 조정
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 2   # 실효 배치 = batch_size × 이 값
  learning_rate: 2.0e-4            # QLoRA 표준 (Full FT는 1e-5~5e-5)
  lr_scheduler_type: "cosine"      # 후반부 안정적 수렴
  warmup_ratio: 0.05               # 전체 스텝의 5% 워밍업
  optim: "paged_adamw_8bit"        # 메모리 효율적 옵티마이저
  fp16: false
  bf16: true                       # [변경] A100/H100: true, V100: false (fp16: true)
  logging_steps: 5
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  report_to: "wandb"

# ============================================================
# 데이터 설정
# ============================================================
data:
  dataset_name: "Younggooo/senti_data2"  # [변경] HuggingFace 데이터셋 이름
  train_ratio: 0.7     # 70% 학습
  val_ratio: 0.15      # 15% 검증 (하이퍼파라미터 조정용)
  test_ratio: 0.15     # 15% 테스트 (최종 1회만 사용)
  seed: 42

# ============================================================
# WandB
# ============================================================
wandb:
  project: "senti-fine-tuning"          # [변경] WandB 프로젝트명
  run_name: "qwen3-14b-qlora-senti"     # [변경] 실험 이름

# ============================================================
# 출력 경로
# ============================================================
output:
  output_dir: "./outputs"
  adapter_dir: "./outputs/adapter"      # LoRA 어댑터만 저장 (~수십MB)

# ============================================================
# 추론 설정
# ============================================================
inference:
  temperature: 0.1       # 낮을수록 결정적 (JSON 출력 일관성)
  max_new_tokens: 128    # [변경] 출력 길이에 따라 조정
```

---

## 4. 파이프라인 상세

### 4-1. 데이터 전처리 (data_loader.py)

핵심 흐름:

```
HuggingFace 데이터 다운로드
  → train/val/test 3분할 (stratified)
  → Chat 템플릿 적용 (system/user/assistant 메시지 → 단일 텍스트)
  → SFTTrainer에 전달
```

**프롬프트 구성** (새 태스크에서 이 부분을 변경):

```python
SYSTEM_PROMPT = """당신은 한국어 감성 분석 전문가입니다. 주어진 텍스트의 감성을 분석하고 아래 JSON 형식으로만 반환하세요.

출력 형식:
{"sentiment": "positive|negative|neutral", "probability": 0.0~1.0, "positive_topics": [...], "negative_topics": [...]}

규칙:
- sentiment는 반드시 positive, negative, neutral 중 하나
- probability는 해당 감성의 확신도 (0.0~1.0 사이 소수)
- JSON만 출력하고 다른 텍스트는 포함하지 마세요"""

USER_TEMPLATE = '다음 텍스트의 감성을 분석하세요:\n"{text}"'
```

**데이터 분할** — Stratified 3-way split:

```python
# 1단계: (train+val) / test 분리
split1 = ds.train_test_split(test_size=0.15, stratify_by_column="sentiment")

# 2단계: train / val 분리
split2 = split1["train"].train_test_split(test_size=val_adjusted, stratify_by_column="sentiment")
```

**Chat 템플릿 적용**:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,   # 학습: False (assistant 포함)
    enable_thinking=False,          # Qwen3 thinking mode 비활성화
)
```

### 4-2. 학습 (train.py)

실행 순서:

```
1. YAML 설정 로드
2. WandB 초기화 (실험 추적)
3. 토크나이저 로드 + pad_token 설정
4. 데이터 준비 (data_loader.prepare_dataset)
5. 토큰 길이 확인 (max_seq_length 적절성 검증)
6. 모델 로드 (4bit 양자화)
   - AutoModelForCausalLM.from_pretrained(quantization_config=...)
   - prepare_model_for_kbit_training(model)
7. LoRA 어댑터 부착
   - get_peft_model(model, lora_config)
8. SFTTrainer로 학습
   - EarlyStoppingCallback(patience=2)
9. LoRA 어댑터만 저장
```

**핵심 코드 패턴**:

```python
# 양자화 모델 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",     # Scaled Dot-Product Attention
)
model = prepare_model_for_kbit_training(model)

# LoRA 적용
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, ...)
model = get_peft_model(model, lora_config)

# SFTTrainer 학습
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(..., dataset_text_field="text", max_length=512),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
model.save_pretrained(adapter_dir)   # 어댑터만 저장 (~수십MB)
```

### 4-3. 평가 (evaluate.py)

Base 모델과 Fine-tuned 모델을 동일 test set에서 비교:

| 지표 | 설명 |
|------|------|
| `json_parse_rate` | 유효한 JSON 출력 비율 |
| `accuracy` | 감성 분류 정확도 |
| `f1_macro` | 클래스별 F1 평균 |
| `prob_mae` | 확률값 예측 오차 (낮을수록 좋음) |
| `pos_topic_f1` | 긍정 토픽 추출 F1 |
| `neg_topic_f1` | 부정 토픽 추출 F1 |

추가 분석:
- Confusion Matrix 출력
- 오답 분석 (틀린 샘플 상세 출력)
- 결과를 `outputs/eval_results.json`에 저장

**평가 시 추론 설정**:
```python
# greedy decoding (재현성 확보)
model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,          # 확률 가장 높은 토큰만 선택
    pad_token_id=tokenizer.pad_token_id,
)
```

### 4-4. 추론 (inference.py)

```bash
cd src && python inference.py --text "이 제품 정말 좋아요"
```

LoRA 어댑터를 base 모델에 병합하여 추론:
```python
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ...)
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()
```

### 4-5. 교차 검증 (cross_validate.py)

소규모 데이터(~150건)에서 평가 신뢰도를 높이기 위해 5-Fold Stratified Cross-Validation 지원:
- 전체 데이터를 5등분 → 각 fold를 test로 사용하여 5번 학습+평가
- 평균 ± 표준편차로 신뢰구간 제공
- WandB에 fold별 별도 run 기록

---

## 5. 새 프로젝트 적용 체크리스트

### Step 1: 프로젝트 복사 및 초기화

```bash
cp -r senti-fine-tuning-test/ new-project/
cd new-project
rm -rf outputs/ wandb/ reports/ .git
git init
```

### Step 2: 설정 변경 (training_config.yaml)

| 항목 | 변경 | 설명 |
|------|------|------|
| `model.name` | 필수 | 사용할 베이스 모델 |
| `model.max_seq_length` | 권장 | 데이터 길이에 맞게 조정 |
| `lora.r` | 권장 | 데이터 규모에 따라 8~64 |
| `lora.lora_dropout` | 권장 | 소규모 데이터: 0.1, 대규모: 0.05 |
| `lora.target_modules` | 필수 | 모델 아키텍처에 따라 다름 |
| `training.num_train_epochs` | 권장 | 소규모: 3~5, 대규모: 1~3 |
| `training.bf16` | 조건부 | GPU에 따라 fp16으로 변경 |
| `data.dataset_name` | 필수 | HuggingFace 데이터셋 경로 |
| `data.train/val/test_ratio` | 선택 | 데이터 규모에 따라 조정 |
| `wandb.project` | 필수 | 프로젝트명 |
| `wandb.run_name` | 필수 | 실험명 |
| `inference.max_new_tokens` | 권장 | 출력 길이에 맞게 |

### Step 3: 데이터 전처리 변경 (data_loader.py)

반드시 변경해야 하는 부분:

1. **SYSTEM_PROMPT** — 새 태스크의 역할, 출력 형식, 규칙 정의
2. **USER_TEMPLATE** — 입력 텍스트 포맷
3. **build_output_json()** — 데이터 컬럼에 맞게 정답 JSON 생성 로직 수정
4. **load_and_split()** — 데이터셋 컬럼명이 다르면 수정
   - `stratify_by_column` 대상 컬럼명
   - 데이터 전처리 로직

### Step 4: 평가 지표 변경 (evaluate.py)

태스크에 맞게 평가 지표 추가/제거:
- 분류: accuracy, f1, confusion matrix
- 생성: BLEU, ROUGE
- 추출: precision, recall, F1
- 파싱 성공률: JSON 등 구조화 출력의 경우

### Step 5: 실행

```bash
# 1. 환경 설정
pip install -r requirements.txt
# .env에 HF_TOKEN, WANDB_API_KEY 입력

# 2. EDA (선택)
jupyter notebook notebooks/eda.ipynb

# 3. 학습
cd src && python train.py

# 4. 평가
python evaluate.py

# 5. 추론
python inference.py --text "분석할 텍스트"

# 6. 교차 검증 (선택, 소규모 데이터)
python cross_validate.py
```

---

## 6. 하이퍼파라미터 튜닝 가이드

### 데이터 규모별 권장 설정

| 데이터 규모 | LoRA r | dropout | epochs | batch | accum | 실효 배치 |
|-------------|--------|---------|--------|-------|-------|-----------|
| ~100건 | 8 | 0.15 | 5~8 | 4 | 2 | 8 |
| ~500건 | 16 | 0.1 | 3~5 | 4 | 2~4 | 8~16 |
| ~2000건 | 32 | 0.05 | 2~3 | 4~8 | 4 | 16~32 |
| ~10000건+ | 64 | 0.05 | 1~2 | 8 | 4~8 | 32~64 |

### target_modules 모델별 참고

| 모델 | Attention | MLP |
|------|-----------|-----|
| Qwen3 | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj |
| Llama3 | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj |
| Mistral | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj |
| Gemma | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj |

> 대부분의 최신 LLM은 동일한 모듈명을 사용합니다.
> 확인: `model.named_modules()` 출력에서 `proj` 포함 레이어 확인

### GPU 메모리별 설정

| GPU | VRAM | 모델 규모 | batch_size | seq_length |
|-----|------|-----------|------------|------------|
| A100 80GB | 80GB | ~30B | 4~16 | 512~2048 |
| A100 40GB | 40GB | ~14B | 4~8 | 512~1024 |
| RTX 4090 | 24GB | ~7B | 2~4 | 256~512 |
| RTX 3090 | 24GB | ~7B | 2~4 | 256~512 |
| T4 | 16GB | ~3B | 1~2 | 256 |

---

## 7. Qwen3 모델 특이사항

이 프로젝트에서 사용된 Qwen3 특이 처리:

1. **Thinking Mode 비활성화**: Qwen3는 기본적으로 `<think>...</think>` 블록을 출력함
   - 학습 시: `enable_thinking=False`로 chat template 적용
   - 추론 시: `enable_thinking=False` + `strip_thinking()` 이중 안전장치

2. **pad_token 설정**: Qwen3는 pad_token이 없으므로 `eos_token`으로 대체
   ```python
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
   ```

3. **trust_remote_code=True**: Qwen3 모델 로딩 시 필수

---

## 8. 실험 관리 (WandB)

- 학습 중 loss, lr, GPU 사용량 실시간 모니터링
- 파라미터 변경 시 `wandb.run_name`을 변경하여 실험 구분
- 교차 검증 시 fold별 별도 run으로 기록

### 실험 비교 팁

```
wandb.run_name 예시:
  qwen3-14b-r16-ep5-lr2e4       # 기본
  qwen3-14b-r32-ep3-lr1e4       # LoRA rank 올림
  qwen3-14b-r16-ep5-lr2e4-mlp   # MLP 모듈 추가
```

---

## 9. 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| CUDA OOM | batch_size 또는 seq_length 과다 | batch 줄이기, seq_length 줄이기, gradient_accumulation 올리기 |
| Loss가 0에 수렴 | 과적합 | epoch 줄이기, dropout 올리기, 데이터 증강 |
| Loss가 안 내려감 | lr 너무 낮거나 rank 부족 | lr 올리기, r 올리기 |
| JSON 파싱 실패 높음 | 프롬프트 불명확 또는 학습 부족 | SYSTEM_PROMPT 개선, epoch 늘리기 |
| V100에서 에러 | bf16 미지원 | `bf16: false, fp16: true`로 변경 |
| warmup_ratio 경고 | trl 1.0+ deprecated | warmup_steps로 직접 계산 (코드에서 자동 처리) |
