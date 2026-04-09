# SOBA Chatbot Fine-tuning

SOBA 마케팅 리서치 챗봇의 대화 생성 모델을 파인튜닝하는 프로젝트입니다.

현재 Claude Sonnet API로 동작하는 대화 생성을 오픈소스 모델(Qwen3-14B/7B)의 QLoRA 파인튜닝 모델로 교체합니다.

## 프로젝트 구조

```
soba-chat-fine-tuning/
├── configs/
│   └── training_config.yaml       # 학습 하이퍼파라미터
├── src/
│   ├── train.py                   # QLoRA 학습 스크립트
│   ├── evaluate.py                # 모델 평가 (test set 추론 + 메트릭)
│   └── inference.py               # 단일 추론 테스트
├── data/
│   └── processed/
│       ├── train.jsonl            # 학습 데이터 (4,342건)
│       ├── val.jsonl              # 검증 데이터 (513건)
│       └── test.jsonl             # 테스트 데이터 (578건)
├── data_generation/               # 학습 데이터 생성 파이프라인
│   ├── generate_data.py           # GPT-4o 비동기 데이터 생성기
│   ├── convert_to_chatml.py       # raw → ChatML 변환기
│   ├── seeds.json                 # 시드 데이터 (249건)
│   └── prompts/                   # 태스크별 생성 프롬프트
├── docs/
│   ├── FINETUNING_SETUP_GUIDE.md  # QLoRA 세팅 가이드
│   ├── SOBA_Training_Data_Report.pdf
│   └── plans/                     # 설계 문서 + 실행 추적
├── requirements.txt
└── .env.example
```

## 모델 교체 범위

| 구분 | 현재 | 변경 후 |
|------|------|---------|
| 대화 생성 (6개 태스크) | Claude Sonnet API | Qwen3-14B Fine-tuned |
| 응답 분석 | Claude Haiku API | 유지 |
| 질문 순서 결정 | Claude Haiku API | 유지 |

## 6개 대화 생성 태스크

| 태스크 | 설명 |
|--------|------|
| response | 사용자 응답에 공감 코멘트 + 다음 질문 리프레이징 |
| intro | 대화 시작 인트로 메시지 |
| first_question | 첫 질문으로 자연스럽게 전환 |
| retry | 불성실 응답에 부드러운 재요청 |
| ending | 대화 종료 감사 메시지 |
| title | 리서치 요청을 15자 이내 제목으로 변환 |

## 학습 데이터

- 시드: 249건 (술 144 + 화장품 105, TPOLEACI 8개 질문 프레임워크)
- 생성: GPT-4o / GPT-4o-mini API로 합성 생성
- 최종: 5,433건 (정제 후)
- 분할: train 4,342 / val 513 / test 578 (seed_id 기준, 누수 방지)
- 비용: $16.08

## 파인튜닝 설정

| 항목 | 값 |
|------|-----|
| 베이스 모델 | Qwen/Qwen3-14B (또는 7B) |
| 방식 | QLoRA (4bit NF4) |
| LoRA r | 32 |
| LoRA alpha | 64 |
| Epochs | 3 |
| Learning Rate | 2e-4 |
| max_seq_length | 768 |
| GPU | A100 80GB (RunPod) |

## 빠른 시작 (RunPod)

```bash
# 1. 클론
git clone https://github.com/kkminari/soba-chat-fine-tuning.git
cd soba-chat-fine-tuning

# 2. 환경 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# HF_TOKEN, WANDB_API_KEY 입력

# 4. 학습
cd src && python train.py

# 5. 평가
python evaluate.py

# 6. 단일 추론 테스트
python inference.py \
    --task response \
    --topic "전통 소주 vs 신세대 소주" \
    --question "언제부터 드시기 시작하셨나요?" \
    --answer "30년 전부터요" \
    --next-question "누구와 함께 드시나요?"
```

## 7B 모델로 실험

```bash
# training_config.yaml에서 모델명만 변경
# model.name: "Qwen/Qwen3-14B" → "Qwen/Qwen3-7B"

cd src && python train.py --config ../configs/training_config_7b.yaml
```

## 평가 기준

| 지표 | PASS 기준 |
|------|----------|
| JSON 파싱 성공률 | >= 95% |
| 코멘트 톤 적절성 | >= 85% |
| 질문 리프레이징 자연스러움 | >= 80% |
| 응답 시간 | < 3초 |
| 존댓말 사용률 | >= 95% |
| 금지어 (설문/조사) 미사용 | 0% |

## 출력

```
outputs/
├── adapter/                # LoRA 어댑터 (~수십MB)
├── eval_results.json       # 평가 메트릭
└── eval_samples.json       # 평가 샘플 (리뷰용)
```
