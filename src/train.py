"""
SOBA 챗봇 파인튜닝 학습 스크립트

Qwen3-14B (또는 7B) 모델을 QLoRA 방식으로 파인튜닝합니다.
학습 데이터는 ChatML 포맷의 JSONL 파일(train.jsonl, val.jsonl)을 사용합니다.

실행:
    cd src && python train.py
    cd src && python train.py --config ../configs/training_config_7b.yaml  # 7B 실험
"""

import json
import os
import sys
from pathlib import Path

import torch
import yaml
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


# ============================================================
# 1. 설정 로드
# ============================================================

def load_config(config_path: str = None) -> dict:
    """
    YAML 설정 파일을 로드합니다.

    Args:
        config_path: 설정 파일 경로. None이면 기본 경로 사용.

    Returns:
        설정 딕셔너리
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"설정 로드: {config_path}")
    print(f"  모델: {config['model']['name']}")
    print(f"  LoRA r: {config['lora']['r']}")
    print(f"  Epochs: {config['training']['num_train_epochs']}")
    print(f"  max_seq_length: {config['model']['max_seq_length']}")

    return config


# ============================================================
# 2. 데이터 로드
# ============================================================

def load_dataset_from_jsonl(data_dir: str) -> tuple[Dataset, Dataset]:
    """
    processed/ 디렉토리에서 train.jsonl, val.jsonl을 로드합니다.

    각 행은 ChatML 포맷:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "task_type": "response"
    }

    SFTTrainer에 넘기기 위해 messages를 chat template으로 변환하여
    "text" 필드를 생성합니다.

    Args:
        data_dir: train.jsonl, val.jsonl이 있는 디렉토리 경로

    Returns:
        (train_dataset, val_dataset) 튜플
    """
    data_dir = Path(data_dir)

    def load_jsonl(path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data = load_jsonl(data_dir / "val.jsonl")

    print(f"\n데이터 로드:")
    print(f"  train: {len(train_data)}건")
    print(f"  val:   {len(val_data)}건")

    # 태스크별 분포 출력
    from collections import Counter
    train_tasks = Counter(d["task_type"] for d in train_data)
    print(f"  태스크 분포: {dict(train_tasks)}")

    # Dataset 객체로 변환
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def apply_chat_template(examples: dict, tokenizer) -> dict:
    """
    messages 필드를 토크나이저의 chat template으로 변환하여 "text" 필드를 생성합니다.

    Qwen3는 기본적으로 <think>...</think> 블록을 출력하므로,
    enable_thinking=False로 비활성화합니다.

    Args:
        examples: 배치 데이터 (messages 필드 포함)
        tokenizer: 토크나이저 인스턴스

    Returns:
        "text" 필드가 추가된 딕셔너리
    """
    texts = []
    for messages in examples["messages"]:
        # chat template 적용
        # add_generation_prompt=False: assistant 응답 포함 (학습용)
        # enable_thinking=False: Qwen3 thinking mode 비활성화
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        texts.append(text)

    return {"text": texts}


# ============================================================
# 3. 모델 로드 (QLoRA)
# ============================================================

def load_model_and_tokenizer(config: dict):
    """
    양자화된 모델과 토크나이저를 로드합니다.

    QLoRA 설정:
    - 4bit NF4 양자화 (FP16 ~28GB → 4bit ~8GB로 축소)
    - bfloat16 연산 (A100/H100에서 최적)
    - Double Quantization (추가 메모리 절약)

    Args:
        config: YAML 설정 딕셔너리

    Returns:
        (model, tokenizer) 튜플
    """
    model_name = config["model"]["name"]
    quant_cfg = config["quantization"]

    print(f"\n모델 로드: {model_name}")

    # --- 토크나이저 ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Qwen3는 pad_token이 없으므로 eos_token으로 대체
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  pad_token 설정: {tokenizer.pad_token}")

    # --- 양자화 설정 ---
    # 4bit 양자화로 모델 크기를 대폭 줄여 GPU 메모리를 절약합니다.
    # NF4 (NormalFloat4)는 FP4보다 정밀하고, double quant로 추가 절약합니다.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    # --- 모델 로드 ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",            # GPU에 자동 배치
        trust_remote_code=True,        # Qwen3 필수
        attn_implementation="sdpa",    # Scaled Dot-Product Attention (효율적)
    )

    # 양자화된 모델을 학습 가능하게 준비
    # gradient checkpointing 활성화, 일부 레이어 float32 변환 등
    model = prepare_model_for_kbit_training(model)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {total_params / 1e9:.1f}B")

    return model, tokenizer


# ============================================================
# 4. LoRA 어댑터 부착
# ============================================================

def apply_lora(model, config: dict):
    """
    모델에 LoRA 어댑터를 부착합니다.

    LoRA 원리: W_new = W_frozen + (alpha/r) * A * B
    - W_frozen: 원래 가중치 (동결, 학습 안 함)
    - A, B: 저랭크 행렬 (이것만 학습)
    - r: 랭크 (클수록 표현력↑, 메모리↑)
    - alpha: 스케일링 계수 (보통 r의 2배)

    학습되는 파라미터는 전체의 약 1~3%만 됩니다.

    Args:
        model: 양자화된 모델
        config: YAML 설정 딕셔너리

    Returns:
        LoRA가 적용된 모델
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 수 출력
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100

    print(f"\nLoRA 적용:")
    print(f"  r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, dropout={lora_cfg['lora_dropout']}")
    print(f"  학습 파라미터: {trainable / 1e6:.1f}M / {total / 1e6:.0f}M ({pct:.2f}%)")

    return model


# ============================================================
# 5. 학습 실행
# ============================================================

def train(config: dict):
    """
    전체 학습 파이프라인을 실행합니다.

    흐름:
    1. WandB 초기화 (실험 추적)
    2. 데이터 로드 (train.jsonl, val.jsonl)
    3. 모델 + 토크나이저 로드 (QLoRA)
    4. LoRA 어댑터 부착
    5. Chat template 적용 (messages → text)
    6. SFTTrainer로 학습
    7. LoRA 어댑터만 저장 (~수십MB)

    Args:
        config: YAML 설정 딕셔너리
    """
    # --- WandB 초기화 ---
    wandb_cfg = config["wandb"]
    wandb.init(
        project=wandb_cfg["project"],
        name=wandb_cfg["run_name"],
        config=config,
    )
    print(f"\nWandB: {wandb_cfg['project']} / {wandb_cfg['run_name']}")

    # --- 데이터 로드 ---
    data_path = Path(__file__).parent.parent / "data" / "processed"
    train_dataset, val_dataset = load_dataset_from_jsonl(str(data_path))

    # --- 모델 + 토크나이저 로드 ---
    model, tokenizer = load_model_and_tokenizer(config)

    # --- LoRA 적용 ---
    model = apply_lora(model, config)

    # --- Chat template 적용 ---
    # messages 배열을 토크나이저의 chat template으로 변환하여 "text" 필드 생성
    print("\nChat template 적용 중...")
    train_dataset = train_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        batched=True,
        desc="train",
    )
    val_dataset = val_dataset.map(
        lambda x: apply_chat_template(x, tokenizer),
        batched=True,
        desc="val",
    )

    # --- 토큰 길이 확인 ---
    # max_seq_length가 적절한지 검증
    sample_lengths = []
    for text in train_dataset["text"][:100]:
        tokens = tokenizer(text, return_tensors="pt")
        sample_lengths.append(tokens["input_ids"].shape[1])

    avg_len = sum(sample_lengths) / len(sample_lengths)
    max_len = max(sample_lengths)
    max_seq = config["model"]["max_seq_length"]

    print(f"\n토큰 길이 (샘플 100건):")
    print(f"  평균: {avg_len:.0f}, 최대: {max_len}")
    print(f"  max_seq_length: {max_seq}")
    if max_len > max_seq:
        print(f"  ⚠️ {max_len} > {max_seq}: 일부 데이터가 잘릴 수 있습니다")

    # --- 학습 설정 ---
    train_cfg = config["training"]
    output_cfg = config["output"]

    # warmup_steps 계산 (warmup_ratio 기반)
    total_steps = (
        len(train_dataset)
        // train_cfg["per_device_train_batch_size"]
        // train_cfg["gradient_accumulation_steps"]
        * train_cfg["num_train_epochs"]
    )
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    sft_config = SFTConfig(
        output_dir=output_cfg["output_dir"],

        # 학습 하이퍼파라미터
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        optim=train_cfg["optim"],

        # 정밀도
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],

        # 로깅 및 저장
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        report_to=train_cfg["report_to"],

        # SFT 설정
        dataset_text_field="text",          # chat template 적용된 텍스트 필드
        max_seq_length=config["model"]["max_seq_length"],
    )

    print(f"\n학습 설정:")
    print(f"  실효 배치: {train_cfg['per_device_train_batch_size']} × {train_cfg['gradient_accumulation_steps']} = {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  총 스텝: ~{total_steps}")
    print(f"  워밍업: {warmup_steps} 스텝")

    # --- 학습 실행 ---
    # EarlyStoppingCallback: eval_loss가 2 epoch 연속 개선 없으면 조기 종료
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n" + "=" * 60)
    print("학습 시작!")
    print("=" * 60)

    trainer.train()

    # --- LoRA 어댑터 저장 ---
    # 전체 모델이 아닌 LoRA 어댑터만 저장 (~수십MB)
    # 추론 시에는 base 모델 + 어댑터를 병합하여 사용
    adapter_dir = output_cfg["adapter_dir"]
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\n어댑터 저장: {adapter_dir}")

    # 어댑터 크기 출력
    adapter_size = sum(
        f.stat().st_size for f in Path(adapter_dir).rglob("*") if f.is_file()
    )
    print(f"어댑터 크기: {adapter_size / 1e6:.1f}MB")

    # --- 최종 메트릭 ---
    final_metrics = trainer.state.log_history
    train_losses = [h["loss"] for h in final_metrics if "loss" in h]
    eval_losses = [h["eval_loss"] for h in final_metrics if "eval_loss" in h]

    print(f"\n최종 메트릭:")
    if train_losses:
        print(f"  train_loss: {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"  eval_loss:  {eval_losses[-1]:.4f} (best: {min(eval_losses):.4f})")

    wandb.finish()
    print("\n학습 완료!")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOBA 챗봇 QLoRA 파인튜닝")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로 (기본: configs/training_config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
