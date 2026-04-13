"""
SOBA 챗봇 파인튜닝 모델 평가 스크립트

test.jsonl 데이터에 대해 모델 추론을 수행하고,
자동 메트릭(JSON 파싱률, 톤 적절성 등)을 산출합니다.

실행:
    cd src && python evaluate.py
    cd src && python evaluate.py --adapter ../outputs/adapter  # 어댑터 경로 지정
"""

import json
import re
import os
from pathlib import Path
from collections import Counter

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


# ============================================================
# 1. 설정 및 모델 로드
# ============================================================

def load_model_for_eval(config: dict, adapter_dir: str):
    """
    Base 모델 + LoRA 어댑터를 로드하여 평가용 모델을 준비합니다.

    Args:
        config: YAML 설정 딕셔너리
        adapter_dir: LoRA 어댑터 디렉토리 경로

    Returns:
        (model, tokenizer) 튜플
    """
    model_name = config["model"]["name"]
    quant_cfg = config["quantization"]

    print(f"모델 로드: {model_name}")
    print(f"어댑터: {adapter_dir}")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    # Base 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 어댑터 병합
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    print("모델 로드 완료 (eval 모드)")
    return model, tokenizer


# ============================================================
# 2. 추론
# ============================================================

def generate_response(model, tokenizer, messages: list[dict], config: dict) -> str:
    """
    ChatML 메시지에 대해 모델 추론을 수행합니다.

    system + user 메시지를 입력하고, assistant 응답을 생성합니다.

    Args:
        model: 파인튜닝된 모델
        tokenizer: 토크나이저
        messages: [{"role": "system", ...}, {"role": "user", ...}]
        config: YAML 설정 (inference 섹션)

    Returns:
        생성된 텍스트
    """
    infer_cfg = config["inference"]

    # system + user만 넣고, assistant 응답을 생성하게 함
    input_messages = [m for m in messages if m["role"] != "assistant"]

    prompt = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,    # assistant 시작 토큰 추가
        enable_thinking=False,         # Qwen3 thinking mode 비활성화
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=infer_cfg["max_new_tokens"],
            temperature=infer_cfg["temperature"],
            do_sample=infer_cfg["temperature"] > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 입력 부분 제거, 생성된 부분만 디코딩
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Qwen3 thinking 블록 제거 (안전장치)
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return text


# ============================================================
# 3. 평가 메트릭
# ============================================================

def evaluate_response_task(predictions: list[dict]) -> dict:
    """
    response 태스크의 예측 결과를 평가합니다.

    평가 항목:
    - JSON 파싱 성공률
    - comment/next_question 키 존재 여부
    - 코멘트 길이 (10~100자)
    - 존댓말 사용률
    - 이모지 미사용률
    - 금지어 미사용률

    Args:
        predictions: [{"expected": "...", "generated": "..."}]

    Returns:
        메트릭 딕셔너리
    """
    emoji_p = re.compile(r"[\U0001F600-\U0001F9FF\U0001FA00-\U0001FA6F\u2600-\u26FF]")
    honorific_p = re.compile(r"(요|세요|네요|습니다|겠어요|드려요|죠|지요|군요|이에요|예요|인가요|나요|할까요|볼까요|실까요|던가요|합니다|됩니다|랍니다)\b")
    survey_w = ["설문", "조사", "서베이"]

    json_ok = 0
    keys_ok = 0
    length_ok = 0
    honorific_ok = 0
    no_emoji = 0
    no_survey = 0
    total = len(predictions)

    for p in predictions:
        gen = p["generated"]

        # JSON 파싱
        try:
            parsed = json.loads(gen)
            json_ok += 1

            # 키 존재
            if "comment" in parsed and "next_question" in parsed:
                keys_ok += 1
                comment = parsed["comment"]

                # 코멘트 길이
                if 10 <= len(comment) <= 100:
                    length_ok += 1

                # 존댓말
                if honorific_p.search(comment):
                    honorific_ok += 1

                # 이모지
                if not emoji_p.search(comment + parsed["next_question"]):
                    no_emoji += 1

                # 금지어
                if not any(w in comment + parsed["next_question"] for w in survey_w):
                    no_survey += 1
        except json.JSONDecodeError:
            pass

    return {
        "json_parse_rate": json_ok / total * 100 if total else 0,
        "keys_present_rate": keys_ok / total * 100 if total else 0,
        "comment_length_ok": length_ok / total * 100 if total else 0,
        "honorific_rate": honorific_ok / total * 100 if total else 0,
        "no_emoji_rate": no_emoji / total * 100 if total else 0,
        "no_survey_word_rate": no_survey / total * 100 if total else 0,
        "total_evaluated": total,
    }


# ============================================================
# 4. 메인 평가 루프
# ============================================================

def run_evaluation(config: dict, adapter_dir: str):
    """
    전체 평가 파이프라인을 실행합니다.

    1. 모델 로드
    2. test.jsonl에서 태스크별 추론
    3. 자동 메트릭 산출
    4. 결과 저장

    Args:
        config: YAML 설정 딕셔너리
        adapter_dir: LoRA 어댑터 경로
    """
    # --- 모델 로드 ---
    model, tokenizer = load_model_for_eval(config, adapter_dir)

    # --- test 데이터 로드 ---
    data_dir = config.get("data", {}).get("data_path", "data/processed")
    test_path = Path(__file__).parent.parent / data_dir / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f]

    print(f"\ntest 데이터: {len(test_data)}건")

    # --- 태스크별 추론 ---
    predictions = {}
    for item in tqdm(test_data, desc="추론"):
        task = item["task_type"]
        if task not in predictions:
            predictions[task] = []

        # 모델에 system + user만 입력, assistant 생성
        generated = generate_response(model, tokenizer, item["messages"], config)

        predictions[task].append({
            "expected": item["messages"][2]["content"],  # 정답
            "generated": generated,                       # 모델 출력
        })

    # --- 메트릭 산출 ---
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)

    results = {}

    # response 태스크 상세 평가
    if "response" in predictions:
        resp_metrics = evaluate_response_task(predictions["response"])
        results["response"] = resp_metrics

        print(f"\n[response] ({resp_metrics['total_evaluated']}건)")
        for k, v in resp_metrics.items():
            if k != "total_evaluated":
                status = "PASS" if v >= 80 else "FAIL"
                print(f"  {k}: {v:.1f}% [{status}]")

    # 기타 태스크: 간단한 길이/형식 체크
    for task in ["intro", "first_question", "retry", "ending", "title"]:
        if task not in predictions:
            continue
        preds = predictions[task]
        non_empty = sum(1 for p in preds if len(p["generated"].strip()) > 0)
        rate = non_empty / len(preds) * 100
        results[task] = {"non_empty_rate": rate, "total": len(preds)}
        print(f"\n[{task}] ({len(preds)}건): 비어있지 않은 응답 {rate:.1f}%")

    # --- 결과 저장 ---
    out_cfg = config.get("output", {}).get("output_dir", "outputs")
    output_dir = Path(__file__).parent.parent / out_cfg
    output_dir.mkdir(parents=True, exist_ok=True)

    # 메트릭 저장
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 전체 예측 결과 저장
    all_predictions = []
    for task, preds in predictions.items():
        for p in preds:
            all_predictions.append({"task": task, **p})

    with open(output_dir / "eval_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 예측 샘플 저장 (리뷰용, 태스크당 5건)
    samples = []
    for task, preds in predictions.items():
        for p in preds[:5]:
            samples.append({"task": task, **p})

    with open(output_dir / "eval_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_dir}/")
    print("  eval_results.json (메트릭)")
    print("  eval_samples.json (샘플 30건)")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOBA 파인튜닝 모델 평가")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA 어댑터 경로 (기본: outputs/adapter)",
    )
    args = parser.parse_args()

    # 설정 로드
    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 어댑터 경로
    adapter_dir = args.adapter
    if adapter_dir is None:
        adapter_dir = str(Path(__file__).parent.parent / "outputs" / "adapter")

    run_evaluation(config, adapter_dir)
