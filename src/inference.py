"""
SOBA 챗봇 파인튜닝 모델 단일 추론 스크립트

학습된 모델에 직접 입력을 넣어 결과를 확인할 수 있습니다.

실행 예시:
    # response 태스크 테스트
    cd src && python inference.py \
        --task response \
        --topic "전통 소주 vs 신세대 소주" \
        --question "언제부터 현재 드시는 소주 브랜드를 마시기 시작하셨나요?" \
        --answer "한 30년 전부터요" \
        --next-question "소주를 함께 마시는 주요 대상이나 모임이 있으신가요?"

    # intro 태스크 테스트
    cd src && python inference.py \
        --task intro \
        --topic "20대 카페 이용 패턴"
"""

import json
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# 태스크별 system prompt (train.py의 convert_to_chatml과 동일)
# ============================================================

SYSTEM_PROMPTS = {
    "response": (
        "당신은 친근한 데이터 수집 챗봇입니다. "
        "사용자 응답에 공감 코멘트를 달고, 다음 질문으로 자연스럽게 전환하세요.\n\n"
        '출력 형식: {"comment": "...", "next_question": "..."}'
    ),
    "intro": "당신은 친근한 데이터 수집 챗봇입니다. 대화 주제를 소개하는 따뜻한 인트로 메시지를 생성하세요.",
    "first_question": "당신은 친근한 데이터 수집 챗봇입니다. 첫 번째 질문으로 자연스럽게 전환하세요.",
    "retry": "당신은 친근한 데이터 수집 챗봇입니다. 불성실한 응답에 부드럽게 재응답을 요청하세요.",
    "ending": "당신은 친근한 데이터 수집 챗봇입니다. 대화를 마무리하는 감사 메시지를 생성하세요.",
    "title": "당신은 마케팅 리서치 제목 생성기입니다. 15자 이내로 짧은 제목을 만드세요.",
}


def build_messages(args) -> list[dict]:
    """
    CLI 인자로부터 ChatML 메시지를 구성합니다.

    Args:
        args: argparse 결과

    Returns:
        [{"role": "system", ...}, {"role": "user", ...}]
    """
    system = SYSTEM_PROMPTS[args.task]

    if args.task == "response":
        user = (
            f"주제: {args.topic}\n"
            f"현재 질문: {args.question}\n"
            f"사용자 응답: {args.answer}\n"
            f"다음 질문: {args.next_question}"
        )
    elif args.task == "intro":
        user = f"주제: {args.topic}\n타겟: {args.target or ''}"
    elif args.task == "first_question":
        user = f"주제: {args.topic}\n첫 질문: {args.question}"
    elif args.task == "retry":
        user = f"주제: {args.topic}\n현재 질문: {args.question}\n부적절 유형: {args.invalid_type or 'too_short'}"
    elif args.task == "ending":
        user = f"주제: {args.topic}\n타겟: {args.target or ''}"
    elif args.task == "title":
        user = args.topic  # 원본 리서치 요청문
    else:
        raise ValueError(f"알 수 없는 태스크: {args.task}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SOBA 파인튜닝 모델 단일 추론")
    parser.add_argument("--task", required=True, choices=SYSTEM_PROMPTS.keys())
    parser.add_argument("--topic", required=True, help="주제 또는 원본 텍스트")
    parser.add_argument("--question", default="", help="현재 질문 (response, first_question, retry)")
    parser.add_argument("--answer", default="", help="사용자 응답 (response)")
    parser.add_argument("--next-question", default="", help="다음 질문 원본 (response)")
    parser.add_argument("--target", default="", help="타겟 (intro, ending)")
    parser.add_argument("--invalid-type", default="too_short", help="부적절 유형 (retry)")
    parser.add_argument("--config", default=None, help="설정 파일 경로")
    parser.add_argument("--adapter", default=None, help="어댑터 경로")
    args = parser.parse_args()

    # 설정 로드
    config_path = args.config or Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    adapter_dir = args.adapter or str(Path(__file__).parent.parent / "outputs" / "adapter")

    # 모델 로드
    model_name = config["model"]["name"]
    quant_cfg = config["quantization"]

    print(f"모델: {model_name}")
    print(f"어댑터: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # 메시지 구성
    messages = build_messages(args)

    # 추론
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    infer_cfg = config["inference"]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=infer_cfg["max_new_tokens"],
            temperature=infer_cfg["temperature"],
            do_sample=infer_cfg["temperature"] > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Qwen3 thinking 블록 제거
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 결과 출력
    print(f"\n{'=' * 50}")
    print(f"태스크: {args.task}")
    print(f"입력:")
    for m in messages:
        print(f"  [{m['role']}] {m['content'][:80]}...")
    print(f"\n출력:")
    print(f"  {text}")

    # response 태스크: JSON 파싱 시도
    if args.task == "response":
        try:
            parsed = json.loads(text)
            print(f"\n파싱 결과:")
            print(f"  코멘트: {parsed.get('comment', '')}")
            print(f"  다음 질문: {parsed.get('next_question', '')}")
        except json.JSONDecodeError:
            print(f"\n⚠️ JSON 파싱 실패")


if __name__ == "__main__":
    main()
