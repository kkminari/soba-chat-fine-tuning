"""
Phase 6 통합 세션 테스트 (end-to-end)

실제 모델을 로드하여 다양한 시나리오에서 추론을 수행합니다.

시나리오:
1. 정상 플로우 (8질문 완주) x 2건
2. 짧은 응답 (1~3단어) x 5건
3. 부정적/무관심 응답 x 5건
4. 긴 응답 (100자+) x 5건
5. 무관한 응답 (retry 테스트) x 3건

실행:
    cd src && python session_test.py --adapter ./outputs_exp3a/adapter
"""

import json
import re
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


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


def generate(model, tokenizer, messages, infer_cfg):
    """단일 추론"""
    input_messages = [m for m in messages if m["role"] != "assistant"]
    prompt = tokenizer.apply_chat_template(
        input_messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=infer_cfg["max_new_tokens"],
            temperature=infer_cfg["temperature"],
            do_sample=infer_cfg["temperature"] > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return text, elapsed


def check_response(text, task):
    """응답 품질 체크"""
    issues = []

    if not text.strip():
        issues.append("EMPTY")
        return False, issues

    if task == "response":
        try:
            parsed = json.loads(text)
            if "comment" not in parsed:
                issues.append("NO_COMMENT_KEY")
            if "next_question" not in parsed:
                issues.append("NO_NEXT_Q_KEY")

            comment = parsed.get("comment", "")
            next_q = parsed.get("next_question", "")

            # 존댓말 체크
            honorific_p = re.compile(r"(요|세요|네요|습니다|겠어요|죠|지요|군요|이에요|예요|나요)\b")
            if not honorific_p.search(comment):
                issues.append("NO_HONORIFIC_COMMENT")

            # 금지어
            for w in ["설문", "조사", "서베이"]:
                if w in comment + next_q:
                    issues.append(f"FORBIDDEN_WORD:{w}")

            # 사진 요청
            for w in ["사진", "보여주", "보내주"]:
                if w in next_q:
                    issues.append(f"PHOTO_REQUEST:{w}")

        except json.JSONDecodeError:
            issues.append("JSON_PARSE_FAIL")
            return False, issues

    elif task == "retry":
        # 존댓말 체크
        honorific_p = re.compile(r"(요|세요|네요|습니다|겠어요|죠|지요|나요)\b")
        if not honorific_p.search(text):
            issues.append("NO_HONORIFIC")

    return len(issues) == 0, issues


# ============================================================
# 테스트 시나리오 정의
# ============================================================

NORMAL_FLOW_QUESTIONS = [
    {
        "topic": "20대 카페 이용 패턴",
        "questions": [
            ("카페에 자주 가시나요?", "거의 매일 가요. 커피를 좋아해서요."),
            ("주로 어떤 음료를 드시나요?", "아메리카노를 가장 자주 마셔요. 가끔 라떼도요."),
            ("카페에서 보통 뭘 하시나요?", "공부하거나 친구랑 수다 떨어요."),
            ("혼자 갈 때와 함께 갈 때 차이가 있나요?", "혼자 가면 집중하고, 같이 가면 대화를 많이 해요."),
            ("카페 선택 기준이 있으시나요?", "분위기랑 가격이요. 너무 비싸면 안 가요."),
            ("한 달에 카페 비용이 얼마나 드나요?", "한 10만 원 정도 쓰는 것 같아요."),
            ("카페 방문 횟수를 줄이고 싶은 생각이 있나요?", "줄이고는 싶은데 습관이 되어서 어렵네요."),
            ("미래에 카페 이용 패턴이 어떻게 변할 것 같나요?", "아마 비슷할 것 같아요. 직장 다니면서도 갈 것 같아요."),
        ],
        "next_questions": [
            "주로 어떤 음료를 드시나요?",
            "카페에서 보통 뭘 하시나요?",
            "혼자 갈 때와 함께 갈 때 차이가 있나요?",
            "카페 선택 기준이 있으시나요?",
            "한 달에 카페 비용이 얼마나 드나요?",
            "카페 방문 횟수를 줄이고 싶은 생각이 있나요?",
            "미래에 카페 이용 패턴이 어떻게 변할 것 같나요?",
            None,  # 마지막 질문
        ],
    },
    {
        "topic": "30대 맞벌이 부부 음주 문화",
        "questions": [
            ("부부가 함께 술을 마시는 시간은 어떻게 만드시나요?", "주말 저녁에 와인 한 잔 마셔요."),
            ("주로 어떤 술을 즐기시나요?", "레드와인을 좋아해요. 가끔 맥주도요."),
            ("술을 마시면서 나누는 대화 주제가 있나요?", "보통 한 주간 있었던 일 얘기해요."),
            ("음주 비용은 한 달에 얼마나 되나요?", "5만 원 정도예요. 마트에서 사니까."),
            ("부부 음주가 관계에 어떤 영향을 주나요?", "대화가 늘어서 좋아요."),
            ("음주량을 줄이려는 노력을 하시나요?", "건강 생각해서 한 잔만 마시려고요."),
            ("외식할 때도 술을 드시나요?", "외식하면 맥주 한 잔 정도요."),
            ("앞으로 음주 습관을 어떻게 유지할 생각이세요?", "지금처럼 적당히 마실 거예요."),
        ],
        "next_questions": [
            "주로 어떤 술을 즐기시나요?",
            "술을 마시면서 나누는 대화 주제가 있나요?",
            "음주 비용은 한 달에 얼마나 되나요?",
            "부부 음주가 관계에 어떤 영향을 주나요?",
            "음주량을 줄이려는 노력을 하시나요?",
            "외식할 때도 술을 드시나요?",
            "앞으로 음주 습관을 어떻게 유지할 생각이세요?",
            None,
        ],
    },
]

SHORT_ANSWERS = [
    ("전통 소주", "언제부터 소주를 드시기 시작하셨나요?", "오래전부터요", "소주를 함께 마시는 주요 대상이 있나요?"),
    ("저도수 맥주", "저도수 맥주를 처음 마셔보신 적은?", "한번요", "어떤 브랜드를 드셔보셨나요?"),
    ("와인 문화", "와인을 즐기시는 빈도는?", "가끔요", "주로 어떤 종류를 선호하시나요?"),
    ("남성 그루밍", "평소 스킨케어 루틴이 있으시나요?", "없어요", "세안 후 바르는 제품이 있나요?"),
    ("카페 문화", "카페에 자주 가시나요?", "네", "주로 어떤 음료를 드시나요?"),
]

NEGATIVE_ANSWERS = [
    ("전통 소주", "소주를 좋아하시나요?", "별로요. 관심 없어요.", "어떤 계기로 마시게 되셨나요?"),
    ("저도수 맥주", "저도수 맥주에 대해 어떻게 생각하세요?", "솔직히 맛없어요. 그냥 일반 맥주가 낫죠.", "어떤 상황에서 선택하시나요?"),
    ("시니어 음주", "술자리를 즐기시나요?", "아니요. 술은 안 마시고 싶어요. 몸도 안 좋고.", "건강 때문에 줄이시게 된 건가요?"),
    ("그루밍 제품", "화장품에 관심이 있으신가요?", "전혀요. 남자가 무슨 화장이에요.", "주변에서 권하신 적은 없으셨나요?"),
    ("음주 문화", "음주가 스트레스 해소에 도움이 되나요?", "아뇨. 오히려 더 우울해져요. 후회만 남아요.", "다른 스트레스 해소법은 있으신가요?"),
]

LONG_ANSWERS = [
    ("카페 문화", "카페에서 보통 뭘 하시나요?",
     "저는 보통 노트북을 가져가서 작업을 해요. 카페가 집보다 집중이 잘 되거든요. 특히 백색소음이 좋아서요. 가끔 친구를 만나면 수다도 떨고 커피도 여러 잔 마시고 그래요. 최근에는 독서 모임도 카페에서 하고 있어요.",
     "카페 선택 시 가장 중요한 기준은 무엇인가요?"),
    ("와인 문화", "와인을 마시는 상황을 알려주세요",
     "저는 주로 기념일이나 특별한 날에 마셔요. 결혼기념일에는 꼭 좋은 와인을 사는데, 작년에는 이탈리아 바롤로를 샀어요. 가격이 좀 나갔지만 분위기가 정말 좋았거든요. 부부끼리 조용히 마시니까 대화도 많아지고 좋더라고요.",
     "와인 선택 시 중요하게 보는 점은 무엇인가요?"),
    ("음주 패턴", "음주 후 다음 날 컨디션은 어떠세요?",
     "예전에는 괜찮았는데 나이가 드니까 숙취가 심해졌어요. 특히 소주를 마시면 다음 날 머리가 아프고 속이 안 좋아요. 그래서 요즘은 맥주 위주로 마시는데 그래도 좀 나아요. 건강검진에서 간 수치도 좀 높다 해서 신경 쓰이네요.",
     "건강을 생각해서 음주량을 조절하시나요?"),
    ("남성 그루밍", "피부 고민이 있으시나요?",
     "네, 건조함이 제일 문제예요. 겨울이면 피부가 갈라지고 각질도 많이 생겨요. 아내가 보습크림 써보라고 해서 한번 써봤는데 좀 나아지긴 했어요. 근데 매일 바르기가 귀찮더라고요. 그래도 요즘은 세안 후에 토너랑 로션은 꼭 발라요.",
     "보습 제품은 어떤 것을 사용하고 계시나요?"),
    ("시니어 음주", "동네 분들과의 술자리는 어떤가요?",
     "한 달에 두세 번은 동네 형님들이랑 만나요. 보통 동네 포차에서 소주에 안주 시켜놓고 이것저것 얘기해요. 건강 이야기도 하고 손주 자랑도 하고. 그런데 가끔은 음주량이 걱정돼요. 의사가 줄이라고 했거든요.",
     "의사의 권고 후 음주량에 변화가 있으셨나요?"),
]

IRRELEVANT_ANSWERS = [
    ("전통 소주", "소주를 마시는 주된 이유가 뭔가요?", "아 배고프다 점심 뭐 먹지", "too_short"),
    ("저도수 맥주", "저도수 맥주를 어디서 구매하시나요?", "ㅋㅋㅋㅋㅋㅋㅋ", "too_short"),
    ("카페 문화", "주로 어떤 카페를 가시나요?", "asdfghjkl 12345", "irrelevant"),
]


def run_normal_flow(model, tokenizer, infer_cfg, scenario, scenario_idx):
    """정상 플로우 테스트 — 8질문 완주"""
    topic = scenario["topic"]
    questions = scenario["questions"]
    next_qs = scenario["next_questions"]
    results = []

    print(f"\n{'='*60}")
    print(f"정상 플로우 #{scenario_idx}: {topic}")
    print(f"{'='*60}")

    all_pass = True
    for i, ((question, answer), next_q) in enumerate(zip(questions, next_qs)):
        user_content = f"주제: {topic}\n현재 질문: {question}\n사용자 응답: {answer}"
        if next_q:
            user_content += f"\n다음 질문: {next_q}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["response"]},
            {"role": "user", "content": user_content},
        ]

        text, elapsed = generate(model, tokenizer, messages, infer_cfg)
        passed, issues = check_response(text, "response")
        if not passed:
            all_pass = False

        status = "OK" if passed else f"ISSUE: {', '.join(issues)}"
        print(f"  Q{i+1}: {question[:40]}...")
        print(f"    A: {answer[:40]}...")
        print(f"    생성: {text[:80]}...")
        print(f"    시간: {elapsed:.2f}s | {status}")

        results.append({
            "question_idx": i + 1,
            "question": question,
            "answer": answer,
            "generated": text,
            "elapsed": round(elapsed, 2),
            "passed": passed,
            "issues": issues,
        })

    return all_pass, results


def run_short_answer_tests(model, tokenizer, infer_cfg):
    """짧은 응답 테스트"""
    results = []
    print(f"\n{'='*60}")
    print("짧은 응답 테스트 (1~3단어)")
    print(f"{'='*60}")

    for i, (topic, question, answer, next_q) in enumerate(SHORT_ANSWERS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["response"]},
            {"role": "user", "content": f"주제: {topic}\n현재 질문: {question}\n사용자 응답: {answer}\n다음 질문: {next_q}"},
        ]
        text, elapsed = generate(model, tokenizer, messages, infer_cfg)
        passed, issues = check_response(text, "response")

        status = "OK" if passed else f"ISSUE: {', '.join(issues)}"
        print(f"  [{i}] 응답: '{answer}' → {text[:80]}... | {elapsed:.2f}s | {status}")

        results.append({"topic": topic, "answer": answer, "generated": text, "elapsed": round(elapsed, 2), "passed": passed, "issues": issues})

    return results


def run_negative_tests(model, tokenizer, infer_cfg):
    """부정적/무관심 응답 테스트"""
    results = []
    print(f"\n{'='*60}")
    print("부정적/무관심 응답 테스트")
    print(f"{'='*60}")

    for i, (topic, question, answer, next_q) in enumerate(NEGATIVE_ANSWERS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["response"]},
            {"role": "user", "content": f"주제: {topic}\n현재 질문: {question}\n사용자 응답: {answer}\n다음 질문: {next_q}"},
        ]
        text, elapsed = generate(model, tokenizer, messages, infer_cfg)
        passed, issues = check_response(text, "response")

        # 추가 체크: 공감적 톤인지
        try:
            parsed = json.loads(text)
            comment = parsed.get("comment", "")
            # 강요적 표현 체크
            forceful = ["꼭", "반드시", "무조건", "해야", "드셔야"]
            for w in forceful:
                if w in comment:
                    issues.append(f"FORCEFUL:{w}")
                    passed = False
        except json.JSONDecodeError:
            pass

        status = "OK" if passed else f"ISSUE: {', '.join(issues)}"
        print(f"  [{i}] 응답: '{answer[:40]}' → {text[:80]}... | {elapsed:.2f}s | {status}")

        results.append({"topic": topic, "answer": answer, "generated": text, "elapsed": round(elapsed, 2), "passed": passed, "issues": issues})

    return results


def run_long_answer_tests(model, tokenizer, infer_cfg):
    """긴 응답 테스트"""
    results = []
    print(f"\n{'='*60}")
    print("긴 응답 테스트 (100자+)")
    print(f"{'='*60}")

    for i, (topic, question, answer, next_q) in enumerate(LONG_ANSWERS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["response"]},
            {"role": "user", "content": f"주제: {topic}\n현재 질문: {question}\n사용자 응답: {answer}\n다음 질문: {next_q}"},
        ]
        text, elapsed = generate(model, tokenizer, messages, infer_cfg)
        passed, issues = check_response(text, "response")

        # 코멘트 길이 체크
        try:
            parsed = json.loads(text)
            comment = parsed.get("comment", "")
            if len(comment) > 100:
                issues.append(f"COMMENT_TOO_LONG({len(comment)})")
        except json.JSONDecodeError:
            pass

        status = "OK" if passed else f"ISSUE: {', '.join(issues)}"
        print(f"  [{i}] 응답({len(answer)}자): {answer[:40]}...")
        print(f"    생성: {text[:80]}... | {elapsed:.2f}s | {status}")

        results.append({"topic": topic, "answer_len": len(answer), "generated": text, "elapsed": round(elapsed, 2), "passed": passed, "issues": issues})

    return results


def run_irrelevant_tests(model, tokenizer, infer_cfg):
    """무관한 응답 → retry 테스트"""
    results = []
    print(f"\n{'='*60}")
    print("무관한 응답 → retry 테스트")
    print(f"{'='*60}")

    for i, (topic, question, answer, invalid_type) in enumerate(IRRELEVANT_ANSWERS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["retry"]},
            {"role": "user", "content": f"주제: {topic}\n현재 질문: {question}\n부적절 유형: {invalid_type}"},
        ]
        text, elapsed = generate(model, tokenizer, messages, infer_cfg)
        passed, issues = check_response(text, "retry")

        status = "OK" if passed else f"ISSUE: {', '.join(issues)}"
        print(f"  [{i}] 무관 응답: '{answer}' → retry: {text[:80]}... | {elapsed:.2f}s | {status}")

        results.append({"topic": topic, "invalid_answer": answer, "generated": text, "elapsed": round(elapsed, 2), "passed": passed, "issues": issues})

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent.parent / "configs" / "training_config_exp3a.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    adapter_dir = args.adapter or str(Path(__file__).parent / "outputs_exp3a" / "adapter")

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
    print("모델 로드 완료\n")

    infer_cfg = config["inference"]

    all_results = {}
    latencies = []

    # 1. 정상 플로우
    normal_all_pass = True
    normal_results = []
    for idx, scenario in enumerate(NORMAL_FLOW_QUESTIONS, 1):
        passed, results = run_normal_flow(model, tokenizer, infer_cfg, scenario, idx)
        if not passed:
            normal_all_pass = False
        normal_results.extend(results)
        latencies.extend([r["elapsed"] for r in results])
    all_results["normal_flow"] = {"passed": normal_all_pass, "details": normal_results}

    # 2. 짧은 응답
    short_results = run_short_answer_tests(model, tokenizer, infer_cfg)
    latencies.extend([r["elapsed"] for r in short_results])
    short_pass = sum(1 for r in short_results if r["passed"])
    all_results["short_answers"] = {"passed": short_pass, "total": len(short_results), "details": short_results}

    # 3. 부정적 응답
    neg_results = run_negative_tests(model, tokenizer, infer_cfg)
    latencies.extend([r["elapsed"] for r in neg_results])
    neg_pass = sum(1 for r in neg_results if r["passed"])
    all_results["negative_answers"] = {"passed": neg_pass, "total": len(neg_results), "details": neg_results}

    # 4. 긴 응답
    long_results = run_long_answer_tests(model, tokenizer, infer_cfg)
    latencies.extend([r["elapsed"] for r in long_results])
    long_pass = sum(1 for r in long_results if r["passed"])
    all_results["long_answers"] = {"passed": long_pass, "total": len(long_results), "details": long_results}

    # 5. 무관한 응답
    irr_results = run_irrelevant_tests(model, tokenizer, infer_cfg)
    latencies.extend([r["elapsed"] for r in irr_results])
    irr_pass = sum(1 for r in irr_results if r["passed"])
    all_results["irrelevant_answers"] = {"passed": irr_pass, "total": len(irr_results), "details": irr_results}

    # 집계
    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    p99 = latencies[int(len(latencies)*0.99)]

    print(f"\n{'='*60}")
    print("통합 세션 테스트 최종 결과")
    print(f"{'='*60}")
    print(f"\n1. 정상 플로우: {'PASS' if normal_all_pass else 'FAIL'} ({len([r for r in normal_results if r['passed']])}/{len(normal_results)})")
    print(f"2. 짧은 응답: {short_pass}/{len(short_results)} 통과")
    print(f"3. 부정적 응답: {neg_pass}/{len(neg_results)} 통과")
    print(f"4. 긴 응답: {long_pass}/{len(long_results)} 통과")
    print(f"5. 무관한 응답(retry): {irr_pass}/{len(irr_results)} 통과")
    print(f"\n응답 시간 (inference):")
    print(f"  p50: {p50:.2f}s")
    print(f"  p95: {p95:.2f}s")
    print(f"  p99: {p99:.2f}s")
    print(f"  PASS 기준 (< 3s): {'PASS' if p95 < 3 else 'FAIL'}")

    all_results["latency"] = {"p50": round(p50, 2), "p95": round(p95, 2), "p99": round(p99, 2)}

    # 결과 저장
    output_path = Path(__file__).parent.parent / "outputs" / "session_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
