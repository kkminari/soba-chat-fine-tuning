"""
Phase 6 수동 평가 스크립트

eval_predictions.jsonl + test.jsonl을 매칭하여
response 태스크 30건을 랜덤 샘플링하고, 수동 평가 시트(CSV)를 생성합니다.

실행:
    cd src && python manual_eval.py
"""

import json
import random
import csv
from pathlib import Path

SEED = 42
SAMPLE_SIZE = 30

def main():
    random.seed(SEED)

    # test.jsonl 로드 (컨텍스트 정보 포함)
    test_path = Path(__file__).parent.parent / "data" / "processed" / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f]

    # eval_predictions.jsonl 로드
    pred_path = Path(__file__).parent.parent / "outputs" / "eval_predictions.jsonl"
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = [json.loads(l) for l in f]

    # response 태스크만 필터 (test_data와 predictions 인덱스 매칭)
    response_items = []
    resp_idx = 0
    for i, item in enumerate(test_data):
        if item["task_type"] == "response":
            pred = predictions[resp_idx]  # predictions는 태스크 순서대로 저장됨
            assert pred["task"] == "response", f"Mismatch at index {i}"
            user_msg = item["messages"][1]["content"]
            response_items.append({
                "index": resp_idx,
                "user_input": user_msg,
                "expected": pred["expected"],
                "generated": pred["generated"],
            })
            resp_idx += 1

    # 실제로는 predictions가 태스크별 순서가 아닐 수 있으므로 직접 매칭
    # predictions에서 response만 추출하고, test_data의 response와 순서 매칭
    response_preds = [p for p in predictions if p["task"] == "response"]
    response_tests = [t for t in test_data if t["task_type"] == "response"]

    assert len(response_preds) == len(response_tests), \
        f"Count mismatch: preds={len(response_preds)}, tests={len(response_tests)}"

    items = []
    for idx, (pred, test) in enumerate(zip(response_preds, response_tests)):
        user_msg = test["messages"][1]["content"]
        items.append({
            "index": idx,
            "user_input": user_msg,
            "expected": pred["expected"],
            "generated": pred["generated"],
        })

    # 30건 랜덤 샘플링
    sampled = random.sample(items, min(SAMPLE_SIZE, len(items)))
    sampled.sort(key=lambda x: x["index"])

    # --- CSV 평가 시트 생성 ---
    output_dir = Path(__file__).parent.parent / "outputs"
    csv_path = output_dir / "manual_eval_sheet.csv"

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "순번", "인덱스",
            "주제/컨텍스트", "사용자응답",
            "기대_코멘트", "기대_다음질문",
            "생성_코멘트", "생성_다음질문",
            "톤적절성(1-5)", "리프레이징자연스러움(1-5)", "비고"
        ])

        for i, item in enumerate(sampled, 1):
            # 컨텍스트 파싱
            lines = item["user_input"].split("\n")
            topic = lines[0].replace("주제: ", "") if len(lines) > 0 else ""
            question = lines[1].replace("현재 질문: ", "") if len(lines) > 1 else ""
            answer = lines[2].replace("사용자 응답: ", "") if len(lines) > 2 else ""
            next_q = lines[3].replace("다음 질문: ", "") if len(lines) > 3 else ""

            context = f"[{topic}] Q: {question}"
            user_answer = answer

            # expected/generated 파싱
            try:
                exp = json.loads(item["expected"])
                exp_comment = exp.get("comment", "")
                exp_next_q = exp.get("next_question", "")
            except json.JSONDecodeError:
                exp_comment = item["expected"]
                exp_next_q = ""

            try:
                gen = json.loads(item["generated"])
                gen_comment = gen.get("comment", "")
                gen_next_q = gen.get("next_question", "")
            except json.JSONDecodeError:
                gen_comment = item["generated"]
                gen_next_q = ""

            writer.writerow([
                i, item["index"],
                context, user_answer,
                exp_comment, exp_next_q,
                gen_comment, gen_next_q,
                "", "", ""
            ])

    print(f"수동 평가 시트 생성: {csv_path}")
    print(f"총 {len(sampled)}건 샘플링 (seed={SEED})")

    # --- 콘솔 미리보기 ---
    print(f"\n{'='*80}")
    print("수동 평가 샘플 미리보기 (30건)")
    print(f"{'='*80}")

    for i, item in enumerate(sampled, 1):
        lines = item["user_input"].split("\n")
        topic = lines[0].replace("주제: ", "") if len(lines) > 0 else ""
        answer = lines[2].replace("사용자 응답: ", "") if len(lines) > 2 else ""

        try:
            gen = json.loads(item["generated"])
            gen_comment = gen.get("comment", "")
            gen_next_q = gen.get("next_question", "")
        except json.JSONDecodeError:
            gen_comment = item["generated"]
            gen_next_q = ""

        try:
            exp = json.loads(item["expected"])
            exp_comment = exp.get("comment", "")
            exp_next_q = exp.get("next_question", "")
        except json.JSONDecodeError:
            exp_comment = item["expected"]
            exp_next_q = ""

        print(f"\n--- [{i:02d}] idx={item['index']} ---")
        print(f"  주제: {topic}")
        print(f"  응답: {answer}")
        print(f"  [기대] 코멘트: {exp_comment}")
        print(f"  [생성] 코멘트: {gen_comment}")
        print(f"  [기대] 다음Q: {exp_next_q}")
        print(f"  [생성] 다음Q: {gen_next_q}")

    # --- JSON으로도 저장 (프로그래밍 평가용) ---
    json_path = output_dir / "manual_eval_samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 저장: {json_path}")


if __name__ == "__main__":
    main()
