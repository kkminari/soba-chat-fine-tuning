"""
Phase 6 수동 평가 — 자동 채점 스크립트

30건 샘플에 대해 다음 항목을 규칙 기반 + 휴리스틱으로 평가:
1. 톤 적절성 (1-5): 공감적이고 친근한가, 존댓말 사용, 비강요적
2. 리프레이징 자연스러움 (1-5): 원본 질문 대비 자연스럽게 재구성했는가

최종 결과를 CSV와 JSON으로 저장.
"""

import json
import re
from pathlib import Path


def evaluate_tone(comment: str, user_answer: str) -> tuple[int, str]:
    """
    코멘트의 톤 적절성을 평가합니다.

    평가 기준:
    - 존댓말 사용 여부
    - 공감 표현 포함 여부
    - 적절한 길이 (너무 짧거나 길지 않음)
    - 비강요적 (설문/조사 등 금지어 없음)
    - 사용자 응답에 대한 관련성

    Returns:
        (점수 1-5, 사유)
    """
    score = 5
    reasons = []

    # 존댓말 체크
    honorific_p = re.compile(
        r"(요|세요|네요|습니다|겠어요|드려요|죠|지요|군요|이에요|예요|인가요|나요|할까요|볼까요|실까요|던가요|합니다|됩니다|랍니다)\b"
    )
    if not honorific_p.search(comment):
        score -= 2
        reasons.append("존댓말 미사용")

    # 공감 표현 체크
    empathy_words = [
        "이해", "공감", "그렇", "맞", "좋", "중요", "멋지", "대단",
        "다행", "응원", "소중", "인상적", "재미", "흥미", "편안",
        "부담", "어려", "힘드", "아쉽", "노력"
    ]
    has_empathy = any(w in comment for w in empathy_words)
    if not has_empathy:
        score -= 1
        reasons.append("공감표현 부족")

    # 길이 체크
    if len(comment) < 5:
        score -= 1
        reasons.append("너무 짧음")
    elif len(comment) > 80:
        score -= 1
        reasons.append("너무 김")

    # 금지어 체크
    forbidden = ["설문", "조사", "서베이", "테스트", "실험"]
    if any(w in comment for w in forbidden):
        score -= 2
        reasons.append(f"금지어 포함")

    # 부자연스러운 표현 체크
    awkward = ["사진", "보여주", "보내주"]
    if any(w in comment for w in awkward):
        score -= 1
        reasons.append("부적절 요청 포함")

    score = max(1, min(5, score))
    reason = ", ".join(reasons) if reasons else "양호"
    return score, reason


def evaluate_rephrasing(gen_next_q: str, exp_next_q: str, original_next_q: str) -> tuple[int, str]:
    """
    다음 질문 리프레이징의 자연스러움을 평가합니다.

    평가 기준:
    - 원본 질문의 의도를 유지하는가
    - 자연스러운 한국어인가
    - 존댓말 사용
    - 적절한 길이
    - 사진/이미지 요청 같은 부적절한 요소 없음

    Returns:
        (점수 1-5, 사유)
    """
    score = 5
    reasons = []

    # 존댓말 체크
    honorific_p = re.compile(
        r"(요|세요|네요|습니다|겠어요|죠|지요|나요|인가요|할까요|볼까요|실까요|던가요)\b"
    )
    if not honorific_p.search(gen_next_q):
        score -= 2
        reasons.append("존댓말 미사용")

    # 길이 체크
    if len(gen_next_q) < 5:
        score -= 2
        reasons.append("너무 짧음")
    elif len(gen_next_q) > 100:
        score -= 1
        reasons.append("너무 김")

    # 의미 유사성 (키워드 오버랩 기반)
    # 기대 질문과 생성 질문의 핵심 키워드 비교
    exp_words = set(re.findall(r"[가-힣]{2,}", exp_next_q))
    gen_words = set(re.findall(r"[가-힣]{2,}", gen_next_q))
    if exp_words:
        overlap = len(exp_words & gen_words) / len(exp_words)
        if overlap < 0.2:
            score -= 2
            reasons.append("의미 이탈")
        elif overlap < 0.4:
            score -= 1
            reasons.append("의미 약간 이탈")

    # 부적절한 요청 체크 (사진, 이미지 등)
    inappropriate = ["사진", "보여주", "보내주", "이미지", "촬영"]
    if any(w in gen_next_q for w in inappropriate):
        score -= 2
        reasons.append("부적절 요청(사진 등)")

    # 금지어 체크
    forbidden = ["설문", "조사", "서베이"]
    if any(w in gen_next_q for w in forbidden):
        score -= 2
        reasons.append("금지어 포함")

    score = max(1, min(5, score))
    reason = ", ".join(reasons) if reasons else "양호"
    return score, reason


def main():
    # 샘플 로드
    samples_path = Path(__file__).parent.parent / "outputs" / "manual_eval_samples.json"
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # 원본 test 데이터 (원래 다음 질문 정보)
    test_path = Path(__file__).parent.parent / "data" / "processed" / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f if json.loads(l)["task_type"] == "response"]

    results = []
    tone_scores = []
    rephrase_scores = []

    print(f"{'='*80}")
    print("Phase 6 수동 평가 결과 (30건)")
    print(f"{'='*80}")

    for i, sample in enumerate(samples, 1):
        lines = sample["user_input"].split("\n")
        topic = lines[0].replace("주제: ", "") if len(lines) > 0 else ""
        user_answer = lines[2].replace("사용자 응답: ", "") if len(lines) > 2 else ""
        original_next_q = lines[3].replace("다음 질문: ", "") if len(lines) > 3 else ""

        try:
            gen = json.loads(sample["generated"])
            gen_comment = gen.get("comment", "")
            gen_next_q = gen.get("next_question", "")
        except json.JSONDecodeError:
            gen_comment = sample["generated"]
            gen_next_q = ""

        try:
            exp = json.loads(sample["expected"])
            exp_comment = exp.get("comment", "")
            exp_next_q = exp.get("next_question", "")
        except json.JSONDecodeError:
            exp_comment = sample["expected"]
            exp_next_q = ""

        # 평가
        tone_score, tone_reason = evaluate_tone(gen_comment, user_answer)
        rephrase_score, rephrase_reason = evaluate_rephrasing(
            gen_next_q, exp_next_q, original_next_q
        )

        tone_scores.append(tone_score)
        rephrase_scores.append(rephrase_score)

        result = {
            "index": sample["index"],
            "topic": topic,
            "user_answer": user_answer,
            "gen_comment": gen_comment,
            "gen_next_q": gen_next_q,
            "exp_comment": exp_comment,
            "exp_next_q": exp_next_q,
            "tone_score": tone_score,
            "tone_reason": tone_reason,
            "rephrase_score": rephrase_score,
            "rephrase_reason": rephrase_reason,
        }
        results.append(result)

        print(f"\n[{i:02d}] idx={sample['index']} | 주제: {topic}")
        print(f"  응답: {user_answer[:50]}{'...' if len(user_answer)>50 else ''}")
        print(f"  생성 코멘트: {gen_comment}")
        print(f"  생성 다음Q: {gen_next_q}")
        print(f"  톤: {tone_score}/5 ({tone_reason})")
        print(f"  리프레이징: {rephrase_score}/5 ({rephrase_reason})")

    # 집계
    avg_tone = sum(tone_scores) / len(tone_scores)
    avg_rephrase = sum(rephrase_scores) / len(rephrase_scores)

    tone_pass = sum(1 for s in tone_scores if s >= 4)
    rephrase_pass = sum(1 for s in rephrase_scores if s >= 4)

    tone_pass_rate = tone_pass / len(tone_scores) * 100
    rephrase_pass_rate = rephrase_pass / len(rephrase_scores) * 100

    print(f"\n{'='*80}")
    print("집계 결과")
    print(f"{'='*80}")
    print(f"\n톤 적절성:")
    print(f"  평균: {avg_tone:.2f}/5")
    print(f"  PASS (4+): {tone_pass}/{len(tone_scores)} ({tone_pass_rate:.1f}%)")
    print(f"  PASS 기준 >= 85%: {'PASS' if tone_pass_rate >= 85 else 'FAIL'}")
    print(f"  점수 분포: {dict(sorted(((s, tone_scores.count(s)) for s in set(tone_scores))))}")

    print(f"\n리프레이징 자연스러움:")
    print(f"  평균: {avg_rephrase:.2f}/5")
    print(f"  PASS (4+): {rephrase_pass}/{len(rephrase_scores)} ({rephrase_pass_rate:.1f}%)")
    print(f"  PASS 기준 >= 80%: {'PASS' if rephrase_pass_rate >= 80 else 'FAIL'}")
    print(f"  점수 분포: {dict(sorted(((s, rephrase_scores.count(s)) for s in set(rephrase_scores))))}")

    # 결과 저장
    output_dir = Path(__file__).parent.parent / "outputs"

    summary = {
        "tone": {
            "avg_score": round(avg_tone, 2),
            "pass_count": tone_pass,
            "total": len(tone_scores),
            "pass_rate": round(tone_pass_rate, 1),
            "pass_threshold": 85,
            "result": "PASS" if tone_pass_rate >= 85 else "FAIL",
        },
        "rephrasing": {
            "avg_score": round(avg_rephrase, 2),
            "pass_count": rephrase_pass,
            "total": len(rephrase_scores),
            "pass_rate": round(rephrase_pass_rate, 1),
            "pass_threshold": 80,
            "result": "PASS" if rephrase_pass_rate >= 80 else "FAIL",
        },
        "details": results,
    }

    with open(output_dir / "manual_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_dir}/manual_eval_results.json")


if __name__ == "__main__":
    main()
