"""
raw JSONL → ChatML 포맷 변환기

Usage:
    python convert_to_chatml.py
"""

import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ============================================================
# 태스크별 ChatML 변환 규칙
# ============================================================

SYSTEM_PROMPTS = {
    "response": (
        "당신은 친근한 데이터 수집 챗봇입니다. "
        "사용자 응답에 공감 코멘트를 달고, 다음 질문으로 자연스럽게 전환하세요.\n\n"
        '출력 형식: {"comment": "...", "next_question": "..."}'
    ),
    "intro": (
        "당신은 친근한 데이터 수집 챗봇입니다. "
        "대화 주제를 소개하는 따뜻한 인트로 메시지를 생성하세요."
    ),
    "first_question": (
        "당신은 친근한 데이터 수집 챗봇입니다. "
        "첫 번째 질문으로 자연스럽게 전환하세요."
    ),
    "retry": (
        "당신은 친근한 데이터 수집 챗봇입니다. "
        "불성실한 응답에 부드럽게 재응답을 요청하세요."
    ),
    "ending": (
        "당신은 친근한 데이터 수집 챗봇입니다. "
        "대화를 마무리하는 감사 메시지를 생성하세요."
    ),
    "title": (
        "당신은 마케팅 리서치 제목 생성기입니다. "
        "15자 이내로 짧은 제목을 만드세요."
    ),
}


def convert_response(r: dict) -> dict:
    user = (
        f"주제: {r['topic']}\n"
        f"현재 질문: {r['current_question']}\n"
        f"사용자 응답: {r['user_answer']}\n"
        f"다음 질문: {r['next_question_original']}"
    )
    assistant = json.dumps(
        {"comment": r["comment"], "next_question": r["next_question_rephrased"]},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["response"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task_type": "response",
        "seed_id": r.get("seed_id", ""),
    }


def convert_intro(r: dict) -> dict:
    user = f"주제: {r['topic']}\n타겟: {r.get('target_audience', '')}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["intro"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["message"]},
        ],
        "task_type": "intro",
        "seed_id": r.get("seed_id", ""),
    }


def convert_first_question(r: dict) -> dict:
    user = f"주제: {r['topic']}\n첫 질문: {r.get('first_question_original', '')}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["first_question"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["message"]},
        ],
        "task_type": "first_question",
        "seed_id": r.get("seed_id", ""),
    }


def convert_retry(r: dict) -> dict:
    user = (
        f"주제: {r['topic']}\n"
        f"현재 질문: {r.get('current_question', '')}\n"
        f"부적절 유형: {r.get('invalid_type', '')}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["retry"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["retry_message"]},
        ],
        "task_type": "retry",
        "seed_id": r.get("seed_id", ""),
    }


def convert_ending(r: dict) -> dict:
    user = f"주제: {r['topic']}\n타겟: {r.get('target_audience', '')}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["ending"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["message"]},
        ],
        "task_type": "ending",
        "seed_id": r.get("seed_id", ""),
    }


def convert_title(r: dict) -> dict:
    user = r.get("original_text", "")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["title"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["title"]},
        ],
        "task_type": "title",
        "seed_id": r.get("seed_id", ""),
    }


CONVERTERS = {
    "response": convert_response,
    "intro": convert_intro,
    "first_question": convert_first_question,
    "retry": convert_retry,
    "ending": convert_ending,
    "title": convert_title,
}


def main():
    # 로드
    with open(RAW_DIR / "all_merged.jsonl", encoding="utf-8") as f:
        raw_data = [json.loads(l) for l in f]

    print(f"입력: {len(raw_data)}건")

    # 변환
    converted = []
    errors = 0
    for r in raw_data:
        task = r.get("task_type", "")
        converter = CONVERTERS.get(task)
        if not converter:
            errors += 1
            continue
        try:
            chatml = converter(r)
            # 검증: assistant 내용이 비어있지 않은지
            if not chatml["messages"][2]["content"].strip():
                errors += 1
                continue
            converted.append(chatml)
        except (KeyError, TypeError):
            errors += 1

    # 저장
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "chatml.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 결과
    task_counts = Counter(c["task_type"] for c in converted)
    print(f"변환 성공: {len(converted)}건")
    print(f"변환 실패: {errors}건")
    print(f"저장: {out_path}")
    print(f"\n태스크별:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task:20s}: {count}건")


if __name__ == "__main__":
    main()
