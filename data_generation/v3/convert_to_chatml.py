"""
v3 raw (task-field) → ChatML 변환기

data/processed_v3/{train,val,test}.jsonl 의 raw 형식을
처리 후 training에서 사용하는 ChatML 형식으로 변환.
원본은 {split}_raw.jsonl로 이름 변경 후 보존.
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "processed_v3"

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
    return _build_msg("response", user, assistant, r)


def convert_intro(r: dict) -> dict:
    user = f"주제: {r['topic']}\n타겟: {r.get('target_audience', '')}"
    return _build_msg("intro", user, r["message"], r)


def convert_first_question(r: dict) -> dict:
    user = f"주제: {r['topic']}\n첫 질문: {r.get('first_question_original', '')}"
    return _build_msg("first_question", user, r["message"], r)


def convert_retry(r: dict) -> dict:
    user = (
        f"주제: {r['topic']}\n"
        f"현재 질문: {r.get('current_question', '')}\n"
        f"부적절 유형: {r.get('invalid_type', '')}"
    )
    return _build_msg("retry", user, r["retry_message"], r)


def convert_ending(r: dict) -> dict:
    user = f"주제: {r['topic']}\n타겟: {r.get('target_audience', '')}"
    return _build_msg("ending", user, r["message"], r)


def convert_title(r: dict) -> dict:
    user = r.get("original_text", "")
    return _build_msg("title", user, r["title"], r)


def _build_msg(task: str, user: str, assistant: str, r: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[task]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task_type": task,
        "seed_id": r.get("seed_id", ""),
        "base_seed_id": r.get("base_seed_id", ""),
        "persona_id": r.get("persona_id", ""),
        "occasion_id": r.get("occasion_id", ""),
    }


CONVERTERS = {
    "response": convert_response,
    "intro": convert_intro,
    "first_question": convert_first_question,
    "retry": convert_retry,
    "ending": convert_ending,
    "title": convert_title,
}


def convert_split(split: str):
    in_path = DATA_DIR / f"{split}.jsonl"
    raw_backup = DATA_DIR / f"{split}_raw.jsonl"

    with open(in_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f]

    # 원본 보존
    with open(raw_backup, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ChatML 변환
    converted = []
    errors = 0
    task_counts = Counter()
    for r in items:
        task = r.get("task_type", "")
        converter = CONVERTERS.get(task)
        if not converter:
            errors += 1
            continue
        try:
            chatml = converter(r)
            if not chatml["messages"][2]["content"].strip():
                errors += 1
                continue
            converted.append(chatml)
            task_counts[task] += 1
        except (KeyError, TypeError) as e:
            errors += 1

    # ChatML 저장 (train.jsonl 덮어쓰기)
    with open(in_path, "w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[{split}] {len(items)} raw → {len(converted)} chatml (errors={errors})")
    print(f"  task: {dict(task_counts)}")


def main():
    print(f"Input dir: {DATA_DIR}")
    for split in ("train", "val", "test"):
        convert_split(split)
    print(f"\n원본 raw은 {{split}}_raw.jsonl로 백업됨")
    print(f"ChatML은 {{split}}.jsonl로 저장됨 (기존 raw 덮어씀)")


if __name__ == "__main__":
    main()
