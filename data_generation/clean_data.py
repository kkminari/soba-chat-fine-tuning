"""
학습 데이터 정제 스크립트

정제 항목:
1. response: 사진/이미지 요청 포함 건 삭제 (user의 다음질문 또는 assistant에 사진 관련 단어)
2. title: 금지어(설문/조사/서베이) → 제거
3. intro: 이모지 제거
4. first_question: 인사말("안녕하세요" 등) 접두사 제거
5. retry: 금지어/사진 포함 건 삭제

실행:
    python data_generation/clean_data.py
"""

import json
import re
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed_v2"

PHOTO_WORDS = ["사진", "보여주", "보내주", "이미지", "촬영"]
FORBIDDEN_WORDS = ["설문", "조사", "서베이"]
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F9FF\U0001FA00-\U0001FA6F"
    r"\u2600-\u26FF\u2700-\u27BF"
    r"\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\u200D\uFE0F]+",
)
GREETING_PATTERN = re.compile(r"^(안녕하세요[!.]?\s*[😊]?\s*)")


def has_photo_words(text: str) -> bool:
    return any(w in text for w in PHOTO_WORDS)


def has_forbidden_words(text: str) -> bool:
    return any(w in text for w in FORBIDDEN_WORDS)


def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text).strip()


def remove_greeting(text: str) -> str:
    """인사말 접두사 제거 후 첫 글자 정리"""
    cleaned = GREETING_PATTERN.sub("", text).strip()
    # 이모지 후 남은 공백 정리
    cleaned = remove_emoji(cleaned).strip()
    if not cleaned:
        return text  # 빈 문자열이 되면 원본 유지
    return cleaned


def clean_title(text: str) -> str:
    """title에서 금지어 제거"""
    for w in FORBIDDEN_WORDS:
        text = text.replace(w, "").strip()
    # 연속 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_item(item: dict) -> dict | None:
    """
    단일 아이템 정제. 삭제 대상이면 None 반환.
    """
    task = item["task_type"]
    messages = item["messages"]
    assistant_content = messages[2]["content"]

    if task == "response":
        user_content = messages[1]["content"]
        # user의 다음질문 또는 assistant에 사진 단어 → 삭제
        if has_photo_words(user_content) or has_photo_words(assistant_content):
            return None
        return item

    elif task == "title":
        if has_forbidden_words(assistant_content):
            cleaned = clean_title(assistant_content)
            if not cleaned or len(cleaned) < 2:
                return None  # 정제 후 너무 짧으면 삭제
            messages[2]["content"] = cleaned
        return item

    elif task == "intro":
        # 이모지 제거
        cleaned = remove_emoji(assistant_content)
        if cleaned:
            messages[2]["content"] = cleaned
        return item

    elif task == "first_question":
        # 인사말 제거
        cleaned = remove_greeting(assistant_content)
        messages[2]["content"] = cleaned
        return item

    elif task == "retry":
        # 금지어 또는 사진 포함 → 삭제
        if has_forbidden_words(assistant_content) or has_photo_words(assistant_content):
            return None
        return item

    elif task == "ending":
        # 이모지 제거 (혹시 있을 경우)
        cleaned = remove_emoji(assistant_content)
        if cleaned:
            messages[2]["content"] = cleaned
        return item

    return item


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_before": 0,
        "total_after": 0,
        "deleted": Counter(),
        "modified": Counter(),
    }

    for split in ["train", "val", "test"]:
        input_path = DATA_DIR / f"{split}.jsonl"
        output_path = OUTPUT_DIR / f"{split}.jsonl"

        with open(input_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]

        stats["total_before"] += len(items)
        cleaned = []

        for item in items:
            task = item["task_type"]
            original_assistant = item["messages"][2]["content"]

            result = clean_item(item)

            if result is None:
                stats["deleted"][task] += 1
            else:
                if result["messages"][2]["content"] != original_assistant:
                    stats["modified"][task] += 1
                cleaned.append(result)

        stats["total_after"] += len(cleaned)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in cleaned:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[{split}] {len(items)} → {len(cleaned)} (삭제 {len(items) - len(cleaned)}건)")

    print(f"\n{'='*60}")
    print("정제 결과 요약")
    print(f"{'='*60}")
    print(f"전체: {stats['total_before']} → {stats['total_after']} (삭제 {stats['total_before'] - stats['total_after']}건)")
    print(f"\n삭제 (태스크별):")
    for task, count in stats["deleted"].most_common():
        print(f"  {task}: {count}건")
    print(f"\n수정 (태스크별):")
    for task, count in stats["modified"].most_common():
        print(f"  {task}: {count}건")

    # 정제 후 검증
    print(f"\n{'='*60}")
    print("정제 후 검증")
    print(f"{'='*60}")

    for split in ["train", "val", "test"]:
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        with open(output_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]

        photo_count = 0
        forbidden_count = 0
        emoji_count = 0
        greeting_count = 0

        for item in items:
            task = item["task_type"]
            assistant = item["messages"][2]["content"]

            if task == "response" and has_photo_words(assistant):
                photo_count += 1
            if task == "title" and has_forbidden_words(assistant):
                forbidden_count += 1
            if task == "intro" and EMOJI_PATTERN.search(assistant):
                emoji_count += 1
            if task == "first_question" and assistant.startswith("안녕"):
                greeting_count += 1

        issues = photo_count + forbidden_count + emoji_count + greeting_count
        status = "CLEAN" if issues == 0 else f"ISSUES: {issues}"
        print(f"  [{split}] 사진:{photo_count} 금지어:{forbidden_count} 이모지:{emoji_count} 인사말:{greeting_count} → {status}")


if __name__ == "__main__":
    main()
