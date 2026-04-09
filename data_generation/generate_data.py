"""
SOBA 챗봇 파인튜닝 데이터 생성기

Usage:
    python generate_data.py --task response          # 핵심 태스크만
    python generate_data.py --task all_minor          # 보조 5개 태스크
    python generate_data.py --task all                # 전체
    python generate_data.py --task response --pilot 5 # 파일럿 (시드 5건)
"""

import asyncio
import json
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_GEN_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_GEN_DIR))

from prompts.response import (
    SYSTEM_PROMPT as RESPONSE_SYSTEM,
    USER_TEMPLATE as RESPONSE_USER,
    get_variation_set,
    get_stage,
)
from prompts.intro import SYSTEM_PROMPT as INTRO_SYSTEM, USER_TEMPLATE as INTRO_USER
from prompts.first_question import SYSTEM_PROMPT as FQ_SYSTEM, USER_TEMPLATE as FQ_USER
from prompts.retry import SYSTEM_PROMPT as RETRY_SYSTEM, USER_TEMPLATE as RETRY_USER
from prompts.ending import SYSTEM_PROMPT as ENDING_SYSTEM, USER_TEMPLATE as ENDING_USER
from prompts.title import SYSTEM_PROMPT as TITLE_SYSTEM, USER_TEMPLATE as TITLE_USER

# ============================================================
# 설정
# ============================================================

MAX_CONCURRENT = 10
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ============================================================
# 비동기 API 호출
# ============================================================

class DataGenerator:
    def __init__(self, pilot_count: Optional[int] = None):
        load_dotenv(DATA_GEN_DIR / ".env")
        load_dotenv(DATA_GEN_DIR.parent / ".env")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.pilot_count = pilot_count

        # 통계
        self.stats = {
            "total_calls": 0,
            "success": 0,
            "failures": 0,
            "total_samples": 0,
        }

        # 시드 데이터 로드
        with open(DATA_GEN_DIR / "topics.json", "r", encoding="utf-8") as f:
            self.topics_data = json.load(f)

        self.topics = self.topics_data["topics"]
        if self.pilot_count:
            self.topics = self.topics[:self.pilot_count]
            print(f"🧪 파일럿 모드: {self.pilot_count}건만 처리")

        # 출력 디렉토리
        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    async def _call_api(self, system: str, user: str, model: str = "gpt-4o") -> Optional[list]:
        """단일 API 호출 (재시도 포함)"""
        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    self.stats["total_calls"] += 1
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.8,
                        max_tokens=2000,
                        response_format={"type": "json_object"},
                    )

                    text = response.choices[0].message.content.strip()
                    parsed = json.loads(text)

                    # JSON 객체 안에 배열이 있는 경우 처리
                    if isinstance(parsed, dict):
                        for v in parsed.values():
                            if isinstance(v, list):
                                return v
                        return [parsed]
                    return parsed if isinstance(parsed, list) else [parsed]

                except json.JSONDecodeError:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                    continue
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        self.stats["failures"] += 1
                        print(f"  ❌ API 실패: {str(e)[:80]}")
                    continue

            self.stats["failures"] += 1
            return None

    # ============================================================
    # 태스크별 생성 함수
    # ============================================================

    async def generate_response(self, topic: dict, q_idx: int) -> list[dict]:
        """코멘트+리프레이징: 1개 질문에 대해 3변형 생성"""
        questions = topic["questions"]
        if q_idx >= len(questions) - 1:
            return []  # 마지막 질문은 다음 질문 없음

        current_q = questions[q_idx]["content"]
        next_q = questions[q_idx + 1]["content"]
        variations = get_variation_set(q_idx)
        stage = get_stage(q_idx)

        user_msg = RESPONSE_USER.format(
            topic=topic["topic"],
            target_audience=topic.get("target_audience", ""),
            current_question=current_q,
            next_question=next_q,
            stage=stage,
            variation_1=variations[0],
            variation_2=variations[1],
            variation_3=variations[2],
        )

        results = await self._call_api(RESPONSE_SYSTEM, user_msg, model="gpt-4o")
        if not results:
            return []

        samples = []
        for r in results:
            if all(k in r for k in ("user_answer", "comment", "next_question")):
                samples.append({
                    "task_type": "response",
                    "seed_id": topic["seed_id"],
                    "domain": topic["domain"],
                    "topic": topic["topic"],
                    "question_index": q_idx,
                    "current_question": current_q,
                    "next_question_original": next_q,
                    "user_answer": r["user_answer"],
                    "comment": r["comment"],
                    "next_question_rephrased": r["next_question"],
                })

        self.stats["success"] += 1
        self.stats["total_samples"] += len(samples)
        return samples

    async def generate_intro(self, topic: dict) -> list[dict]:
        """인트로 메시지 생성"""
        user_msg = INTRO_USER.format(
            topic=topic["topic"],
            target_audience=topic.get("target_audience", ""),
        )
        results = await self._call_api(INTRO_SYSTEM, user_msg, model="gpt-4o-mini")
        if not results:
            return []

        samples = []
        for r in results:
            msg = r.get("message", "")
            if msg:
                samples.append({
                    "task_type": "intro",
                    "seed_id": topic["seed_id"],
                    "domain": topic["domain"],
                    "topic": topic["topic"],
                    "tone": r.get("tone", ""),
                    "message": msg,
                })

        self.stats["success"] += 1
        self.stats["total_samples"] += len(samples)
        return samples

    async def generate_first_question(self, topic: dict) -> list[dict]:
        """첫 질문 전환 메시지 생성"""
        first_q = topic["questions"][0]["content"]
        user_msg = FQ_USER.format(
            topic=topic["topic"],
            first_question=first_q,
        )
        results = await self._call_api(FQ_SYSTEM, user_msg, model="gpt-4o-mini")
        if not results:
            return []

        samples = []
        for r in results:
            msg = r.get("message", "")
            if msg:
                samples.append({
                    "task_type": "first_question",
                    "seed_id": topic["seed_id"],
                    "domain": topic["domain"],
                    "topic": topic["topic"],
                    "first_question_original": first_q,
                    "style": r.get("style", ""),
                    "message": msg,
                })

        self.stats["success"] += 1
        self.stats["total_samples"] += len(samples)
        return samples

    async def generate_retry(self, topic: dict) -> list[dict]:
        """재시도 메시지 생성 (질문 1개 랜덤 선택)"""
        import random
        q = random.choice(topic["questions"])
        user_msg = RETRY_USER.format(
            topic=topic["topic"],
            current_question=q["content"],
        )
        results = await self._call_api(RETRY_SYSTEM, user_msg, model="gpt-4o-mini")
        if not results:
            return []

        samples = []
        for r in results:
            msg = r.get("retry_message", "")
            if msg:
                samples.append({
                    "task_type": "retry",
                    "seed_id": topic["seed_id"],
                    "domain": topic["domain"],
                    "topic": topic["topic"],
                    "current_question": q["content"],
                    "invalid_type": r.get("invalid_type", ""),
                    "retry_message": msg,
                })

        self.stats["success"] += 1
        self.stats["total_samples"] += len(samples)
        return samples

    async def generate_ending(self, topic: dict) -> list[dict]:
        """종료 메시지 생성"""
        user_msg = ENDING_USER.format(
            topic=topic["topic"],
            target_audience=topic.get("target_audience", ""),
        )
        results = await self._call_api(ENDING_SYSTEM, user_msg, model="gpt-4o-mini")
        if not results:
            return []

        samples = []
        for r in results:
            msg = r.get("message", "")
            if msg:
                samples.append({
                    "task_type": "ending",
                    "seed_id": topic["seed_id"],
                    "domain": topic["domain"],
                    "topic": topic["topic"],
                    "tone": r.get("tone", ""),
                    "message": msg,
                })

        self.stats["success"] += 1
        self.stats["total_samples"] += len(samples)
        return samples

    async def generate_title(self, topic: dict) -> list[dict]:
        """제목 생성"""
        user_msg = TITLE_USER.format(
            seed_task_text=topic.get("seed_task_text", topic["topic"]),
        )
        results = await self._call_api(TITLE_SYSTEM, user_msg, model="gpt-4o-mini")
        if not results:
            return []

        samples = []
        for r in results:
            title = r.get("title", "")
            if title:
                samples.append({
                    "task_type": "title",
                    "seed_id": topic["seed_id"],
                    "domain": topic["domain"],
                    "original_text": topic.get("seed_task_text", topic["topic"]),
                    "style": r.get("style", ""),
                    "title": title,
                })

        self.stats["success"] += 1
        self.stats["total_samples"] += len(samples)
        return samples

    # ============================================================
    # 실행 오케스트레이터
    # ============================================================

    def _save_results(self, task_type: str, results: list[dict]):
        """결과를 JSONL로 저장"""
        out_file = self.raw_dir / f"{task_type}.jsonl"
        with open(out_file, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    async def run_response_task(self):
        """핵심 태스크: 코멘트+리프레이징"""
        print("\n📝 코멘트+리프레이징 생성 시작")
        total = len(self.topics) * 7  # 질문 8개 중 7개 전환
        done = 0

        # 모든 (topic, q_idx) 쌍에 대해 태스크 생성
        async def process_one(topic, q_idx):
            nonlocal done
            results = await self.generate_response(topic, q_idx)
            if results:
                self._save_results("response", results)
            done += 1
            if done % 50 == 0 or done == total:
                pct = done / total * 100
                print(f"  진행: {done}/{total} ({pct:.0f}%) | 샘플: {self.stats['total_samples']}")

        tasks = []
        for topic in self.topics:
            for q_idx in range(7):  # Q0→Q1, Q1→Q2, ..., Q6→Q7
                tasks.append(process_one(topic, q_idx))

        await asyncio.gather(*tasks)

    async def run_minor_tasks(self):
        """보조 5개 태스크"""
        for task_name, gen_fn in [
            ("intro", self.generate_intro),
            ("first_question", self.generate_first_question),
            ("retry", self.generate_retry),
            ("ending", self.generate_ending),
            ("title", self.generate_title),
        ]:
            print(f"\n📝 {task_name} 생성 시작")

            async def process_one(topic, fn=gen_fn, name=task_name):
                results = await fn(topic)
                if results:
                    self._save_results(name, results)

            tasks = [process_one(t) for t in self.topics]
            await asyncio.gather(*tasks)
            print(f"  ✅ {task_name} 완료 | 누적 샘플: {self.stats['total_samples']}")

    async def run(self, task: str):
        """메인 실행"""
        start = time.time()

        # 기존 raw 파일 정리
        if task in ("response", "all"):
            (self.raw_dir / "response.jsonl").unlink(missing_ok=True)
        if task in ("all_minor", "all"):
            for name in ("intro", "first_question", "retry", "ending", "title"):
                (self.raw_dir / f"{name}.jsonl").unlink(missing_ok=True)

        print("=" * 60)
        print(f"🚀 SOBA 파인튜닝 데이터 생성")
        print(f"   태스크: {task}")
        print(f"   시드: {len(self.topics)}건")
        print(f"   동시 처리: {MAX_CONCURRENT}")
        print("=" * 60)

        if task in ("response", "all"):
            await self.run_response_task()

        if task in ("all_minor", "all"):
            await self.run_minor_tasks()

        elapsed = time.time() - start

        print("\n" + "=" * 60)
        print(f"✅ 생성 완료!")
        print(f"   API 호출: {self.stats['total_calls']}회")
        print(f"   성공: {self.stats['success']}회")
        print(f"   실패: {self.stats['failures']}회")
        print(f"   생성 샘플: {self.stats['total_samples']}건")
        print(f"   소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        print(f"   저장 위치: {self.raw_dir}/")
        print("=" * 60)

        # 파일별 건수 출력
        print("\n📊 파일별 건수:")
        for f in sorted(self.raw_dir.glob("*.jsonl")):
            count = sum(1 for _ in open(f, encoding="utf-8"))
            print(f"  {f.name}: {count}건")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SOBA 파인튜닝 데이터 생성")
    parser.add_argument(
        "--task",
        choices=["response", "all_minor", "all"],
        default="all",
        help="생성할 태스크 (response: 핵심만, all_minor: 보조만, all: 전체)",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="파일럿 모드: 지정한 수만큼만 시드 처리",
    )
    args = parser.parse_args()

    generator = DataGenerator(pilot_count=args.pilot)
    asyncio.run(generator.run(args.task))


if __name__ == "__main__":
    main()
