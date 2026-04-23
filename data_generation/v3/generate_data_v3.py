"""
SOBA v3 학습 데이터 생성기

다축 설계 기반 (persona 10 × occasion 4 × variation_matrix 48 × starting_phrase rotation).

Usage:
    # Pilot 모드 (태스크별 30건 = 총 ~180건)
    python generate_data_v3.py --pilot 30

    # 본 생성 (약 6,000건)
    python generate_data_v3.py

    # 특정 태스크만
    python generate_data_v3.py --task response
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

# 경로
ROOT = Path(__file__).parent.parent.parent
V3_DIR = Path(__file__).parent
sys.path.insert(0, str(V3_DIR))

from prompts.response import (
    SYSTEM_PROMPT as RESPONSE_SYSTEM,
    USER_TEMPLATE as RESPONSE_USER,
    get_stage,
)
from prompts.intro import SYSTEM_PROMPT as INTRO_SYSTEM, USER_TEMPLATE as INTRO_USER
from prompts.first_question import (
    SYSTEM_PROMPT as FQ_SYSTEM,
    USER_TEMPLATE as FQ_USER,
)
from prompts.retry import (
    SYSTEM_PROMPT as RETRY_SYSTEM,
    USER_TEMPLATE as RETRY_USER,
    INVALID_TYPES,
)
from prompts.ending import SYSTEM_PROMPT as ENDING_SYSTEM, USER_TEMPLATE as ENDING_USER
from prompts.title import SYSTEM_PROMPT as TITLE_SYSTEM, USER_TEMPLATE as TITLE_USER

# ============================================================
# 설정
# ============================================================
MAX_CONCURRENT = 12
MAX_RETRIES = 3
RETRY_DELAY = 2.0
RANDOM_SEED = 42

# 생성 파라미터 풀 (High Priority #2)
TEMP_POOL = [0.8, 1.0, 1.2]
TOP_P_POOL = [0.92, 0.95]
FREQ_PEN_POOL = [0.3, 0.5]
PRES_PEN_POOL = [0.2, 0.4]

# 모델
MODEL_RESPONSE = "gpt-4o"
MODEL_OTHERS = "gpt-4o-mini"

# 태스크별 생성 목표 (본 생성 시)
TASK_TARGETS = {
    "intro": 400,           # 1 per seed
    "first_question": 400,  # 1 per seed
    "retry": 1200,          # seed × 3 invalid_types
    "ending": 400,          # 1 per seed
    "title": 400,           # 1 call per seed (2 records 출력)
    "response": 2800,       # seed × 7 q_idx transitions
}


# ============================================================
# 생성기
# ============================================================
class V3Generator:
    def __init__(self, pilot_per_task: Optional[int] = None, output_suffix: str = ""):
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.pilot_per_task = pilot_per_task

        random.seed(RANDOM_SEED)

        # 데이터 로드
        with open(V3_DIR / "seeds_v3.json", encoding="utf-8") as f:
            self.seeds = json.load(f)["seeds"]
        with open(V3_DIR / "personas.json", encoding="utf-8") as f:
            self.personas = {p["id"]: p for p in json.load(f)["personas"]}
        with open(V3_DIR / "occasions.json", encoding="utf-8") as f:
            self.occasions = {o["id"]: o for o in json.load(f)["occasions"]}
        with open(V3_DIR / "starting_phrases.json", encoding="utf-8") as f:
            self.starting_pools = json.load(f)["pools"]
        with open(V3_DIR.parent / "variation_matrix.json", encoding="utf-8") as f:
            self.vmatrix = json.load(f)

        # 출력 디렉터리
        suffix = f"_{output_suffix}" if output_suffix else ""
        self.out_dir = ROOT / "data" / "processed_v3"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.out_dir / f"raw_all{suffix}.jsonl"

        # 통계
        self.stats = {
            "total_calls": 0,
            "success": 0,
            "failures": 0,
            "total_samples": 0,
            "by_task": {},
        }

    # ------------------------------------------------------------
    # 공통 헬퍼
    # ------------------------------------------------------------
    def _persona_fmt(self, seed):
        p = self.personas[seed["persona_id"]]
        return {
            "persona_name": p["name"],
            "persona_vocab": p["vocab_hint"],
        }

    def _occasion_fmt(self, seed):
        o = self.occasions[seed["occasion_id"]]
        return {
            "occasion_name": o["name"],
            "occasion_context": o["context"],
        }

    def _pick_starting_phrase(self, task: str) -> str:
        pool = self.starting_pools.get(task, [""])
        return random.choice(pool)

    def _pick_response_variation(self) -> dict:
        """variation_matrix.json의 48조합 중 랜덤 선택"""
        rs = random.choice(self.vmatrix["response_styles"])
        cs_label = "(생성 시점 stage 적용)"
        sent = random.choice(self.vmatrix["user_sentiments"])
        return {
            "response_style": f"{rs['label']} — {rs['instruction']}",
            "sentiment": f"{sent['label']} — {sent['instruction']}",
            "combo_id": f"{rs['id']}_{sent['id']}",
        }

    def _rand_params(self) -> dict:
        """API 호출 파라미터 랜덤화"""
        return {
            "temperature": random.choice(TEMP_POOL),
            "top_p": random.choice(TOP_P_POOL),
            "frequency_penalty": random.choice(FREQ_PEN_POOL),
            "presence_penalty": random.choice(PRES_PEN_POOL),
        }

    async def _call_api(
        self, system: str, user: str, model: str
    ) -> Optional[list]:
        params = self._rand_params()
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
                        max_tokens=1500,
                        response_format={"type": "json_object"},
                        **params,
                    )
                    text = response.choices[0].message.content.strip()
                    parsed = json.loads(text)
                    # 리스트로 정규화
                    if isinstance(parsed, dict):
                        list_val = None
                        for v in parsed.values():
                            if isinstance(v, list):
                                list_val = v
                                break
                        parsed = list_val if list_val is not None else [parsed]
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    # dict 아닌 요소 자동 변환 (문자열이면 message/title로 매핑 시도)
                    normalized = []
                    for x in parsed:
                        if isinstance(x, dict):
                            normalized.append(x)
                        elif isinstance(x, str) and x.strip():
                            # 태스크별로 다른 키가 기대되므로 호출자가 처리할 수 있도록
                            # 범용 키 여러 개 삽입
                            normalized.append({
                                "message": x.strip(),
                                "retry_message": x.strip(),
                                "title": x.strip(),
                            })
                    return normalized
                except json.JSONDecodeError:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                    continue
                except Exception as e:
                    msg = str(e)[:100]
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        self.stats["failures"] += 1
                        print(f"  API fail: {msg}")
                    continue
            self.stats["failures"] += 1
            return None

    def _save(self, records: list[dict]):
        with open(self.out_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------
    # 태스크 1: intro
    # ------------------------------------------------------------
    async def gen_intro(self, seed: dict) -> list[dict]:
        user = INTRO_USER.format(
            topic=seed["topic"],
            target_audience=seed["target_audience"],
            starting_phrase=self._pick_starting_phrase("intro"),
            **self._persona_fmt(seed),
            **self._occasion_fmt(seed),
        )
        results = await self._call_api(INTRO_SYSTEM, user, MODEL_OTHERS)
        if not results:
            return []
        out = []
        for r in results:
            msg = r.get("message", "").strip()
            if msg:
                out.append({
                    "task_type": "intro",
                    "seed_id": seed["seed_id"],
                    "base_seed_id": seed["base_seed_id"],
                    "persona_id": seed["persona_id"],
                    "occasion_id": seed["occasion_id"],
                    "domain": seed["domain"],
                    "topic": seed["topic"],
                    "target_audience": seed["target_audience"],
                    "message": msg,
                })
        self.stats["success"] += 1
        self.stats["total_samples"] += len(out)
        return out

    # ------------------------------------------------------------
    # 태스크 2: first_question
    # ------------------------------------------------------------
    async def gen_first_question(self, seed: dict) -> list[dict]:
        first_q = seed["questions"][0]["content"]
        user = FQ_USER.format(
            topic=seed["topic"],
            first_question=first_q,
            starting_phrase=self._pick_starting_phrase("first_question"),
            **self._persona_fmt(seed),
            **self._occasion_fmt(seed),
        )
        results = await self._call_api(FQ_SYSTEM, user, MODEL_OTHERS)
        if not results:
            return []
        out = []
        for r in results:
            msg = r.get("message", "").strip()
            if msg:
                out.append({
                    "task_type": "first_question",
                    "seed_id": seed["seed_id"],
                    "base_seed_id": seed["base_seed_id"],
                    "persona_id": seed["persona_id"],
                    "occasion_id": seed["occasion_id"],
                    "domain": seed["domain"],
                    "topic": seed["topic"],
                    "first_question_original": first_q,
                    "message": msg,
                })
        self.stats["success"] += 1
        self.stats["total_samples"] += len(out)
        return out

    # ------------------------------------------------------------
    # 태스크 3: retry
    # ------------------------------------------------------------
    async def gen_retry(
        self, seed: dict, invalid_type: str
    ) -> list[dict]:
        q = random.choice(seed["questions"])
        persona = self.personas[seed["persona_id"]]
        user = RETRY_USER.format(
            topic=seed["topic"],
            current_question=q["content"],
            invalid_type=invalid_type,
            invalid_description=INVALID_TYPES[invalid_type],
            persona_name=persona["name"],
            starting_phrase=self._pick_starting_phrase("retry"),
        )
        results = await self._call_api(RETRY_SYSTEM, user, MODEL_OTHERS)
        if not results:
            return []
        out = []
        for r in results:
            msg = r.get("retry_message", "").strip()
            if msg:
                out.append({
                    "task_type": "retry",
                    "seed_id": seed["seed_id"],
                    "base_seed_id": seed["base_seed_id"],
                    "persona_id": seed["persona_id"],
                    "occasion_id": seed["occasion_id"],
                    "domain": seed["domain"],
                    "topic": seed["topic"],
                    "current_question": q["content"],
                    "invalid_type": invalid_type,
                    "retry_message": msg,
                })
        self.stats["success"] += 1
        self.stats["total_samples"] += len(out)
        return out

    # ------------------------------------------------------------
    # 태스크 4: ending
    # ------------------------------------------------------------
    async def gen_ending(self, seed: dict) -> list[dict]:
        user = ENDING_USER.format(
            topic=seed["topic"],
            starting_phrase=self._pick_starting_phrase("ending"),
            **self._persona_fmt(seed),
            **self._occasion_fmt(seed),
        )
        results = await self._call_api(ENDING_SYSTEM, user, MODEL_OTHERS)
        if not results:
            return []
        out = []
        for r in results:
            msg = r.get("message", "").strip()
            if msg:
                out.append({
                    "task_type": "ending",
                    "seed_id": seed["seed_id"],
                    "base_seed_id": seed["base_seed_id"],
                    "persona_id": seed["persona_id"],
                    "occasion_id": seed["occasion_id"],
                    "domain": seed["domain"],
                    "topic": seed["topic"],
                    "target_audience": seed["target_audience"],
                    "message": msg,
                })
        self.stats["success"] += 1
        self.stats["total_samples"] += len(out)
        return out

    # ------------------------------------------------------------
    # 태스크 5: title
    # ------------------------------------------------------------
    async def gen_title(self, seed: dict) -> list[dict]:
        user = TITLE_USER.format(
            seed_task_text=seed["seed_task_text"],
            domain=seed["domain"],
        )
        results = await self._call_api(TITLE_SYSTEM, user, MODEL_OTHERS)
        if not results:
            return []
        out = []
        for r in results:
            title = r.get("title", "").strip()
            if title and len(title) <= 15:
                out.append({
                    "task_type": "title",
                    "seed_id": seed["seed_id"],
                    "base_seed_id": seed["base_seed_id"],
                    "persona_id": seed["persona_id"],
                    "occasion_id": seed["occasion_id"],
                    "domain": seed["domain"],
                    "original_text": seed["seed_task_text"],
                    "style": r.get("style", ""),
                    "title": title,
                })
        self.stats["success"] += 1
        self.stats["total_samples"] += len(out)
        return out

    # ------------------------------------------------------------
    # 태스크 6: response (핵심)
    # ------------------------------------------------------------
    async def gen_response(self, seed: dict, q_idx: int) -> list[dict]:
        questions = seed["questions"]
        if q_idx >= len(questions) - 1:
            return []
        current_q = questions[q_idx]["content"]
        next_q = questions[q_idx + 1]["content"]

        variation = self._pick_response_variation()
        stage = get_stage(q_idx)
        persona = self.personas[seed["persona_id"]]
        occasion = self.occasions[seed["occasion_id"]]

        user = RESPONSE_USER.format(
            topic=seed["topic"],
            target_audience=seed["target_audience"],
            current_question=current_q,
            next_question=next_q,
            stage=stage,
            response_style=variation["response_style"],
            sentiment=variation["sentiment"],
            persona_name=persona["name"],
            persona_vocab=persona["vocab_hint"],
            occasion_name=occasion["name"],
            occasion_context=occasion["context"],
        )
        results = await self._call_api(RESPONSE_SYSTEM, user, MODEL_RESPONSE)
        if not results:
            return []
        out = []
        for r in results:
            if all(k in r for k in ("user_answer", "comment", "next_question")):
                out.append({
                    "task_type": "response",
                    "seed_id": seed["seed_id"],
                    "base_seed_id": seed["base_seed_id"],
                    "persona_id": seed["persona_id"],
                    "occasion_id": seed["occasion_id"],
                    "variation_combo": variation["combo_id"],
                    "domain": seed["domain"],
                    "topic": seed["topic"],
                    "question_index": q_idx,
                    "current_question": current_q,
                    "next_question_original": next_q,
                    "user_answer": r["user_answer"],
                    "comment": r["comment"],
                    "next_question_rephrased": r["next_question"],
                })
        self.stats["success"] += 1
        self.stats["total_samples"] += len(out)
        return out

    # ------------------------------------------------------------
    # 오케스트레이션
    # ------------------------------------------------------------
    def _seed_subset(self, n: Optional[int]) -> list[dict]:
        if n is None:
            return self.seeds
        # pilot: persona×occasion 조합 분포 유지하여 n개 추출
        by_combo = {}
        for s in self.seeds:
            key = (s["persona_id"], s["occasion_id"])
            by_combo.setdefault(key, []).append(s)
        picks = []
        per_combo = max(1, n // len(by_combo))
        for v in by_combo.values():
            picks.extend(v[:per_combo])
        random.shuffle(picks)
        return picks[:n]

    async def run(self, task_filter: Optional[str] = None):
        start = time.time()

        targets = TASK_TARGETS if self.pilot_per_task is None else {
            k: self.pilot_per_task for k in TASK_TARGETS
        }
        if task_filter:
            targets = {task_filter: targets[task_filter]}

        # 출력 초기화
        if self.out_path.exists():
            self.out_path.unlink()

        print("=" * 64)
        mode = f"PILOT {self.pilot_per_task}/task" if self.pilot_per_task else "FULL"
        print(f"SOBA v3 데이터 생성 — {mode}")
        print(f"Seeds: {len(self.seeds)}, 출력: {self.out_path}")
        print(f"Targets: {targets}")
        print("=" * 64)

        # Intro
        if "intro" in targets:
            n = targets["intro"]
            subset = self._seed_subset(n)
            print(f"\n[intro] 생성 시작 — 목표 {n}건")
            tasks = [self._gen_and_save("intro", self.gen_intro(s)) for s in subset]
            await asyncio.gather(*tasks)

        # First question
        if "first_question" in targets:
            n = targets["first_question"]
            subset = self._seed_subset(n)
            print(f"\n[first_question] 생성 시작 — 목표 {n}건")
            tasks = [
                self._gen_and_save("first_question", self.gen_first_question(s))
                for s in subset
            ]
            await asyncio.gather(*tasks)

        # Retry (seed × 3 invalid types rotation)
        if "retry" in targets:
            n = targets["retry"]
            # n = seed_count × 3; 반대로 seed_count = n / 3
            seed_count = max(1, n // 3)
            subset = self._seed_subset(seed_count)
            invalid_types_list = list(INVALID_TYPES.keys())
            print(
                f"\n[retry] 생성 시작 — 목표 {n}건 ({seed_count} seeds × 3 types)"
            )
            tasks = []
            for i, s in enumerate(subset):
                # Each seed gets 3 different invalid_types
                chosen = random.sample(invalid_types_list, k=min(3, len(invalid_types_list)))
                for it in chosen:
                    tasks.append(self._gen_and_save("retry", self.gen_retry(s, it)))
            await asyncio.gather(*tasks)

        # Ending
        if "ending" in targets:
            n = targets["ending"]
            subset = self._seed_subset(n)
            print(f"\n[ending] 생성 시작 — 목표 {n}건")
            tasks = [self._gen_and_save("ending", self.gen_ending(s)) for s in subset]
            await asyncio.gather(*tasks)

        # Title
        if "title" in targets:
            n = targets["title"]
            # n calls → ~2n records (keyword + summary)
            subset = self._seed_subset(n)
            print(f"\n[title] 생성 시작 — {n} calls (~{n*2} records)")
            tasks = [self._gen_and_save("title", self.gen_title(s)) for s in subset]
            await asyncio.gather(*tasks)

        # Response (핵심)
        if "response" in targets:
            n = targets["response"]
            # n = seed_count × 7 q_idx; 필요 seed = n/7
            seed_count = max(1, (n + 6) // 7)
            subset = self._seed_subset(seed_count)
            print(f"\n[response] 생성 시작 — 목표 {n}건 ({seed_count} seeds × 7 q_idx)")
            tasks = []
            for s in subset:
                for q_idx in range(7):
                    tasks.append(self._gen_and_save("response", self.gen_response(s, q_idx)))
            await asyncio.gather(*tasks)

        elapsed = time.time() - start
        print("\n" + "=" * 64)
        print(f"생성 완료")
        print(f"  API 호출: {self.stats['total_calls']}")
        print(f"  실패: {self.stats['failures']}")
        print(f"  총 샘플: {self.stats['total_samples']}")
        print(f"  소요: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        print(f"  저장: {self.out_path}")
        print("  태스크별:")
        for task, count in sorted(self.stats["by_task"].items()):
            print(f"    {task}: {count}")
        print("=" * 64)

    async def _gen_and_save(self, task: str, coro):
        """개별 생성 결과 저장 + 태스크별 카운트"""
        records = await coro
        if records:
            self._save(records)
            self.stats["by_task"][task] = self.stats["by_task"].get(task, 0) + len(records)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SOBA v3 데이터 생성기")
    parser.add_argument("--pilot", type=int, default=None, help="Pilot mode: N/task (e.g. 30)")
    parser.add_argument(
        "--task",
        choices=list(TASK_TARGETS.keys()),
        default=None,
        help="특정 태스크만 생성",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="출력 파일 suffix (raw_all_<suffix>.jsonl)",
    )
    args = parser.parse_args()

    gen = V3Generator(pilot_per_task=args.pilot, output_suffix=args.output_suffix)
    asyncio.run(gen.run(task_filter=args.task))


if __name__ == "__main__":
    main()
