"""
SOBA v3 품질 검증 스크립트 — 10개 정량 GO 기준 + human review

Usage:
    # dedup 후 본 검증
    python check_quality.py --input-dir data/processed_v3/ --output quality_report.json

    # human review 포함
    python check_quality.py --input-dir data/processed_v3/ \
        --human-review data/processed_v3/human_review.json \
        --output quality_report.json

    # pilot 검증 (누수/splitratio 느슨)
    python check_quality.py --input-dir data/processed_v3/pilot/ --pilot
"""

import argparse
import asyncio
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv


ROOT = Path(__file__).parent.parent.parent

# GO 기준
THRESHOLDS = {
    "starting_phrase_top1_pct": 0.20,       # ≤
    "first_3words_unique_pct": 0.50,        # ≥
    "leakage_base_seed_id": 0,              # =
    "avg_pairwise_cosine_per_seed": 0.85,   # ≤
    "variation_matrix_combos_used": 30,     # ≥
    "persona_occasion_combos_used": 35,     # ≥
    "persona_occasion_balance_std_ratio": 0.30,  # ≤
    "task_ratio_edge": 0.02,                # ≤ (편차)
    "v3_v2_avg_xsim": 0.80,                 # ≤
    "v3_v2_max_xsim": 0.95,                 # ≤
    "human_good_pct": 0.80,                 # ≥
    "human_bad_pct": 0.05,                  # ≤
}

# 태스크 목록
TASKS = ["intro", "first_question", "retry", "ending", "title", "response"]


def extract_text(item: dict) -> str:
    """품질 검증용 assistant 텍스트"""
    t = item["task_type"]
    if t == "response":
        return item.get("comment", "")  # comment가 보조 검증 핵심
    elif t in ("intro", "first_question", "ending"):
        return item.get("message", "")
    elif t == "retry":
        return item.get("retry_message", "")
    elif t == "title":
        return item.get("title", "")
    return ""


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def first_n_words(text: str, n: int) -> str:
    return " ".join(normalize(text).split()[:n])


# ============================================================
# 메트릭 계산
# ============================================================
def compute_starting_phrase_metrics(items: list[dict]) -> dict:
    """태스크별 시작어구(첫 어절) top-1 비율 + 첫 3어절 고유 비율"""
    result = {"top1_pct": {}, "first3_unique_pct": {}}
    by_task = defaultdict(list)
    for it in items:
        by_task[it["task_type"]].append(it)
    for task, task_items in by_task.items():
        firsts = [first_n_words(extract_text(it), 1) for it in task_items]
        threes = [first_n_words(extract_text(it), 3) for it in task_items]
        # 빈 문자열 제외
        firsts = [f for f in firsts if f]
        threes = [t for t in threes if t]
        if not firsts:
            continue
        top1_count = Counter(firsts).most_common(1)[0][1]
        result["top1_pct"][task] = top1_count / len(firsts)
        result["first3_unique_pct"][task] = len(set(threes)) / len(threes) if threes else 0
    return result


def compute_leakage(items_by_split: dict[str, list[dict]]) -> dict:
    """base_seed_id split 누수"""
    groups = {
        k: {it["base_seed_id"] for it in v} for k, v in items_by_split.items()
    }
    return {
        "train_val": len(groups.get("train", set()) & groups.get("val", set())),
        "train_test": len(groups.get("train", set()) & groups.get("test", set())),
        "val_test": len(groups.get("val", set()) & groups.get("test", set())),
        "group_counts": {k: len(v) for k, v in groups.items()},
    }


def compute_task_ratio(items_by_split: dict[str, list[dict]]) -> dict:
    """train/val/test의 task 비율 편차"""
    split_ratios = {}
    for split, items in items_by_split.items():
        if not items:
            split_ratios[split] = {}
            continue
        counts = Counter(it["task_type"] for it in items)
        total = sum(counts.values())
        split_ratios[split] = {t: counts.get(t, 0) / total for t in TASKS}
    # 편차
    if "train" in split_ratios and "val" in split_ratios and "test" in split_ratios:
        max_diff = 0.0
        for t in TASKS:
            vals = [split_ratios[s].get(t, 0) for s in ("train", "val", "test")]
            diff = max(vals) - min(vals)
            max_diff = max(max_diff, diff)
        split_ratios["max_edge"] = max_diff
    return split_ratios


def compute_combo_usage(items: list[dict]) -> dict:
    """variation_matrix 및 persona×occasion 조합 사용 수"""
    # response 태스크만 variation_combo 있음
    vm_combos = Counter()
    for it in items:
        if it["task_type"] == "response" and "variation_combo" in it:
            vm_combos[it["variation_combo"]] += 1

    po_combos = Counter(
        (it.get("persona_id"), it.get("occasion_id")) for it in items if it.get("persona_id")
    )

    po_counts = np.array(list(po_combos.values())) if po_combos else np.array([0])
    po_std_ratio = (
        float(po_counts.std() / po_counts.mean()) if po_counts.mean() > 0 else 0.0
    )

    return {
        "variation_matrix_combos_used": len(vm_combos),
        "variation_matrix_samples": sum(vm_combos.values()),
        "persona_occasion_combos_used": len(po_combos),
        "persona_occasion_std_ratio": po_std_ratio,
    }


async def get_embeddings(texts: list[str]) -> np.ndarray:
    """OpenAI 임베딩"""
    from openai import AsyncOpenAI

    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_vecs = []
    for i in range(0, len(texts), 100):
        chunk = texts[i : i + 100]
        resp = await client.embeddings.create(model="text-embedding-3-small", input=chunk)
        all_vecs.extend([d.embedding for d in resp.data])
    arr = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


async def compute_seed_avg_cosine(items: list[dict]) -> float:
    """같은 base_seed_id 내 샘플들의 평균 pairwise cosine"""
    by_seed = defaultdict(list)
    for it in items:
        by_seed[it["base_seed_id"]].append(it)

    # 샘플 수가 2+인 seed만
    eligible = [(sid, v) for sid, v in by_seed.items() if len(v) >= 2]
    if not eligible:
        return 0.0

    # 메모리 절약: seed 단위로 처리, 각 seed별 평균 → 전체 평균
    all_texts = []
    offsets = []  # (seed_idx, start_idx, count)
    cursor = 0
    for sid, v in eligible:
        offsets.append((sid, cursor, len(v)))
        for it in v:
            all_texts.append(extract_text(it))
        cursor += len(v)

    emb = await get_embeddings(all_texts)
    seed_means = []
    for sid, start, n in offsets:
        block = emb[start : start + n]
        sim = block @ block.T
        # diag 제외 평균
        iu = np.triu_indices(n, k=1)
        if len(iu[0]) == 0:
            continue
        seed_means.append(float(sim[iu].mean()))
    return float(np.mean(seed_means)) if seed_means else 0.0


# ============================================================
# 메인 검증
# ============================================================
async def check_quality(
    input_dir: Path,
    human_review_path: Optional[Path] = None,
    pilot: bool = False,
) -> dict:
    # Load train/val/test
    items_by_split = {}
    all_items = []
    for split in ("train", "val", "test"):
        path = input_dir / f"{split}.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = [json.loads(l) for l in f]
            items_by_split[split] = data
            all_items.extend(data)

    if not all_items:
        # pilot 모드: raw_all.jsonl 단일 파일 허용
        raw_path = input_dir / "raw_all.jsonl"
        if raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                all_items = [json.loads(l) for l in f]
            items_by_split = {"all": all_items}
        else:
            return {"error": "no input files found"}

    report = {
        "total": len(all_items),
        "by_task": dict(Counter(it["task_type"] for it in all_items)),
        "pilot_mode": pilot,
    }

    # 1. Starting phrase
    sp = compute_starting_phrase_metrics(all_items)
    report["starting_phrase_top1_pct"] = sp["top1_pct"]
    report["first_3words_unique_pct"] = sp["first3_unique_pct"]

    # 2. Leakage (pilot 스킵)
    if not pilot and len(items_by_split) == 3:
        report["leakage"] = compute_leakage(items_by_split)
        report["task_ratio"] = compute_task_ratio(items_by_split)

    # 3. Combo usage
    report["combo_usage"] = compute_combo_usage(all_items)

    # 4. Seed avg cosine (태스크 response 한정, cost 절감)
    response_items = [it for it in all_items if it["task_type"] == "response"]
    if response_items and len(response_items) >= 2:
        print("Computing seed avg cosine (response task) ...")
        report["avg_pairwise_cosine_per_seed"] = await compute_seed_avg_cosine(response_items)
    else:
        report["avg_pairwise_cosine_per_seed"] = None

    # 5. v3 vs v2 cross-sim (dedup_report.json에서 읽어옴)
    dedup_report_path = input_dir / "dedup_report.json"
    if dedup_report_path.exists():
        with open(dedup_report_path, encoding="utf-8") as f:
            dedup = json.load(f)
        for stg in dedup.get("stages", []):
            if stg.get("name") == "v3_v2_xsim":
                avg_xsim = stg.get("avg_xsim_by_task", {})
                max_xsim = stg.get("max_xsim_by_task", {})
                report["v3_v2_xsim"] = {
                    "avg_by_task": avg_xsim,
                    "max_by_task": max_xsim,
                    "overall_avg": float(np.mean(list(avg_xsim.values()))) if avg_xsim else None,
                    "overall_max": float(max(max_xsim.values())) if max_xsim else None,
                }
                break

    # 6. Human review
    if human_review_path and human_review_path.exists():
        with open(human_review_path, encoding="utf-8") as f:
            hr = json.load(f)
        report["human_review"] = hr

    # ============================================================
    # GO/FAIL 판정
    # ============================================================
    fail = []
    thresholds = {**THRESHOLDS}
    if pilot:
        # pilot 모드에서는 기준 느슨하게
        thresholds["starting_phrase_top1_pct"] = 0.35
        thresholds["first_3words_unique_pct"] = 0.30
        thresholds["variation_matrix_combos_used"] = 10
        thresholds["persona_occasion_combos_used"] = 20

    # Starting phrase
    for task, pct in report["starting_phrase_top1_pct"].items():
        if pct > thresholds["starting_phrase_top1_pct"]:
            fail.append(f"starting_phrase_top1[{task}]={pct:.2%} > {thresholds['starting_phrase_top1_pct']:.2%}")

    # First 3 words unique
    for task, pct in report["first_3words_unique_pct"].items():
        if pct < thresholds["first_3words_unique_pct"]:
            fail.append(f"first_3words_unique[{task}]={pct:.2%} < {thresholds['first_3words_unique_pct']:.2%}")

    # Leakage
    if "leakage" in report:
        lk = report["leakage"]
        if lk["train_val"] > 0 or lk["train_test"] > 0 or lk["val_test"] > 0:
            fail.append(f"leakage: t-v={lk['train_val']}, t-t={lk['train_test']}, v-t={lk['val_test']}")

    # Task ratio
    if "task_ratio" in report and "max_edge" in report["task_ratio"]:
        edge = report["task_ratio"]["max_edge"]
        if edge > thresholds["task_ratio_edge"]:
            fail.append(f"task_ratio_edge={edge:.3f} > {thresholds['task_ratio_edge']}")

    # Seed cosine
    if report.get("avg_pairwise_cosine_per_seed") is not None:
        cos = report["avg_pairwise_cosine_per_seed"]
        if cos > thresholds["avg_pairwise_cosine_per_seed"]:
            fail.append(f"avg_pairwise_cosine_per_seed={cos:.3f} > {thresholds['avg_pairwise_cosine_per_seed']}")

    # Combo usage
    cu = report["combo_usage"]
    if cu["variation_matrix_combos_used"] < thresholds["variation_matrix_combos_used"]:
        fail.append(f"variation_matrix_combos_used={cu['variation_matrix_combos_used']} < {thresholds['variation_matrix_combos_used']}")
    if cu["persona_occasion_combos_used"] < thresholds["persona_occasion_combos_used"]:
        fail.append(f"persona_occasion_combos_used={cu['persona_occasion_combos_used']} < {thresholds['persona_occasion_combos_used']}")
    if cu["persona_occasion_std_ratio"] > thresholds["persona_occasion_balance_std_ratio"]:
        fail.append(f"persona_occasion_std_ratio={cu['persona_occasion_std_ratio']:.3f} > {thresholds['persona_occasion_balance_std_ratio']}")

    # v3 vs v2
    if "v3_v2_xsim" in report and report["v3_v2_xsim"].get("overall_avg") is not None:
        avg = report["v3_v2_xsim"]["overall_avg"]
        max_v = report["v3_v2_xsim"]["overall_max"]
        if avg > thresholds["v3_v2_avg_xsim"]:
            fail.append(f"v3_v2_avg_xsim={avg:.3f} > {thresholds['v3_v2_avg_xsim']}")
        if max_v > thresholds["v3_v2_max_xsim"]:
            fail.append(f"v3_v2_max_xsim={max_v:.3f} > {thresholds['v3_v2_max_xsim']}")

    # Human review
    if "human_review" in report:
        hr = report["human_review"]
        for task in TASKS:
            if task in hr.get("good_pct", {}):
                if hr["good_pct"][task] < thresholds["human_good_pct"]:
                    fail.append(f"human_good[{task}]={hr['good_pct'][task]:.2%} < {thresholds['human_good_pct']:.2%}")
            if task in hr.get("bad_pct", {}):
                if hr["bad_pct"][task] > thresholds["human_bad_pct"]:
                    fail.append(f"human_bad[{task}]={hr['bad_pct'][task]:.2%} > {thresholds['human_bad_pct']:.2%}")

    report["go_status"] = "PASS" if not fail else "FAIL"
    report["fail_reasons"] = fail
    report["thresholds"] = thresholds
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--human-review", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pilot", action="store_true", help="Pilot 모드 (기준 느슨, leakage/split 스킵)")
    args = parser.parse_args()

    report = asyncio.run(
        check_quality(args.input_dir, args.human_review, args.pilot)
    )

    # 요약 출력
    print("\n" + "=" * 64)
    print(f"품질 검증 결과 — {report.get('go_status', 'N/A')}")
    print("=" * 64)
    print(f"총 샘플: {report.get('total')}")
    print(f"태스크별: {report.get('by_task')}")
    print()
    print("시작어구 top-1 점유:")
    for task, pct in sorted(report.get("starting_phrase_top1_pct", {}).items()):
        print(f"  {task:16s}: {pct:.2%}")
    print("\n첫 3어절 고유 비율:")
    for task, pct in sorted(report.get("first_3words_unique_pct", {}).items()):
        print(f"  {task:16s}: {pct:.2%}")
    cu = report.get("combo_usage", {})
    print(f"\nvariation_matrix combos: {cu.get('variation_matrix_combos_used')}")
    print(f"persona×occasion combos: {cu.get('persona_occasion_combos_used')} (std/mean={cu.get('persona_occasion_std_ratio', 0):.3f})")
    if "avg_pairwise_cosine_per_seed" in report:
        print(f"avg pairwise cosine/seed: {report['avg_pairwise_cosine_per_seed']}")
    if "leakage" in report:
        print(f"Leakage: {report['leakage']}")
    if "v3_v2_xsim" in report:
        print(f"v3 vs v2 xsim: avg={report['v3_v2_xsim'].get('overall_avg')}, max={report['v3_v2_xsim'].get('overall_max')}")
    if report.get("fail_reasons"):
        print("\n[FAIL REASONS]")
        for r in report["fail_reasons"]:
            print(f"  - {r}")

    # 저장
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
