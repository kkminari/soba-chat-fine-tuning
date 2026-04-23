"""
SOBA v3 데이터 정제·분할 파이프라인

단계:
1. NFC 정규화 + 공백·구두점 통일
2. sha256 exact dedup
3. 태스크별 차등 near-dup:
   - response/intro/first_question/ending: MinHash 어절 5-gram, Jaccard 0.7
   - retry: MinHash 어절 3-gram, Jaccard 0.75
   - title: Levenshtein ≤ 2 + exact hash
4. OpenAI text-embedding-3-small + cosine ≥ 0.92
5. v3 vs v2 cross-similarity (max > 0.95 제거)
6. StratifiedGroupKFold(groups=base_seed_id, stratify=task_type)

Usage:
    python dedup_pipeline.py --input data/processed_v3/raw_all.jsonl --output-dir data/processed_v3/
    python dedup_pipeline.py --input .../raw_all_pilot.jsonl --output-dir data/processed_v3/pilot/ --skip-v2-xsim
"""

import argparse
import asyncio
import hashlib
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
from datasketch import MinHash, MinHashLSH
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).parent.parent.parent

# ============================================================
# 유틸: 텍스트 추출 & 정규화
# ============================================================

# 태스크별 dedup 대상 텍스트 추출
def extract_text(item: dict) -> str:
    t = item["task_type"]
    if t == "response":
        return f"{item.get('comment','')}\n{item.get('next_question_rephrased','')}"
    elif t == "intro":
        return item.get("message", "")
    elif t == "first_question":
        return item.get("message", "")
    elif t == "retry":
        return item.get("retry_message", "")
    elif t == "ending":
        return item.get("message", "")
    elif t == "title":
        return item.get("title", "")
    return ""


_NORMALIZE_PUNCT = re.compile(r"[\s\.,\!\?\~\-·…『』「」\(\)\[\]{}'\"]+")


def normalize(text: str, for_hash: bool = False) -> str:
    """NFC 정규화 + 공백·구두점 통일"""
    t = unicodedata.normalize("NFC", text).strip()
    if for_hash:
        # 해시용: 공백·구두점 모두 제거
        t = _NORMALIZE_PUNCT.sub("", t).lower()
    else:
        # 일반 정규화: 공백 통일
        t = re.sub(r"\s+", " ", t)
    return t


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# 1단계: NFC + exact hash
# ============================================================
def stage_exact(items: list[dict]) -> tuple[list[dict], dict]:
    seen = set()
    kept = []
    removed = 0
    for it in items:
        text = extract_text(it)
        if not text.strip():
            removed += 1
            continue
        h = sha256_hash(normalize(text, for_hash=True))
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        it["_norm_text"] = normalize(text, for_hash=False)
        kept.append(it)
    return kept, {"stage": "exact", "removed": removed, "kept": len(kept)}


# ============================================================
# 2단계: MinHash LSH (긴 텍스트용)
# ============================================================
def shingle(text: str, n: int) -> set[str]:
    """어절 n-gram set"""
    tokens = text.split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def minhash_sig(shingles: set[str], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def stage_minhash(items: list[dict], ngram: int, threshold: float) -> tuple[list[dict], int]:
    """태스크 내부에서 MinHash LSH로 near-dup 제거"""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    sigs = {}
    kept_idx = []
    removed = 0

    for i, it in enumerate(items):
        text = it["_norm_text"]
        sh = shingle(text, ngram)
        if not sh:
            kept_idx.append(i)
            continue
        m = minhash_sig(sh)
        dup = lsh.query(m)
        if dup:
            removed += 1
            continue
        key = f"i{i}"
        lsh.insert(key, m)
        sigs[key] = i
        kept_idx.append(i)

    return [items[i] for i in kept_idx], removed


# ============================================================
# 3단계: Levenshtein (짧은 텍스트, title용)
# ============================================================
def stage_levenshtein(items: list[dict], max_dist: int = 2) -> tuple[list[dict], int]:
    kept = []
    removed = 0
    for it in items:
        text = it["_norm_text"]
        is_dup = False
        for k in kept:
            d = Levenshtein.distance(text, k["_norm_text"])
            if d <= max_dist:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(it)
    return kept, removed


# ============================================================
# 4단계: OpenAI 임베딩 기반 cosine dedup
# ============================================================
async def get_embeddings(texts: list[str], model: str = "text-embedding-3-small", batch: int = 100) -> np.ndarray:
    """OpenAI 임베딩 배치 조회. 반환: (N, D) float32"""
    from openai import AsyncOpenAI

    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_vecs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = await client.embeddings.create(model=model, input=chunk)
        all_vecs.extend([d.embedding for d in resp.data])
    arr = np.array(all_vecs, dtype=np.float32)
    # L2 정규화 (cosine = dot product)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def stage_embedding_dedup(
    items: list[dict], embeddings: np.ndarray, cos_threshold: float = 0.92
) -> tuple[list[dict], int]:
    """태스크 내 cosine ≥ threshold인 페어에서 하나 keep"""
    n = len(items)
    if n < 2:
        return items, 0
    keep_mask = np.ones(n, dtype=bool)
    # upper triangle 검사 (i < j)
    for i in range(n):
        if not keep_mask[i]:
            continue
        # i 기준으로 나머지와 cosine
        sims = embeddings[i] @ embeddings[i + 1 :].T
        dup_mask_tail = sims >= cos_threshold
        # 중복 표시 (i+1부터의 인덱스)
        for j_offset in np.where(dup_mask_tail)[0]:
            j = i + 1 + j_offset
            keep_mask[j] = False
    removed = int((~keep_mask).sum())
    return [it for it, m in zip(items, keep_mask) if m], removed


# ============================================================
# 5단계: v3 vs v2 cross-similarity
# ============================================================
def load_v2_texts_by_task(v2_dir: Path) -> dict[str, list[str]]:
    """v2 train/val/test 모두 로드, 태스크별 assistant content 추출"""
    out = defaultdict(list)
    for split in ("train", "val", "test"):
        path = v2_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                t = item["task_type"]
                content = item["messages"][2]["content"]
                if t == "response":
                    # v2 response의 assistant는 JSON 문자열
                    try:
                        parsed = json.loads(content)
                        text = f"{parsed.get('comment','')}\n{parsed.get('next_question','')}"
                    except json.JSONDecodeError:
                        text = content
                else:
                    text = content
                out[t].append(normalize(text, for_hash=False))
    return out


async def stage_v3_v2_xsim(
    v3_items: list[dict], v2_dir: Path, max_xsim: float = 0.95
) -> tuple[list[dict], dict]:
    """v3 vs v2 cross-similarity로 max > max_xsim 샘플 제거"""
    v2_by_task = load_v2_texts_by_task(v2_dir)
    # v2 임베딩 (태스크별)
    stats = {"removed": 0, "avg_xsim_by_task": {}, "max_xsim_by_task": {}}
    kept = []

    # 태스크별 분리 계산
    v3_by_task = defaultdict(list)
    for it in v3_items:
        v3_by_task[it["task_type"]].append(it)

    for task, v3_task_items in v3_by_task.items():
        if task not in v2_by_task or not v2_by_task[task]:
            kept.extend(v3_task_items)
            continue
        v2_texts = v2_by_task[task]
        v3_texts = [it["_norm_text"] for it in v3_task_items]

        v2_emb = await get_embeddings(v2_texts)
        v3_emb = await get_embeddings(v3_texts)

        # 전체 cosine 매트릭스
        sim = v3_emb @ v2_emb.T  # (n_v3, n_v2)
        max_per_v3 = sim.max(axis=1)

        stats["avg_xsim_by_task"][task] = float(max_per_v3.mean())
        stats["max_xsim_by_task"][task] = float(max_per_v3.max())

        keep_mask = max_per_v3 <= max_xsim
        stats["removed"] += int((~keep_mask).sum())

        for it, k in zip(v3_task_items, keep_mask):
            if k:
                kept.append(it)

    return kept, stats


# ============================================================
# 6단계: StratifiedGroupKFold split
# ============================================================
def stage_split(
    items: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """8:1:1 split with base_seed_id groups"""
    groups = [it["base_seed_id"] for it in items]
    tasks = [it["task_type"] for it in items]
    X = np.arange(len(items))

    # 10-fold으로 쪼갠 뒤 8/1/1로 재조합
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(X, tasks, groups=groups))
    # fold_indices[i] = (train_idx, test_idx_for_fold_i)
    # 간단하게: fold 0,1의 test를 각각 val, test로 쓰고 나머지 = train
    test_idx = set(fold_indices[0][1].tolist())
    val_idx = set(fold_indices[1][1].tolist())
    train_idx = [i for i in range(len(items)) if i not in test_idx and i not in val_idx]

    train = [items[i] for i in train_idx]
    val = [items[i] for i in sorted(val_idx)]
    test = [items[i] for i in sorted(test_idx)]

    # 누수 검증
    train_groups = {it["base_seed_id"] for it in train}
    val_groups = {it["base_seed_id"] for it in val}
    test_groups = {it["base_seed_id"] for it in test}
    leak_tv = len(train_groups & val_groups)
    leak_tt = len(train_groups & test_groups)
    leak_vt = len(val_groups & test_groups)

    info = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "leakage_train_val": leak_tv,
        "leakage_train_test": leak_tt,
        "leakage_val_test": leak_vt,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
    }
    return train, val, test, info


# ============================================================
# 메인
# ============================================================
async def run(
    input_path: Path,
    output_dir: Path,
    skip_v2_xsim: bool = False,
    cos_threshold: float = 0.92,
):
    report = {"stages": [], "input": str(input_path)}

    # Load
    with open(input_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f]
    report["stages"].append({"name": "load", "count": len(items)})
    print(f"[load] {len(items)} items")

    # Stage 1: NFC + exact
    items, info = stage_exact(items)
    report["stages"].append({"name": "exact_dedup", **info})
    print(f"[exact] removed={info['removed']}, kept={info['kept']}")

    # Stage 2-3: 태스크별 near-dup
    by_task = defaultdict(list)
    for it in items:
        by_task[it["task_type"]].append(it)

    MINHASH_CONFIG = {
        "response": {"ngram": 5, "threshold": 0.7},
        "intro": {"ngram": 5, "threshold": 0.7},
        "first_question": {"ngram": 5, "threshold": 0.7},
        "ending": {"ngram": 5, "threshold": 0.7},
        "retry": {"ngram": 3, "threshold": 0.75},
    }

    kept_all = []
    near_dup_report = {}
    for task, task_items in by_task.items():
        if task == "title":
            kept, rm = stage_levenshtein(task_items, max_dist=2)
            near_dup_report[task] = {"method": "levenshtein", "removed": rm, "kept": len(kept)}
        elif task in MINHASH_CONFIG:
            cfg = MINHASH_CONFIG[task]
            kept, rm = stage_minhash(task_items, cfg["ngram"], cfg["threshold"])
            near_dup_report[task] = {"method": "minhash", "ngram": cfg["ngram"], "removed": rm, "kept": len(kept)}
        else:
            kept = task_items
            near_dup_report[task] = {"method": "none", "removed": 0, "kept": len(kept)}
        kept_all.extend(kept)
        print(f"[near_dup][{task}] {near_dup_report[task]}")

    items = kept_all
    report["stages"].append({"name": "near_dup", **near_dup_report})

    # Stage 4: embedding dedup (per task)
    by_task = defaultdict(list)
    for it in items:
        by_task[it["task_type"]].append(it)

    embed_report = {}
    kept_all = []
    for task, task_items in by_task.items():
        if len(task_items) < 2:
            kept_all.extend(task_items)
            embed_report[task] = {"removed": 0, "kept": len(task_items)}
            continue
        texts = [it["_norm_text"] for it in task_items]
        print(f"[embed][{task}] embedding {len(texts)} texts ...")
        emb = await get_embeddings(texts)
        kept, rm = stage_embedding_dedup(task_items, emb, cos_threshold=cos_threshold)
        embed_report[task] = {"removed": rm, "kept": len(kept)}
        kept_all.extend(kept)
        print(f"[embed][{task}] removed={rm}, kept={len(kept)}")

    items = kept_all
    report["stages"].append({"name": "embedding_dedup", "per_task": embed_report})

    # Stage 5: v3 vs v2 cross-similarity
    if not skip_v2_xsim:
        v2_dir = ROOT / "data" / "processed_v2"
        print(f"[xsim] v3 vs v2 cross-similarity (v2_dir={v2_dir}) ...")
        items, xsim_info = await stage_v3_v2_xsim(items, v2_dir)
        report["stages"].append({"name": "v3_v2_xsim", **xsim_info})
        print(f"[xsim] removed={xsim_info['removed']}")
    else:
        print("[xsim] SKIPPED")

    # Stage 6: split
    # _norm_text 필드 제거 (최종 저장 전)
    for it in items:
        it.pop("_norm_text", None)

    train, val, test, split_info = stage_split(items)
    report["stages"].append({"name": "split", **split_info})
    print(f"[split] train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"        leakage: t↔v={split_info['leakage_train_val']}, t↔t={split_info['leakage_train_test']}, v↔t={split_info['leakage_val_test']}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for it in data:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  saved: {path} ({len(data)})")

    report["final"] = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "total": len(train) + len(val) + len(test),
    }
    with open(output_dir / "dedup_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\ndedup report: {output_dir / 'dedup_report.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-v2-xsim", action="store_true")
    parser.add_argument("--cos-threshold", type=float, default=0.92)
    args = parser.parse_args()

    asyncio.run(run(args.input, args.output_dir, args.skip_v2_xsim, args.cos_threshold))


if __name__ == "__main__":
    main()
