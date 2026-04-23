"""
seeds_v3.json 생성기

178개 고유 base topic에 10 persona × 4 occasion = 40 combo를 할당하여
총 400개 시드 생성. 각 combo는 10개 seed (10개 서로 다른 base_topic 커버).
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
TOPICS_PATH = Path(__file__).parent.parent / "topics.json"
OUT_PATH = ROOT / "seeds_v3.json"

TARGET_SEEDS = 400
SEEDS_PER_COMBO = 10  # 40 combos × 10 = 400
SEED_RANDOM = 42


def base_seed_id(seed_id: str) -> str:
    """alc_005_v1_v2_v3 -> alc_005"""
    return re.sub(r"(_v\d+)+$", "", seed_id)


def main():
    random.seed(SEED_RANDOM)

    # Load sources
    with open(TOPICS_PATH, encoding="utf-8") as f:
        topics_data = json.load(f)
    topics = topics_data["topics"]

    with open(ROOT / "personas.json", encoding="utf-8") as f:
        personas = json.load(f)["personas"]
    with open(ROOT / "occasions.json", encoding="utf-8") as f:
        occasions = json.load(f)["occasions"]

    # Deduplicate by topic string: keep shortest seed_id as canonical
    topic_to_canonical = {}
    for t in topics:
        key = t["topic"]
        if key not in topic_to_canonical or len(t["seed_id"]) < len(
            topic_to_canonical[key]["seed_id"]
        ):
            topic_to_canonical[key] = t

    base_topics = list(topic_to_canonical.values())
    print(f"Unique base topics: {len(base_topics)}")

    # Build 40 combos
    combos = [(p["id"], o["id"]) for p in personas for o in occasions]
    assert len(combos) == 40, f"Expected 40 combos, got {len(combos)}"

    # Index personas/occasions for name lookup
    p_by_id = {p["id"]: p for p in personas}
    o_by_id = {o["id"]: o for o in occasions}

    # Shuffle base topics for random-but-reproducible assignment
    shuffled = base_topics.copy()
    random.shuffle(shuffled)

    seeds_v3 = []
    combo_topic_pairs = set()  # avoid (combo, topic) duplicates

    # Round-robin: each combo gets SEEDS_PER_COMBO base topics,
    # advancing through shuffled list
    # Ensure each combo's 10 topics are distinct
    for combo_idx, (pid, oid) in enumerate(combos):
        assigned_topics = []
        offset = combo_idx  # start offset by combo to diversify topic access
        attempts = 0
        while len(assigned_topics) < SEEDS_PER_COMBO and attempts < 1000:
            t = shuffled[(offset + len(assigned_topics) * 17 + attempts) % len(shuffled)]
            # Use stride 17 (coprime with 178) to spread picks
            topic_key = t["topic"]
            if (combo_idx, topic_key) not in combo_topic_pairs:
                combo_topic_pairs.add((combo_idx, topic_key))
                assigned_topics.append(t)
            attempts += 1

        # Create seeds for this combo
        persona = p_by_id[pid]
        occasion = o_by_id[oid]
        for i, t in enumerate(assigned_topics):
            base_id = base_seed_id(t["seed_id"])
            domain_prefix = t["seed_id"].split("_")[0]  # alc or cos
            new_seed_id = f"{domain_prefix}_{pid}{oid}_{len(seeds_v3):04d}"

            seed = {
                "seed_id": new_seed_id,
                "base_seed_id": base_id,
                "persona_id": pid,
                "occasion_id": oid,
                "topic": t["topic"],
                "target_audience": (
                    f"{persona['name']} / {occasion['name']}"
                ),
                "domain": t["domain"],
                "seed_task_text": t["seed_task_text"],
                "questions": t["questions"],
            }
            seeds_v3.append(seed)

    # Validation
    combo_counts = Counter((s["persona_id"], s["occasion_id"]) for s in seeds_v3)
    base_counts = Counter(s["base_seed_id"] for s in seeds_v3)
    base_ids = set(s["base_seed_id"] for s in seeds_v3)

    print(f"\nGenerated seeds: {len(seeds_v3)}")
    print(f"Unique base_seed_id: {len(base_ids)}")
    print(f"Combos covered: {len(combo_counts)} / 40")
    print(f"Combo size min/max/mean: {min(combo_counts.values())} / {max(combo_counts.values())} / {sum(combo_counts.values()) / len(combo_counts):.1f}")
    print(f"Base topic usage min/max/mean: {min(base_counts.values())} / {max(base_counts.values())} / {sum(base_counts.values()) / len(base_counts):.2f}")
    print(f"Domain split: {Counter(s['domain'] for s in seeds_v3)}")

    # Write output
    output = {
        "description": "SOBA v3 학습 데이터 생성용 시드. 178 base topic × 40 (persona×occasion) 조합으로 400 시드 분배.",
        "total": len(seeds_v3),
        "base_topics": len(base_ids),
        "combos": 40,
        "seeds_per_combo": SEEDS_PER_COMBO,
        "random_seed": SEED_RANDOM,
        "seeds": seeds_v3,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
