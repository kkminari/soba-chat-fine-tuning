# 학습 데이터 중복 제거 방법론 가이드

**작성일**: 2026-04-23
**대상**: SOBA 챗봇 파인튜닝 데이터 (Qwen3-8B QLoRA, 한국어 SFT ~5K)
**목적**: train 3,970 / val 478 / test 533 데이터의 중복·누수·다양성 부족 문제 해결을 위한 표준 방법론과 SOBA 적용 전략 정리

---

## 1. 한 줄 결론

SOBA 5K SFT 데이터의 핵심 처방은 **"MinHash LSH + multilingual-e5 임베딩 dedup + StratifiedGroupKFold(base_seed_id)" 3단 콤보**이며, 정제 후 부족분은 Evol-Instruct + Persona 다변화로 채우면 LIMA 결과(1K로도 충분)에 비추어 충분히 고품질 모델 학습이 가능합니다.

가장 시급한 즉시 조치는 다음 2가지:
1. **base_seed_id 기반 split 재구성으로 누수 차단**
2. **intro/retry/ending 시작어구 rotation 도입**

---

## 2. LLM 학습 데이터 중복 제거 표준 방법론

### 2.1 Exact Deduplication (해시 기반)

**원리**: 문서/문장을 정규화(NFC + 공백·구두점 제거) 후 SHA-256 해시로 비교, 동일 해시 제거.

| 항목 | 내용 |
|------|------|
| 장점 | 매우 빠름 (O(N)), 구현 간단, false positive 0 |
| 단점 | 공백·구두점·어미 한 글자만 달라도 못 잡음 |
| 한국어 적합성 | 기본 정규화(NFC) + 공백·구두점 제거 필수 |
| **SOBA 추천도** | ★★★ (전처리 첫 단계로 무조건 적용) |

### 2.2 MinHash + LSH (Locality Sensitive Hashing)

**원리**: 문서를 n-gram(shingles) 집합으로 변환 → MinHash 시그니처 생성 → LSH 버킷팅으로 Jaccard 유사도 ≥ 임계치 후보만 비교. The Pile/RedPajama/Dolma의 표준 방식.

| 항목 | 내용 |
|------|------|
| 장점 | 수백만 문서 스케일 가능, 어순·어미 차이 견딤, 검증된 표준 |
| 단점 | 의미 동치(동의어 치환)는 못 잡음, n-gram 크기·임계치 튜닝 필요 |
| 한국어 적합성 | 어절/형태소 토큰화 권장 (음절 n-gram도 가능) |
| **SOBA 추천도** | ★★★ (5K 규모는 수초~수분 처리, 변형 3종 탐지에 최적) |

> **권장 파라미터**: 어절 5-gram shingle, num_perm=128, Jaccard 임계 **0.7**

### 2.3 SimHash (Charikar, 2002)

**원리**: 토큰 가중치 기반 비트벡터 해시 → 해밍 거리(보통 ≤3)로 유사 판정.

| 항목 | 내용 |
|------|------|
| 장점 | 매우 빠름, 메모리 효율적 |
| 단점 | 짧은 텍스트(1~3문장)에서 정확도 저하, 한국어 짧은 응답에 약함 |
| **SOBA 추천도** | ★ (intro/title 같은 짧은 텍스트엔 부적합) |

### 2.4 Embedding 기반 (Semantic Dedup)

**원리**: Sentence-BERT, multilingual-e5, BGE-M3 등으로 임베딩 → cosine 유사도 ≥ 임계치 제거. Meta의 **SemDeDup** (Abbas et al. 2023)이 대표.

| 항목 | 내용 |
|------|------|
| 장점 | 의미 동치(어휘 다른데 뜻 같음) 탐지, 어순 무관 |
| 단점 | 임베딩 비용, FAISS/ScaNN 인덱스 필요, false positive 가능 |
| 한국어 적합성 | multilingual-e5-large, BGE-M3, ko-sroberta-multitask 권장 |
| **SOBA 추천도** | ★★★ (변형 3종이 어순만 바꾼 경우 핵심 탐지기) |

> **권장**: cosine ≥ **0.92** (한국어 존댓말·어미 변화 false positive 회피)

### 2.5 Suffix Array / Substring Match

**원리**: 전체 코퍼스의 suffix array 구축 → 50자 이상 동일 substring 탐지·제거 (Lee et al. 2021의 ExactSubstr).

| 항목 | 내용 |
|------|------|
| 장점 | 사전학습 corpus의 boilerplate 제거에 강력 |
| 단점 | 구현 복잡, SFT 짧은 데이터엔 과함 |
| **SOBA 추천도** | ★ (사전학습용, SFT 5K엔 불필요) |

### 2.6 기법 비교표

| 기법 | 속도 | 어순 변화 | 의미 동치 | 5K 한국어 적합성 |
|------|------|-----------|-----------|------------------|
| Exact hash | ★★★ | X | X | ★★★ |
| MinHash+LSH | ★★★ | O | △ | ★★★ |
| SimHash | ★★★ | O | X | ★ |
| Embedding cosine | ★★ | O | O | ★★★ |
| Suffix array | ★★ | △ | X | ★ |

---

## 3. 주요 논문·사례

### 3.1 핵심 논문

| # | 저자/연도 | 제목 | URL |
|---|---|---|---|
| 1 | Lee et al. (2021, Google) | Deduplicating Training Data Makes Language Models Better | https://arxiv.org/abs/2107.06499 |
| 2 | Abbas et al. (2023, Meta FAIR) | SemDeDup: Data-efficient learning through semantic deduplication | https://arxiv.org/abs/2303.09540 |
| 3 | Tirumala et al. (2023, Meta) | D4: Document De-Duplication and Diversification | https://arxiv.org/abs/2308.12284 |
| 4 | Zhou et al. (2023) | LIMA: Less Is More for Alignment | https://arxiv.org/abs/2305.11206 |
| 5 | Chen et al. (2023) | AlpaGasus (Alpaca 52K → 9K 정선) | https://arxiv.org/abs/2307.08701 |
| 6 | Wang et al. (2022) | Self-Instruct | https://arxiv.org/abs/2212.10560 |
| 7 | Xu et al. (2023) | WizardLM / Evol-Instruct | https://arxiv.org/abs/2304.12244 |
| 8 | Penedo et al. (2024) | FineWeb | https://arxiv.org/abs/2406.17557 |
| 9 | Soldaini et al. (2024) | Dolma | https://arxiv.org/abs/2402.00159 |
| 10 | Tencent (2024) | PersonaHub | https://arxiv.org/abs/2406.20094 |

### 3.2 대형 코퍼스 큐레이션 파이프라인

| 데이터셋 | 중복 제거 방식 |
|----------|----------------|
| The Pile (EleutherAI) | MinHash LSH (Jaccard 0.87) |
| RedPajama-v2 | Exact + MinHash + Bloom filter, document-level |
| The Stack v2 | 코드 hash + near-dup MinHash |
| Dolma (AI2) | URL dedup + paragraph-level Bloom + MinHash |
| FineWeb-Edu | MinHash LSH (num_perm=112, threshold 0.7~0.8), 덤프별 |
| SlimPajama | Global MinHash dedup → RedPajama 627B의 50% 절감 |

### 3.3 SFT 데이터셋의 중복 처리

- **Alpaca-cleaned**: 원본 52K에서 잘못된 응답·중복 약 **9K 제거** (≈17%)
- **Dolly 15K**: 사람 작성 → 중복은 적으나 카테고리 균형 수동 검수
- **OpenAssistant**: tree-구조 중복 제거 + 품질 평점 필터
- **LIMA (Zhou et al. 2023)**: 1,000건 고품질만으로 SFT — **SOBA 5K 한국어에 가장 직접적 시사점**
- **Qwen, Llama-3 SFT**: 명시 공개는 제한적이나 "exact + n-gram + embedding" 3단계가 사실상 표준

---

## 4. Data Leakage 방지 (Split 누수)

### 4.1 SOBA의 핵심 위험

`alc_005`(base) → `alc_005_v1` → `alc_005_v1_v2_v3` 같은 **augmentation 계보**가 train/val/test에 흩어지면 모델이 base를 외운 뒤 val/test 변형을 "맞추는" 현상 발생. 현재 val 26%, test 32% 누수는 **검증 신뢰성을 사실상 무효화**.

### 4.2 표준 해법: Group-aware Splitting

```python
from sklearn.model_selection import StratifiedGroupKFold

# group_id = base_seed_id (예: alc_005_v1_v2 → alc_005)
splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, test_idx in splitter.split(X, task_types, groups=base_seed_ids):
    # task 비율 유지 + base_seed_id 누수 0 보장
    break
```

| 도구 | 용도 |
|------|------|
| `GroupShuffleSplit` | 단일 train/test 분할 |
| `GroupKFold` | k-fold 교차검증 |
| **`StratifiedGroupKFold`** | 그룹 + 클래스 비율 동시 보존 (SOBA의 6-task 균형에 최적) |

### 4.3 SOBA 권장 절차

1. `base_seed_id` 추출 함수 작성 (`alc_005_v1_v2_v3` → `alc_005`)
2. 동일 `base_seed_id`의 모든 변형(v1, v2, v3...)은 반드시 같은 split에 배치
3. task별 비율은 `StratifiedGroupKFold`로 보존
4. split 후 누수 검증: `assert len(set(train_groups) & set(val_groups)) == 0`

---

## 5. Diversity 향상 기법

### 5.1 시작어구 강제 다변화 (SOBA 즉시 적용)

- intro "안녕하세요" 99.8% 문제 → **rotation pool** 운영: `["안녕하세요", "반갑습니다", "환영합니다", "처음 뵙겠습니다", "좋은 하루입니다", 빈 시작, ...]` 가중 샘플링
- 생성 시 첫 토큰 banlist + first-token logit 페널티
- ending/retry도 동일 방식 적용

### 5.2 Self-Instruct / Evol-Instruct

- **Self-Instruct** (Wang 2022): 175 seed → LLM이 자기증식, ROUGE-L < 0.7 필터
- **Evol-Instruct** (WizardLM): "deepening / concretizing / increasing reasoning steps / adding constraints" 5종 진화 연산자
- **SOBA 적용**: 9조합만 순환 중인 variation_matrix를 Evol-Instruct식 진화 연산자로 확장

### 5.3 Persona-based Diversification

- **PersonaHub** (Tencent 2024): 10억 페르소나로 합성 데이터 다양화
- **SOBA 적용**: variation_matrix 48조합을 실제 활용 + 마케팅 리서치 페르소나(30대 직장인/40대 주부/Z세대 학생/50대 시니어 등) 명시

### 5.4 생성 다양성 파라미터

| 기법 | 효과 |
|------|------|
| Temperature 0.8~1.2 | 어휘 다양성 |
| Top-p 0.92~0.95 | 자연스러움 유지 |
| Diverse Beam Search (Vijayakumar 2018) | 그룹간 페널티로 표면 다양성 |
| Typical sampling | 정보량 평준화 |
| Frequency / presence penalty | 반복 어절 억제 |

### 5.5 Template-free 프롬프트

- "다음 형식으로 답해라: 안녕하세요, ..." 같은 **고정 템플릿 제거**
- 대신 "자연스러운 마케팅 리서처처럼 답하되 시작어구는 자유롭게"
- few-shot 예시도 시작어구를 의도적으로 분산

---

## 6. 한국어 특화 고려사항 (★ 주의 박스)

> **필수 주의 5가지**
>
> 1. **NFC 정규화 필수**: 한글은 NFC/NFD 분리로 동일 문자도 다른 바이트열이 됨 → exact dedup 누락 원인 1순위
> 2. **존댓말·어미 false positive**: "~합니다 / ~해요 / ~한다" 차이를 임베딩이 같다고 판단할 수 있음 → cosine 임계 **0.92 이상** 권장 (영어 통상 0.85보다 높게)
> 3. **형태소 vs 어절 vs 음절 n-gram**: MinHash shingle 단위 선택 중요
>    - 어절 n-gram(2~3): 빠르고 무난, **SOBA 권장**
>    - 형태소(KoNLPy/Mecab): 정확도 높으나 분석기 의존
>    - 음절 n-gram(4~6): 분석기 없이 강건, 짧은 문장에 효과적
> 4. **조사·어미 정규화 옵션**: dedup 전용 정규화 함수에서 종결어미 통일("했습니다"→"하다") 후 비교 → 표면 다양성은 보존하되 의미 중복만 제거
> 5. **짧은 텍스트 주의**: title(15자 이내), retry(평균 48자)는 SimHash·MinHash 모두 오탐 높음 → exact hash + 시작어구 분포 검증으로 보완

### 한국어 임베딩 모델 추천

| 모델 | 특징 | 추천 용도 |
|------|------|-----------|
| `intfloat/multilingual-e5-large` | 다국어 SOTA급, 한국어 우수 | **SOBA 1순위** |
| `BAAI/bge-m3` | 다국어, dense+sparse+multi-vec | 정밀 dedup |
| `jhgan/ko-sroberta-multitask` | 순수 한국어 SBERT | 빠른 베이스라인 |
| `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | KLUE 기반 | 학술 검증용 |

---

## 7. 품질 vs 양 Trade-off

### 7.1 5K SFT의 적정 다양성 임계치

- **LIMA (Meta 2023)**: 1,000건 고품질로 GPT-4와 견줄 alignment 달성 → "5K 한국어 SFT는 충분, 단 다양성·품질 확보 시"
- **AlpaGasus (Chen 2023)**: Alpaca 52K → ChatGPT 평가로 9K 정선 → 더 우수한 성능
- **결론**: SOBA 5K에서 dedup 후 **2~3K로 감소해도 충분**, 단 6 task × 페르소나 다양성 보장 시

### 7.2 부족 시 보강 전략

1. **Evol-Instruct 진화 생성**: seed 1개 → 5~10개 진화
2. **Back-translation**: 한→영→한으로 자연스러운 패러프레이즈
3. **PersonaHub 방식**: 페르소나 × 시나리오 cartesian
4. **GPT-4o/Claude 3.5 합성** + 인간 1차 검수 (50건 sampling)
5. **Curriculum 구성**: 쉬운 → 어려운 케이스 단계화

---

## 8. 실무 도구/라이브러리

| 도구 | 기법 | 설치 | 한국어 |
|------|------|------|--------|
| **datasketch** | MinHash, MinHashLSH, HyperLogLog | `pip install datasketch` | tokenizer 직접 |
| **text-dedup** (HF) | MinHash + Suffix Array + SimHash | `pip install text-dedup` | tokenizer 교체 |
| **SemDeDup** (FAIR) | embedding cluster dedup | GitHub: facebookresearch/SemDeDup | 임베딩만 교체 |
| **datatrove** (HF) | FineWeb 파이프라인 | `pip install datatrove` | 한국어 모듈 추가 필요 |
| **rensa** (Rust MinHash) | 초고속 MinHash | `pip install rensa` | tokenizer 직접 |
| **FAISS** | 임베딩 ANN 검색 | `pip install faiss-cpu` | 무관 |
| **scikit-learn** | GroupKFold, StratifiedGroupKFold | `pip install scikit-learn` | 무관 |
| **sentence-transformers** | 임베딩 추출 | `pip install sentence-transformers` | 무관 |
| **KoNLPy / Mecab-ko** | 한국어 형태소 분석 | `pip install konlpy` | 한국어 전용 |
| **soynlp** | 한국어 통계 토큰화 | `pip install soynlp` | 한국어 전용 |

---

## 9. SOBA 권장 적용 파이프라인

```
[원본 ~6,500건 (재생성 후)]
     │
     ▼
① NFC 정규화 + 공백·구두점 통일
     │
     ▼
② Exact hash dedup (sha256)
     │   ── 완전 동일 제거 (예상 영향 ~1%)
     ▼
③ MinHash LSH (어절 5-gram, num_perm=128, Jaccard 0.7)
     │   ── 변형 3종 어순 차이 탐지 (예상 영향 30~50%)
     ▼
④ Embedding semantic dedup
     │   ── multilingual-e5-large + FAISS, cosine ≥ 0.92
     │   ── task별로 따로 수행 (intro끼리, response끼리)
     │   ── (seed_id, question_index) 그룹 내 1개만 keep
     ▼
⑤ 시작어구 분포 검사 + rotation 검증
     │   ── 단일 시작어구 점유율 > 20%면 경고
     ▼
⑥ Group-aware split (StratifiedGroupKFold)
     │   ── group = base_seed_id, stratify = task_type
     │   ── 누수 0% 검증
     ▼
⑦ 부족분 보강 (필요 시)
     │   ── Evol-Instruct + PersonaHub + GPT-4o 합성
     │   ── 50건 인간 검수 sampling
     ▼
[정제 데이터셋: 약 2,500~3,500건 추정]
```

### 단계별 도구·임계치 권장

| 단계 | 도구 | 파라미터 |
|------|------|----------|
| ① 정규화 | `unicodedata.normalize('NFC', ...)` | 공백·이모지·구두점 통일 |
| ② Exact | Python `hashlib.sha256` | 정규화 후 해시 |
| ③ MinHash | `datasketch.MinHashLSH` | num_perm=128, threshold=0.7, 어절 5-gram |
| ④ Embedding | `sentence-transformers` + `faiss` | e5-large, cosine ≥ 0.92 |
| ⑤ 시작어구 | 자체 함수 | 첫 어절 분포, 점유율 캡 20% |
| ⑥ Split | `sklearn.StratifiedGroupKFold` | groups=base_seed_id |
| ⑦ 보강 | GPT-4o/Claude API + Evol-Instruct prompt | task별 부족분 채움 |

---

## 10. 검증 체크리스트 (정제 후 필수 확인)

- [ ] task별 시작어구 top-1 점유율 ≤ 20%
- [ ] (seed_id, question_index) 그룹당 평균 cosine 유사도 ≤ 0.85
- [ ] train ∩ val ∩ test의 base_seed_id 교집합 = ∅
- [ ] 6개 task 비율이 train/val/test에서 ±2% 이내 균형
- [ ] variation_matrix 48조합 중 실제 사용 조합 수 ≥ 30
- [ ] 정제 후 token 길이 분포가 원본과 유사 (편향 도입 X)

---

## 11. 참고문헌 (전체)

1. Lee et al. (2021), *Deduplicating Training Data Makes Language Models Better*, https://arxiv.org/abs/2107.06499
2. Abbas et al. (2023), *SemDeDup*, https://arxiv.org/abs/2303.09540
3. Tirumala et al. (2023), *D4*, https://arxiv.org/abs/2308.12284
4. Zhou et al. (2023), *LIMA*, https://arxiv.org/abs/2305.11206
5. Chen et al. (2023), *AlpaGasus*, https://arxiv.org/abs/2307.08701
6. Wang et al. (2022), *Self-Instruct*, https://arxiv.org/abs/2212.10560
7. Xu et al. (2023), *WizardLM / Evol-Instruct*, https://arxiv.org/abs/2304.12244
8. Penedo et al. (2024), *FineWeb*, https://arxiv.org/abs/2406.17557
9. Soldaini et al. (2024), *Dolma*, https://arxiv.org/abs/2402.00159
10. Tencent (2024), *PersonaHub*, https://arxiv.org/abs/2406.20094
11. Broder (1997), *On the resemblance and containment of documents* — MinHash 원본
12. Charikar (2002), *Similarity estimation techniques from rounding algorithms* — SimHash 원본
13. HuggingFace `text-dedup`: https://github.com/ChenghaoMou/text-dedup
14. Meta `SemDeDup`: https://github.com/facebookresearch/SemDeDup
15. HuggingFace `datatrove`: https://github.com/huggingface/datatrove
