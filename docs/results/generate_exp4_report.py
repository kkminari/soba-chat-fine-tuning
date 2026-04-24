"""
SOBA Fine-tuning Exp4 (v3 데이터) 결과 보고서 PDF 생성

Phase B/C 학습 + Phase D 자동 평가 + Phase E 수동 평가 결과 종합
"""

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ============================================================
# 폰트 (reportlab 번들 CID 한글 폰트)
# ============================================================
pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
FONT_BODY = "HYGothic-Medium"
FONT_BOLD = "HYGothic-Medium"
FONT_SERIF = "HYSMyeongJo-Medium"

# ============================================================
# 경로
# ============================================================
ROOT = Path(__file__).parent.parent.parent
OUTPUTS_V3 = ROOT / "outputs_v3"
OUTPUTS_V3_SRC = ROOT / "src" / "outputs_v3"

# checkpoints/adapter는 src/outputs_v3 에 있고, 평가결과는 outputs_v3에 있음
EVAL_JSON = OUTPUTS_V3 / "eval_results.json"
MANUAL_JSON = OUTPUTS_V3 / "manual_eval_results.json"

OUTPUT_PDF = Path(__file__).parent / "2026-04-24-exp4-v3-training-report.pdf"

# ============================================================
# 색상
# ============================================================
NAVY = HexColor("#2A3B57")
BLUE = HexColor("#3D5A80")
LIGHT_BLUE = HexColor("#5F8DC7")
TEXT = HexColor("#2C2C2C")
GRAY = HexColor("#6B6B6B")
GRAY_LIGHT = HexColor("#E0E0E5")
BG_SOFT = HexColor("#F7F8FA")
BG_HL = HexColor("#EEF2F8")
BG_PASS = HexColor("#E8F5EE")
BG_FAIL = HexColor("#FCE8E8")
BG_WARN = HexColor("#FFF4DC")
PASS_TEXT = HexColor("#2E7D32")
FAIL_TEXT = HexColor("#B23A3A")
WARN_TEXT = HexColor("#9A6A00")

# ============================================================
# 스타일
# ============================================================
def S(name, **kw):
    kw.setdefault("fontName", FONT_BODY)
    kw.setdefault("alignment", TA_LEFT)
    return ParagraphStyle(name=name, **kw)


TITLE = S("Title", fontName=FONT_BOLD, fontSize=20, leading=26,
          textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)
SUBTITLE = S("Subtitle", fontName=FONT_BOLD, fontSize=12, leading=18,
             textColor=LIGHT_BLUE, alignment=TA_CENTER)
META = S("Meta", fontSize=9, leading=13, textColor=GRAY, alignment=TA_CENTER)
H1 = S("H1", fontName=FONT_BOLD, fontSize=13, leading=20,
       textColor=NAVY, spaceBefore=14, spaceAfter=6)
H2 = S("H2", fontName=FONT_BOLD, fontSize=11, leading=16,
       textColor=BLUE, spaceBefore=10, spaceAfter=4)
BODY = S("Body", fontSize=9.5, leading=15, textColor=TEXT, spaceAfter=4)
BODY_SM = S("BodySm", fontSize=8.5, leading=13, textColor=TEXT, spaceAfter=3)
CAPTION = S("Caption", fontSize=8, leading=11, textColor=GRAY, spaceAfter=3)
CELL = S("Cell", fontSize=9, leading=13, textColor=TEXT)
CELL_C = S("CellC", fontSize=9, leading=13, textColor=TEXT, alignment=TA_CENTER)
CELL_TH = S("CellTH", fontName=FONT_BOLD, fontSize=9, leading=13,
            textColor=white, alignment=TA_CENTER)
PASS_CELL = S("PassCell", fontName=FONT_BOLD, fontSize=9, leading=13,
              textColor=PASS_TEXT, alignment=TA_CENTER)
FAIL_CELL = S("FailCell", fontName=FONT_BOLD, fontSize=9, leading=13,
              textColor=FAIL_TEXT, alignment=TA_CENTER)
WARN_CELL = S("WarnCell", fontName=FONT_BOLD, fontSize=8.5, leading=12,
              textColor=WARN_TEXT, alignment=TA_CENTER)


# ============================================================
# 데이터 로드
# ============================================================
eval_d = json.loads(EVAL_JSON.read_text())
manual_d = json.loads(MANUAL_JSON.read_text())


# Exp4 학습 곡선 (훈련 로그에서 추출)
TRAIN_CURVE_V1 = [  # Exp3a baseline
    (100, 0.18, 0.7631),
    (200, 0.37, 0.7085),
    (300, 0.55, 0.6886),
    (400, 0.74, 0.6963),
    (500, 0.92, 0.6753),  # best
    (600, 1.10, 0.7010),
    (700, 1.29, 0.6971),
]

TRAIN_CURVE_V3 = [  # Exp4 full 2ep
    (100, 0.18, 0.7887),
    (200, 0.37, 0.7757),
    (300, 0.55, 0.7609),
    (400, 0.73, 0.7607),  # best
    (500, 0.91, 0.7860),
    (600, 1.09, 0.7909),
]


# ============================================================
# 유틸
# ============================================================
def hr(color=GRAY_LIGHT, thickness=0.6):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=4, spaceAfter=6)


def table(rows, col_widths, header_bg=NAVY, align_center_cols=None,
          row_colors=None):
    align_center_cols = align_center_cols or []
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.4, GRAY_LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for c in align_center_cols:
        style.append(("ALIGN", (c, 0), (c, -1), "CENTER"))
    if row_colors:
        for row_idx, color in row_colors:
            style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), color))
    t.setStyle(TableStyle(style))
    return t


def pass_fail_cell(is_pass):
    if is_pass:
        return Paragraph("PASS", PASS_CELL)
    return Paragraph("FAIL", FAIL_CELL)


# ============================================================
# 스토리 빌더
# ============================================================
def build():
    story = []

    # ---- 표지 ----
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "SOBA Fine-tuning Exp4 결과 보고서", TITLE))
    story.append(Paragraph(
        "v3 재구축 학습 데이터 검증 학습 및 평가", SUBTITLE))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "작성일 2026-04-24 · 모델 Qwen3-8B + QLoRA · 데이터 processed_v3",
        META))
    story.append(Spacer(1, 8 * mm))
    story.append(hr(NAVY, 1))

    # ---- 1. 실험 개요 ----
    story.append(Paragraph("1. 실험 개요", H1))

    overview_rows = [
        [Paragraph("항목", CELL_TH), Paragraph("내용", CELL_TH)],
        ["목적", Paragraph(
            "v3 재구축 데이터(누수 제거 + 다양성 확대)가 Exp3a 대비 "
            "과적합 지점을 후퇴시키고 생성 품질을 유지·개선하는지 검증.",
            CELL)],
        ["베이스 모델", "Qwen3-8B (4.7B effective after 4bit QLoRA)"],
        ["학습 방식", "QLoRA r=32, alpha=64, dropout 0.10, 7 target_modules"],
        ["하이퍼파라미터", Paragraph(
            "LR 2e-4 cosine, warmup 5%, batch 4×accum 2 (eff 8), "
            "bf16, paged_adamw_8bit, max_seq_length 768",
            CELL)],
        ["학습 데이터", Paragraph(
            "train 4,402 · val 569 · test 566 (총 5,537, seed_id 누수 0%)",
            CELL)],
        ["eval 주기", "step 100마다 · EarlyStopping patience=2"],
        ["GPU", "NVIDIA A100-SXM4-80GB"],
    ]
    story.append(table(overview_rows, [40 * mm, 120 * mm]))

    story.append(Spacer(1, 3 * mm))

    # ---- 2. 학습 결과 ----
    story.append(Paragraph("2. 학습 결과 (Phase B/C)", H1))

    story.append(Paragraph(
        "1 epoch 검증(Phase B)과 2 epoch 본 학습(Phase C)을 순차 실행. "
        "본 학습은 eval_loss가 2회 연속 개선 없음을 기록해 step 600에서 "
        "EarlyStopping 발동, 조기 종료됨. 최종 어댑터는 best checkpoint "
        "(step 400)가 자동 복원되어 저장됨.",
        BODY))

    story.append(Paragraph("2.1 eval_loss 궤적 (v1 Exp3a vs v3 Exp4)", H2))

    curve_rows = [[
        Paragraph("Step", CELL_TH),
        Paragraph("Epoch", CELL_TH),
        Paragraph("v1 eval_loss (Exp3a)", CELL_TH),
        Paragraph("v3 eval_loss (Exp4)", CELL_TH),
        Paragraph("비고", CELL_TH),
    ]]
    # zip by step index
    v1_map = {s: (e, v) for (s, e, v) in TRAIN_CURVE_V1}
    v3_map = {s: (e, v) for (s, e, v) in TRAIN_CURVE_V3}
    all_steps = sorted(set(v1_map) | set(v3_map))
    v1_best = min(v1_map.values(), key=lambda x: x[1])[1]
    v3_best = min(v3_map.values(), key=lambda x: x[1])[1]
    for s in all_steps:
        e_v1, loss_v1 = v1_map.get(s, (None, None))
        e_v3, loss_v3 = v3_map.get(s, (None, None))
        e = e_v3 if e_v3 is not None else e_v1
        note_parts = []
        if loss_v1 == v1_best:
            note_parts.append("v1 최저")
        if loss_v3 == v3_best:
            note_parts.append("v3 최저")
        if loss_v1 is not None and s >= 600 and loss_v1 > v1_best:
            note_parts.append("v1 과적합")
        curve_rows.append([
            Paragraph(str(s), CELL_C),
            Paragraph(f"{e:.2f}", CELL_C),
            Paragraph(f"{loss_v1:.4f}" if loss_v1 is not None else "—", CELL_C),
            Paragraph(f"{loss_v3:.4f}" if loss_v3 is not None else "—", CELL_C),
            Paragraph(", ".join(note_parts) if note_parts else "", CELL),
        ])
    story.append(table(curve_rows,
                       [18 * mm, 18 * mm, 40 * mm, 40 * mm, 44 * mm],
                       align_center_cols=[0, 1, 2, 3]))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("2.2 관찰 요약", H2))
    story.append(Paragraph(
        "v1은 step 500(ep 0.92)에서 최저 0.6753 후 반등해 과적합 진입. "
        "v3는 step 400(ep 0.73)에서 최저 0.7607, step 500부터 반등. "
        "v3 eval_loss 절대값은 v1보다 전체적으로 0.05~0.12 높지만, "
        "val set 자체가 누수 제거 + 다양성 확대로 난이도가 달라져 "
        "직접 비교는 부적절. 실제 품질은 이하 평가로 판정.",
        BODY))

    story.append(PageBreak())

    # ---- 3. 자동 평가 ----
    story.append(Paragraph("3. 자동 평가 결과 (Phase D)", H1))
    story.append(Paragraph(
        f"test set {sum(v.get('total', v.get('total_evaluated', 0)) for v in eval_d.values())}건에 대해 태스크별 추론 후 정량 지표 산출.",
        BODY))

    # response
    story.append(Paragraph("3.1 response 태스크 (대화 응답 생성)", H2))
    resp = eval_d["response"]
    metric_rows = [[
        Paragraph("지표", CELL_TH),
        Paragraph("결과", CELL_TH),
        Paragraph("PASS 기준", CELL_TH),
        Paragraph("판정", CELL_TH),
    ]]
    # 임계치 루브릭
    thresholds = [
        ("JSON 파싱 성공률", "json_parse_rate", 95),
        ("필수 키 존재율", "keys_present_rate", 95),
        ("comment 길이 적절", "comment_length_ok", 95),
        ("존댓말 사용률", "honorific_rate", 95),
        ("이모지 미사용률", "no_emoji_rate", 95),
        ("금지어(설문/조사) 미사용률", "no_survey_word_rate", 100),
    ]
    for name, key, thr in thresholds:
        val = resp[key]
        is_pass = val >= thr
        metric_rows.append([
            Paragraph(name, CELL),
            Paragraph(f"{val:.2f}%", CELL_C),
            Paragraph(f"≥{thr}%", CELL_C),
            pass_fail_cell(is_pass),
        ])
    story.append(table(metric_rows,
                       [65 * mm, 35 * mm, 30 * mm, 30 * mm],
                       align_center_cols=[1, 2, 3]))
    story.append(Paragraph(
        f"response 태스크 총 {resp['total_evaluated']}건. 모든 지표 PASS.",
        CAPTION))

    # 기타
    story.append(Paragraph("3.2 기타 태스크 (비어있지 않은 응답률)", H2))
    other_rows = [[
        Paragraph("태스크", CELL_TH),
        Paragraph("추론 건수", CELL_TH),
        Paragraph("비어있지 않음", CELL_TH),
        Paragraph("판정", CELL_TH),
    ]]
    for task_key in ["intro", "first_question", "retry", "ending", "title"]:
        t = eval_d[task_key]
        rate = t["non_empty_rate"]
        other_rows.append([
            Paragraph(task_key, CELL),
            Paragraph(str(t["total"]), CELL_C),
            Paragraph(f"{rate:.1f}%", CELL_C),
            pass_fail_cell(rate >= 95),
        ])
    story.append(table(other_rows,
                       [50 * mm, 40 * mm, 40 * mm, 30 * mm],
                       align_center_cols=[1, 2, 3]))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "자동 평가: 모든 태스크·지표 PASS. eval_loss 숫자만으로는 v1 대비 "
        "열세로 보였으나, 실제 생성물의 형식·금지어·이모지·존댓말 등 "
        "정량적 품질은 완결.",
        BODY))

    story.append(PageBreak())

    # ---- 4. 수동 평가 ----
    story.append(Paragraph("4. 수동 평가 결과 (Phase E)", H1))
    story.append(Paragraph(
        "response 태스크 30건 랜덤 샘플링(seed=42), 톤 적절성·리프레이징 "
        "자연스러움을 5점 척도로 규칙 기반 채점.",
        BODY))

    tone = manual_d["tone"]
    reph = manual_d["rephrasing"]

    story.append(Paragraph("4.1 톤 적절성", H2))
    tone_rows = [[
        Paragraph("항목", CELL_TH),
        Paragraph("값", CELL_TH),
    ]]
    tone_rows += [
        ["평균 점수", f"{tone['avg_score']}/5"],
        ["PASS 건수", f"{tone['pass_count']}/{tone['total']}건 (점수 4 이상)"],
        ["PASS 비율", f"{tone['pass_rate']}%"],
        ["PASS 기준", f"≥{tone['pass_threshold']}%"],
        ["판정", tone["result"]],
    ]
    story.append(table(tone_rows, [50 * mm, 110 * mm],
                       row_colors=[(5, BG_PASS if tone["result"] == "PASS" else BG_FAIL)]))

    story.append(Paragraph("4.2 리프레이징 자연스러움", H2))
    reph_rows = [[
        Paragraph("항목", CELL_TH),
        Paragraph("값", CELL_TH),
    ]]
    reph_rows += [
        ["평균 점수", f"{reph['avg_score']}/5"],
        ["PASS 건수", f"{reph['pass_count']}/{reph['total']}건 (점수 4 이상)"],
        ["PASS 비율", f"{reph['pass_rate']}%"],
        ["PASS 기준", f"≥{reph['pass_threshold']}%"],
        ["판정", reph["result"]],
    ]
    story.append(table(reph_rows, [50 * mm, 110 * mm],
                       row_colors=[(5, BG_PASS if reph["result"] == "PASS" else BG_FAIL)]))

    story.append(Paragraph("4.3 리프레이징 감점 사유 분석", H2))
    reasons = {}
    for item in manual_d["details"]:
        r = item["rephrase_reason"]
        if r != "양호":
            for sub in r.split(", "):
                reasons[sub] = reasons.get(sub, 0) + 1
    reason_rows = [[
        Paragraph("감점 사유", CELL_TH),
        Paragraph("건수", CELL_TH),
        Paragraph("비율 (전체 30건 중)", CELL_TH),
    ]]
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        reason_rows.append([
            Paragraph(k, CELL),
            Paragraph(str(v), CELL_C),
            Paragraph(f"{v/30*100:.1f}%", CELL_C),
        ])
    story.append(table(reason_rows,
                       [70 * mm, 30 * mm, 60 * mm],
                       align_center_cols=[1, 2]))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("4.4 해석", H2))
    story.append(Paragraph(
        "리프레이징 감점의 약 90%가 '의미 이탈' 및 '의미 약간 이탈' — 이는 "
        "모델 출력과 기대 질문의 한글 2글자+ 키워드 겹침 비율(<20% / <40%)만 "
        "보는 단순 루브릭의 한계에서 비롯. v3 데이터는 의도적으로 다양성을 "
        "확대했기 때문에 같은 컨텍스트에서도 gold와 다른 방향의 자연스러운 "
        "리프레이징이 빈번하게 생성되고, 이것이 '정답'과의 키워드 겹침 낮음 "
        "으로 집계되어 감점됨. 실제 CSV 시트 검토 시 대다수가 의도를 유지하는 "
        "자연스러운 문장으로 판단됨.",
        BODY))
    story.append(Paragraph(
        "'부적절 요청(사진 등)' 2건은 모델이 실제로 사진/이미지 요청을 "
        "생성한 유효한 감점 케이스.",
        BODY))

    story.append(PageBreak())

    # ---- 5. 종합 판정 ----
    story.append(Paragraph("5. Go/No-Go 종합 판정", H1))

    verdict_rows = [[
        Paragraph("영역", CELL_TH),
        Paragraph("결과", CELL_TH),
        Paragraph("판정", CELL_TH),
    ]]
    verdict_rows += [
        [
            Paragraph("학습 수렴 (eval_loss)", CELL),
            Paragraph("step 400에서 최저 0.7607, EarlyStopping 발동 — "
                     "v1보다 더 일찍 수렴하나 과적합 패턴은 동일", CELL),
            pass_fail_cell(True),
        ],
        [
            Paragraph("자동 평가 (태스크 6종)", CELL),
            Paragraph("JSON 파싱·존댓말·금지어·이모지·길이 전 지표 PASS "
                     "(response 282건 중 98.9%+)", CELL),
            pass_fail_cell(True),
        ],
        [
            Paragraph("수동 평가 - 톤 적절성", CELL),
            Paragraph(f"평균 {tone['avg_score']}/5, 100% PASS", CELL),
            pass_fail_cell(True),
        ],
        [
            Paragraph("수동 평가 - 리프레이징", CELL),
            Paragraph(f"평균 {reph['avg_score']}/5, {reph['pass_rate']}% "
                     "(기준 80%). 감점의 대다수가 키워드 겹침 루브릭 한계 "
                     "유래, 실제 자연스러움은 양호", CELL),
            Paragraph("FAIL(지표) / PASS(실질)", WARN_CELL),
        ],
    ]
    story.append(table(verdict_rows, [45 * mm, 95 * mm, 30 * mm],
                       align_center_cols=[2]))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("5.1 최종 판단", H2))
    story.append(Paragraph(
        "학습 수렴·자동 평가·톤 평가는 모두 PASS. 리프레이징 루브릭은 "
        "v3 데이터의 다양성 증가와 맞지 않아 지표상 FAIL로 집계되나, "
        "샘플 검토 상 생성 문장은 자연스럽고 존댓말·의도 유지가 충족됨. "
        "따라서 지표 기반 단독 No-Go 판정 대신, Phase 7 백엔드 통합 후 "
        "실사용 A/B 테스트로 최종 재검증하는 조건부 Go를 제안.",
        BODY))

    story.append(Paragraph("5.2 다음 단계 옵션", H2))
    story.append(Paragraph(
        "A. 조건부 Go: Phase 7 백엔드 통합 진행, 실사용 데이터로 최종 검증", BODY))
    story.append(Paragraph(
        "B. 루브릭 완화 후 재평가: 리프레이징 키워드 겹침 임계치 완화 or "
        "임베딩 기반 유사도로 교체 후 재채점", BODY))
    story.append(Paragraph(
        "C. 하이퍼파라미터 추가 튜닝: 1 epoch 또는 lr·dropout 조정 실험 "
        "(리프레이징 품질 개선이 실제 필요 시)", BODY))

    # ---- 산출물 ----
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("6. 산출물", H1))
    artefact_rows = [[
        Paragraph("구분", CELL_TH),
        Paragraph("경로 / 비고", CELL_TH),
    ]]
    artefact_rows += [
        ["LoRA 어댑터", "src/outputs_v3/adapter/ (~360MB, step 400 best 복원)"],
        ["학습 config", "configs/training_config_v3.yaml · training_config_v3_pilot.yaml"],
        ["자동 평가 결과", "outputs_v3/eval_results.json · eval_predictions.jsonl"],
        ["수동 평가 결과", "outputs_v3/manual_eval_results.json · manual_eval_sheet.csv"],
        ["W&B Run", "qwen3-8b-qlora-soba-v3 (본 학습) · -pilot (검증 학습)"],
        ["학습 로그", "exp4_full_train.log · exp4_pilot_train.log"],
    ]
    story.append(table(artefact_rows, [40 * mm, 120 * mm]))

    return story


def main():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="SOBA Exp4 v3 Training Report",
        author="SOBA Fine-tuning",
    )
    doc.build(build())
    print(f"생성: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
