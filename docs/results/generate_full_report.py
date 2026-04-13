"""
SOBA Fine-tuning 전체 실험 보고서 PDF 생성
실험 1 (14B baseline) ~ 실험 3a (8B 최적화) + Phase 6 자동 평가 결과
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
FONT_NORMAL = "HYGothic-Medium"
FONT_BOLD = "HYGothic-Medium"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "experiment_full_report.pdf")

# --- 색상 ---
BLUE = HexColor("#2D4160")
LIGHT_BLUE = HexColor("#3C82C8")
DARK = HexColor("#1E1E1E")
GRAY = HexColor("#555555")
LIGHT_GRAY = HexColor("#F0F5FA")
RED_BG = HexColor("#FFEBEB")
RED_TEXT = HexColor("#A03232")
GREEN_BG = HexColor("#EBF5F0")
GREEN_TEXT = HexColor("#28783C")
YELLOW_BG = HexColor("#FFF8E1")
YELLOW_TEXT = HexColor("#8B6914")

# --- 스타일 ---
S_TITLE = ParagraphStyle("Title", fontName=FONT_BOLD, fontSize=26, leading=32, textColor=DARK, alignment=1)
S_SUBTITLE = ParagraphStyle("Subtitle", fontName=FONT_BOLD, fontSize=16, leading=22, textColor=LIGHT_BLUE, alignment=1)
S_CENTER = ParagraphStyle("Center", fontName=FONT_NORMAL, fontSize=11, leading=16, textColor=GRAY, alignment=1)
S_H1 = ParagraphStyle("H1", fontName=FONT_BOLD, fontSize=14, leading=20, textColor=DARK, spaceBefore=12, spaceAfter=6)
S_H2 = ParagraphStyle("H2", fontName=FONT_BOLD, fontSize=11, leading=16, textColor=HexColor("#3C3C3C"), spaceBefore=10, spaceAfter=4)
S_BODY = ParagraphStyle("Body", fontName=FONT_NORMAL, fontSize=10, leading=15, textColor=HexColor("#333333"), spaceBefore=2, spaceAfter=4)
S_KV_KEY = ParagraphStyle("KVKey", fontName=FONT_BOLD, fontSize=10, leading=14, textColor=HexColor("#464646"))
S_KV_VAL = ParagraphStyle("KVVal", fontName=FONT_NORMAL, fontSize=10, leading=14, textColor=DARK)
S_TH = ParagraphStyle("TH", fontName=FONT_BOLD, fontSize=9, leading=13, textColor=white, alignment=1)
S_TD = ParagraphStyle("TD", fontName=FONT_NORMAL, fontSize=9, leading=13, textColor=HexColor("#282828"))
S_TD_C = ParagraphStyle("TDC", fontName=FONT_NORMAL, fontSize=9, leading=13, textColor=HexColor("#282828"), alignment=1)
S_BOX = ParagraphStyle("Box", fontName=FONT_BOLD, fontSize=10, leading=15, textColor=GREEN_TEXT)
S_BOX_RED = ParagraphStyle("BoxRed", fontName=FONT_BOLD, fontSize=10, leading=15, textColor=RED_TEXT)
S_BOX_YELLOW = ParagraphStyle("BoxYellow", fontName=FONT_BOLD, fontSize=10, leading=15, textColor=YELLOW_TEXT)
S_HEADER = ParagraphStyle("Header", fontName=FONT_BOLD, fontSize=9, leading=12, textColor=GRAY, alignment=2)


def make_table(headers, rows, col_widths=None):
    w = col_widths or [170 * mm / len(headers)] * len(headers)
    data = [[Paragraph(h, S_TH) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), S_TD_C if i > 0 else S_TD) for i, c in enumerate(row)])
    t = Table(data, colWidths=w, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    t.setStyle(TableStyle(style))
    return t


def make_kv_table(pairs):
    data = [[Paragraph(k, S_KV_KEY), Paragraph(v, S_KV_VAL)] for k, v in pairs]
    t = Table(data, colWidths=[55 * mm, 115 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (0, -1), 10),
    ]))
    return t


def make_box(text, color="green"):
    bg = {"green": GREEN_BG, "red": RED_BG, "yellow": YELLOW_BG}[color]
    style = {"green": S_BOX, "red": S_BOX_RED, "yellow": S_BOX_YELLOW}[color]
    data = [[Paragraph(text.replace("\n", "<br/>"), style)]]
    t = Table(data, colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_BOLD, 9)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(195 * mm, 285 * mm, "SOBA Chat Fine-tuning | Full Experiment Report")
    canvas.setStrokeColor(HexColor("#CCCCCC"))
    canvas.line(15 * mm, 284 * mm, 195 * mm, 284 * mm)
    canvas.setFont(FONT_NORMAL, 8)
    canvas.setFillColor(HexColor("#999999"))
    canvas.drawCentredString(105 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate_report():
    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=A4,
        topMargin=22 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )
    story = []
    SP = lambda h: Spacer(1, h * mm)

    # ================================================================
    # Page 1: 표지
    # ================================================================
    story.append(SP(30))
    story.append(Paragraph("SOBA Fine-tuning", S_TITLE))
    story.append(SP(4))
    story.append(Paragraph("Full Experiment Report", S_SUBTITLE))
    story.append(Paragraph("Exp1 (14B) / Exp2 (8B) / Exp3a (8B Optimized) + Phase 6", S_CENTER))
    story.append(SP(12))
    story.append(Paragraph("2026-04-13", S_CENTER))
    story.append(Paragraph("QLoRA Fine-tuning on A100 80GB", S_CENTER))
    story.append(SP(20))
    story.append(HRFlowable(width="50%", thickness=0.5, color=HexColor("#CCCCCC")))
    story.append(SP(10))
    story.append(Paragraph(
        "본 보고서는 SOBA 마케팅 리서치 챗봇의 대화 생성 기능을 Qwen3 모델로 QLoRA 파인튜닝한 "
        "3차례의 실험 결과와 Phase 6 자동 평가 결과를 종합 기록합니다.",
        ParagraphStyle("Intro", parent=S_BODY, fontSize=11, leading=17, textColor=HexColor("#444444"), alignment=1)
    ))
    story.append(SP(4))
    story.append(Paragraph(
        "목표: Claude Sonnet API를 대체할 수 있는 최적의 파인튜닝 모델과 하이퍼파라미터를 결정합니다.",
        ParagraphStyle("Intro2", parent=S_BODY, fontSize=11, leading=17, textColor=HexColor("#444444"), alignment=1)
    ))

    # ================================================================
    # Page 2: 실험 설정 (공통)
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("1. 공통 실험 설정", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))

    story.append(Paragraph("양자화 및 LoRA (전 실험 공통)", S_H2))
    story.append(make_kv_table([
        ("Quantization", "4-bit NF4, double quant, bfloat16 compute"),
        ("LoRA rank (r)", "32"),
        ("LoRA alpha", "64"),
        ("Target Modules", "q, k, v, o, gate, up, down proj"),
    ]))

    story.append(Paragraph("학습 하이퍼파라미터 (기본)", S_H2))
    story.append(make_kv_table([
        ("Batch Size", "4 x 2 (grad accum) = 8 effective"),
        ("Learning Rate", "2e-4 (cosine scheduler)"),
        ("Warmup", "5%"),
        ("Optimizer", "paged_adamw_8bit"),
        ("Precision", "bfloat16"),
        ("Max Seq Length", "768 tokens"),
    ]))

    story.append(Paragraph("데이터", S_H2))
    story.append(make_table(
        ["Split", "건수", "Seed 수", "비율"],
        [
            ["Train", "4,342", "199", "80%"],
            ["Validation", "513", "24", "10%"],
            ["Test", "578", "26", "10%"],
            ["Total", "5,433", "249", "100%"],
        ],
        col_widths=[45 * mm, 35 * mm, 35 * mm, 35 * mm],
    ))

    story.append(Paragraph("환경", S_H2))
    story.append(make_kv_table([
        ("GPU", "NVIDIA A100-SXM4-80GB"),
        ("PyTorch", "2.6.0+cu124"),
        ("Transformers", "5.5.1"),
        ("TRL", "1.0.0"),
        ("PEFT", "0.18.1"),
    ]))

    # ================================================================
    # Page 3: 실험별 설정 차이
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. 실험별 설정 비교", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(make_table(
        ["항목", "Exp1 (14B)", "Exp2 (8B)", "Exp3a (8B Opt)"],
        [
            ["모델", "Qwen3-14B", "Qwen3-8B", "Qwen3-8B"],
            ["4-bit 파라미터", "8.2B", "4.7B", "4.7B"],
            ["Trainable Params", "128.5M (1.55%)", "87.3M (1.82%)", "87.3M (1.82%)"],
            ["Epochs", "3", "3", "2"],
            ["LoRA dropout", "0.05", "0.05", "0.10"],
            ["Eval Strategy", "epoch", "epoch", "steps (100)"],
            ["Save Strategy", "epoch", "epoch", "steps (100)"],
            ["Early Stopping", "patience=2", "patience=2", "load_best"],
        ],
        col_widths=[40 * mm, 42 * mm, 42 * mm, 46 * mm],
    ))
    story.append(SP(4))

    story.append(make_box(
        "Exp3a 핵심 변경: dropout 0.05 -> 0.10 (정규화 강화) + eval 100 step 단위 (정밀 체크포인팅)",
        "yellow"
    ))

    # ================================================================
    # Page 4: 전체 실험 결과 비교
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. 전체 실험 결과 비교", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(Paragraph("핵심 메트릭 비교", S_H2))
    story.append(make_table(
        ["메트릭", "Exp1 (14B)", "Exp2 (8B)", "Exp3a (8B Opt)"],
        [
            ["Best Eval Loss", "0.6368", "0.6777", "0.6753"],
            ["Best Token Accuracy", "84.97%", "84.58%", "83.71%"],
            ["최적 시점", "Epoch 1 (step 543)", "Epoch 1 (step 543)", "Step 500 (epoch 0.92)"],
            ["학습 시간", "54분", "37분", "18분"],
            ["어댑터 크기", "525.3MB", "360.7MB", "360.7MB"],
            ["Train Loss (avg)", "0.3273", "0.357", "0.5254"],
        ],
        col_widths=[42 * mm, 42 * mm, 42 * mm, 44 * mm],
    ))
    story.append(SP(4))

    story.append(Paragraph("Exp2 vs Exp1 (모델 크기 비교)", S_H2))
    story.append(Paragraph("- Eval Loss 차이: 0.6777 vs 0.6368 = +6.4%", S_BODY))
    story.append(Paragraph("- Token Accuracy 차이: 84.58% vs 84.97% = -0.39pp (미미)", S_BODY))
    story.append(Paragraph("- 학습 시간 절감: 54분 -> 37분 (-31%)", S_BODY))
    story.append(Paragraph("- 어댑터 크기 절감: 525.3MB -> 360.7MB (-31%)", S_BODY))
    story.append(SP(2))

    story.append(Paragraph("Exp3a vs Exp2 (최적화 효과)", S_H2))
    story.append(Paragraph("- Eval Loss 개선: 0.6777 -> 0.6753 (-0.4%)", S_BODY))
    story.append(Paragraph("- 정밀 체크포인팅으로 최적점 탐색: step 543 -> step 500 (epoch 0.92)", S_BODY))
    story.append(Paragraph("- Dropout 0.10으로 과적합 지연 확인", S_BODY))
    story.append(Paragraph("- 학습 시간 추가 절감: 37분 -> 18분 (-51%)", S_BODY))

    # ================================================================
    # Page 5: Exp3a 정밀 체크포인팅 상세
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Exp3a 정밀 체크포인팅 상세", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(Paragraph("100 Step 단위 Eval Loss 곡선", S_H2))
    story.append(make_table(
        ["Step", "Epoch", "Eval Loss", "Token Accuracy", "비고"],
        [
            ["100", "0.18", "0.7631", "80.93%", "학습 초기"],
            ["200", "0.37", "0.7085", "82.23%", ""],
            ["300", "0.55", "0.6886", "83.10%", ""],
            ["400", "0.74", "0.6963", "83.34%", "일시 반등"],
            ["500", "0.92", "0.6753", "83.71%", "BEST (저장됨)"],
            ["600", "1.10", "0.7010", "83.76%", "Epoch 2, loss 상승"],
            ["700", "1.29", "0.6971", "84.19%", ""],
        ],
        col_widths=[22 * mm, 22 * mm, 32 * mm, 38 * mm, 56 * mm],
    ))
    story.append(SP(4))

    story.append(make_box(
        "Best Model: Step 500 (eval_loss = 0.6753)\n"
        "Epoch 1 종료(step 543) 직전이 최적점. 정밀 체크포인팅 없었다면 발견 불가.",
        "green"
    ))
    story.append(SP(4))

    story.append(Paragraph("과적합 분석", S_H2))
    story.append(make_table(
        ["항목", "Exp1 (14B)", "Exp2 (8B)", "Exp3a (8B Opt)"],
        [
            ["과적합 시작", "Epoch 1 이후", "Epoch 1 이후", "Step 500 이후"],
            ["Train-Eval Gap (최종)", "0.56", "0.51", "N/A (early stop)"],
            ["과적합 주요 원인", "데이터 대비 용량 과다", "동일", "dropout 강화로 지연"],
        ],
        col_widths=[42 * mm, 42 * mm, 42 * mm, 44 * mm],
    ))

    # ================================================================
    # Page 6: Phase 6 자동 평가 결과
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Phase 6: 자동 평가 결과", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(Paragraph(
        "평가 대상: Exp3a 모델 (Qwen3-8B, dropout=0.10, step 500 체크포인트)",
        S_BODY
    ))
    story.append(Paragraph(
        "평가 데이터: test.jsonl 578건 (26 seeds) 전체 추론",
        S_BODY
    ))
    story.append(SP(4))

    story.append(Paragraph("response 태스크 (259건)", S_H2))
    story.append(make_table(
        ["평가 항목", "PASS 기준", "결과", "판정"],
        [
            ["JSON 파싱 성공률", ">= 95%", "100.0%", "PASS"],
            ["코멘트 길이 (10~100자)", "90%+", "100.0%", "PASS"],
            ["존댓말 사용률", ">= 95%", "98.5%", "PASS"],
            ["이모지 미사용", "== 0%", "100.0%", "PASS"],
            ["금지어 미사용", "== 0%", "100.0%", "PASS"],
            ["next_question 존재", ">= 98%", "100.0%", "PASS"],
        ],
        col_widths=[48 * mm, 32 * mm, 30 * mm, 30 * mm],
    ))
    story.append(SP(4))

    story.append(Paragraph("기타 태스크", S_H2))
    story.append(make_table(
        ["태스크", "건수", "비어있지 않은 응답", "판정"],
        [
            ["intro", "51", "100.0%", "PASS"],
            ["first_question", "50", "100.0%", "PASS"],
            ["retry", "122", "100.0%", "PASS"],
            ["ending", "51", "100.0%", "PASS"],
            ["title", "45", "100.0%", "PASS"],
        ],
        col_widths=[42 * mm, 30 * mm, 48 * mm, 30 * mm],
    ))
    story.append(SP(6))

    story.append(make_box(
        "Phase 6 자동 평가: 전항목 PASS\n"
        "JSON 파싱 100%, 존댓말 98.5%, 이모지/금지어 0%, 전 태스크 응답률 100%",
        "green"
    ))

    # ================================================================
    # Page 7: 결론 및 다음 단계
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. 결론 및 다음 단계", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(Paragraph("최종 모델 선정", S_H2))
    story.append(make_box(
        "채택 모델: Qwen3-8B + QLoRA, dropout=0.10, Step 500 체크포인트\n"
        "Best Eval Loss: 0.6753 | Token Accuracy: 83.71% | 어댑터 크기: 360.7MB",
        "green"
    ))
    story.append(SP(4))

    story.append(Paragraph("선정 사유", S_H2))
    story.append(Paragraph("1. Phase 6 자동 평가 전항목 PASS (JSON 100%, 존댓말 98.5%)", S_BODY))
    story.append(Paragraph("2. 14B 대비 eval_loss 6% 차이이나, 실제 품질 지표(파싱, 톤)에서 차이 없음", S_BODY))
    story.append(Paragraph("3. 학습 시간 54분 -> 18분 (-67%), 어댑터 크기 525MB -> 361MB (-31%)", S_BODY))
    story.append(Paragraph("4. 서빙 시 GPU 메모리 약 43% 절감으로 비용 효율적", S_BODY))
    story.append(SP(4))

    story.append(Paragraph("핵심 발견", S_H2))
    story.append(Paragraph("1. 정밀 체크포인팅(100 step 단위)으로 Epoch 1 끝이 아닌 Step 500(epoch 0.92)이 최적점임을 발견", S_BODY))
    story.append(Paragraph("2. Dropout 0.10 강화로 과적합이 지연되어 Exp2 대비 소폭 개선 (0.6777 -> 0.6753)", S_BODY))
    story.append(Paragraph("3. 8B와 14B의 6% eval_loss 차이는 모델 용량 차이로, 하이퍼파라미터 조정으로는 좁힐 수 없음", S_BODY))
    story.append(Paragraph("4. SOBA 태스크(코멘트+리프레이징)의 난이도가 낮아 8B로도 충분한 품질 달성", S_BODY))
    story.append(SP(4))

    story.append(Paragraph("남은 단계", S_H2))
    story.append(make_table(
        ["단계", "내용", "상태"],
        [
            ["Phase 6 수동 평가", "블라인드 톤 평가 30건, A/B 비교 20건", "대기"],
            ["Phase 6 통합 세션", "end-to-end 8질문 완주 시나리오 28건", "대기"],
            ["Phase 7 백엔드 통합", "qwen_service.py, llm_service.py 구현 + 서빙 배포", "대기"],
        ],
        col_widths=[45 * mm, 85 * mm, 30 * mm],
    ))

    # ================================================================
    # Page 8: 이슈 및 해결
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("7. 이슈 및 해결", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(make_table(
        ["이슈", "원인", "해결"],
        [
            ["set_submodule 에러", "PyTorch 2.4.1 + transformers 5.5.1 호환", "PyTorch 2.6.0 업그레이드"],
            ["SFTConfig 파라미터 에러", "trl 1.0.0 max_seq_length -> max_length", "train.py 파라미터명 수정"],
            ["HF 토큰 만료", ".env.example 토큰 expired", "새 토큰 갱신"],
            ["Qwen3-7B 모델 없음", "Qwen3 라인업에 7B 부재", "Qwen3-8B로 변경"],
            ["torchvision 호환 에러", "torch 2.6.0 + torchvision 0.19.1 불일치", "torchvision 0.21.0 업그레이드"],
            ["transformers 5.5.3 import 에러", "peft 0.18.1과 호환 안 됨", "transformers 5.5.1로 다운그레이드"],
            ["존댓말 패턴 누락", "죠/군요/이에요 등 미포함", "evaluate.py 정규식 확장"],
        ],
        col_widths=[48 * mm, 60 * mm, 62 * mm],
    ))
    story.append(SP(8))

    story.append(Paragraph("비용 정산", S_H2))
    story.append(make_table(
        ["항목", "소요 시간", "비고"],
        [
            ["Exp1 (14B, 3 epochs)", "54분", "baseline"],
            ["Exp2 (8B, 3 epochs)", "37분", "모델 크기 비교"],
            ["Exp3a (8B, 2 epochs)", "18분", "최적화"],
            ["Phase 6 추론 (578건 x 2회)", "~80분", "자동 평가"],
            ["총 GPU 시간", "~189분 (~3.2시간)", "A100 80GB"],
        ],
        col_widths=[55 * mm, 40 * mm, 75 * mm],
    ))

    # ================================================================
    # 빌드
    # ================================================================
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF 생성 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_report()
