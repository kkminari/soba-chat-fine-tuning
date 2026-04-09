"""
실험 1 (Qwen3-14B baseline) 결과 보고서 PDF 생성 스크립트
reportlab 기반 - CJK 한글 정상 렌더링
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# --- 폰트 등록 (CID 한글 폰트 - 임베딩 불필요, 깨짐 없음) ---
pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
FONT_NORMAL = "HYGothic-Medium"
FONT_BOLD = "HYGothic-Medium"  # CID에는 bold 없으므로 동일 사용

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "experiment1_14b_baseline_report.pdf")

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

# --- 스타일 ---
S_TITLE = ParagraphStyle("Title", fontName=FONT_BOLD, fontSize=26, leading=32, textColor=DARK, alignment=1)
S_SUBTITLE = ParagraphStyle("Subtitle", fontName=FONT_BOLD, fontSize=18, leading=24, textColor=LIGHT_BLUE, alignment=1)
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
S_HEADER = ParagraphStyle("Header", fontName=FONT_BOLD, fontSize=9, leading=12, textColor=GRAY, alignment=2)


def make_table(headers, rows, col_widths=None):
    """테이블 생성 헬퍼"""
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
    """키-값 쌍 테이블"""
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
    """하이라이트 박스"""
    bg = GREEN_BG if color == "green" else RED_BG
    style = S_BOX if color == "green" else S_BOX_RED
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
    """페이지 헤더/푸터"""
    canvas.saveState()
    canvas.setFont(FONT_BOLD, 9)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(195 * mm, 285 * mm, "SOBA Chat Fine-tuning | Experiment Report")
    canvas.setStrokeColor(HexColor("#CCCCCC"))
    canvas.line(15 * mm, 284 * mm, 195 * mm, 284 * mm)
    # Footer
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
    story.append(SP(35))
    story.append(Paragraph("SOBA Fine-tuning", S_TITLE))
    story.append(SP(4))
    story.append(Paragraph("Experiment 1: Qwen3-14B Baseline", S_SUBTITLE))
    story.append(SP(12))
    story.append(Paragraph("2026-04-09", S_CENTER))
    story.append(Paragraph("QLoRA Fine-tuning on A100 80GB", S_CENTER))
    story.append(SP(4))
    story.append(Paragraph("WandB: wandb.ai/mina_kwak-pmi/soba-chatbot-finetuning/runs/2jnuktci", S_CENTER))
    story.append(SP(20))
    story.append(HRFlowable(width="50%", thickness=0.5, color=HexColor("#CCCCCC")))
    story.append(SP(10))
    story.append(Paragraph(
        "본 보고서는 SOBA 마케팅 리서치 챗봇의 대화 생성 기능을 "
        "Qwen3-14B 모델로 QLoRA 파인튜닝한 첫 번째 실험 결과를 기록합니다.",
        ParagraphStyle("Intro", parent=S_BODY, fontSize=11, leading=17, textColor=HexColor("#444444"), alignment=1)
    ))
    story.append(SP(4))
    story.append(Paragraph(
        "목표: Claude Sonnet API를 대체할 수 있는 파인튜닝 모델의 baseline 성능을 확인하고, "
        "후속 실험(7B 비교, 하이퍼파라미터 최적화)의 기준점을 수립합니다.",
        ParagraphStyle("Intro2", parent=S_BODY, fontSize=11, leading=17, textColor=HexColor("#444444"), alignment=1)
    ))

    # ================================================================
    # Page 2: 실험 설정
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("1. 실험 설정", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))

    story.append(Paragraph("모델 및 양자화", S_H2))
    story.append(make_kv_table([
        ("Base Model", "Qwen/Qwen3-14B (8.2B params after 4-bit)"),
        ("Quantization", "4-bit NF4, double quant, bfloat16 compute"),
        ("LoRA rank (r)", "32"),
        ("LoRA alpha", "64"),
        ("LoRA dropout", "0.05"),
        ("Target Modules", "q, k, v, o, gate, up, down proj"),
        ("Trainable Params", "128.5M / 8,291M (1.55%)"),
    ]))

    story.append(Paragraph("학습 하이퍼파라미터", S_H2))
    story.append(make_kv_table([
        ("Epochs", "3"),
        ("Batch Size", "4 x 2 (grad accum) = 8 effective"),
        ("Learning Rate", "2e-4 (cosine scheduler)"),
        ("Warmup", "5%"),
        ("Optimizer", "paged_adamw_8bit"),
        ("Precision", "bfloat16"),
        ("Max Seq Length", "768 tokens"),
        ("Early Stopping", "patience=2 (eval_loss)"),
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
    story.append(Paragraph(
        "태스크 분포 (train): response 1,902 / retry 920 / ending 398 / intro 394 / first_question 387 / title 341",
        S_BODY
    ))

    story.append(Paragraph("환경", S_H2))
    story.append(make_kv_table([
        ("GPU", "NVIDIA A100-SXM4-80GB"),
        ("PyTorch", "2.6.0+cu124"),
        ("Transformers", "5.5.1"),
        ("TRL", "1.0.0"),
        ("PEFT", "0.18.1"),
        ("bitsandbytes", "0.49.2"),
    ]))

    # ================================================================
    # Page 3: 학습 결과
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. 학습 결과", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))

    story.append(Paragraph("핵심 메트릭", S_H2))
    story.append(make_table(
        ["Metric", "값", "비고"],
        [
            ["Train Loss (최종)", "0.1541", "Epoch 3 종료 시점"],
            ["Train Loss (평균)", "0.3273", "전체 학습 평균"],
            ["Eval Loss (최종)", "0.7112", "Epoch 3 종료 시점"],
            ["Eval Loss (Best)", "0.6368", "Epoch 1 (저장된 체크포인트)"],
            ["Mean Token Accuracy", "84.97%", "Epoch 2 시점"],
            ["학습 시간", "54분 21초", "3,261초"],
            ["어댑터 크기", "525.3MB", "LoRA weights + tokenizer"],
        ],
        col_widths=[55 * mm, 40 * mm, 75 * mm],
    ))

    story.append(Paragraph("Epoch별 Loss 추이", S_H2))
    story.append(make_table(
        ["Epoch", "Step", "Train Loss", "Eval Loss", "Token Accuracy"],
        [
            ["1", "~543", "~0.35", "0.637", "84.4%"],
            ["2", "~1,086", "~0.25", "0.654", "85.0%"],
            ["3", "~1,629", "~0.15", "0.711", "84.9%"],
        ],
        col_widths=[22 * mm, 30 * mm, 36 * mm, 36 * mm, 46 * mm],
    ))

    story.append(make_box(
        "Best Model: Epoch 1 (eval_loss = 0.6368)\n"
        "load_best_model_at_end=true 설정으로 Epoch 1 체크포인트가 어댑터로 저장됨",
        "green"
    ))
    story.append(SP(4))

    story.append(Paragraph("Train Loss 곡선 특성", S_H2))
    story.append(Paragraph("- 초기 loss ~2.0에서 시작하여 step 500 부근에서 ~0.3으로 급격히 하강", S_BODY))
    story.append(Paragraph("- Epoch 2~3 구간(step 500~1629)에서 0.15~0.25 수준으로 완만히 수렴", S_BODY))
    story.append(Paragraph("- Learning rate는 cosine schedule에 따라 2e-4에서 점진적으로 감소", S_BODY))
    story.append(Paragraph("- Gradient norm은 0.15~0.3 수준으로 안정적 (발산 없음)", S_BODY))

    # ================================================================
    # Page 4: 과적합 분석
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. 과적합 분석", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(make_box(
        "진단: 전형적인 과적합 (Overfitting)\n"
        "train_loss는 지속 하강하지만 eval_loss는 Epoch 1 이후 상승",
        "red"
    ))
    story.append(SP(4))

    story.append(Paragraph("근거", S_H2))
    story.append(Paragraph("1. Train-Eval Loss 갭: 0.15 vs 0.71 (Epoch 3 기준, 갭 = 0.56)", S_BODY))
    story.append(Paragraph("2. Eval Loss 방향: Epoch 1(0.637) → Epoch 2(0.654) → Epoch 3(0.711) 지속 상승", S_BODY))
    story.append(Paragraph("3. Token Accuracy 정체: Epoch 2에서 85.0% 소폭 상승 후 Epoch 3에서 84.9%로 하락", S_BODY))
    story.append(Paragraph("4. Train Entropy: 0.17까지 하락 (학습 데이터에 대해 매우 확신 높음 = 암기 징후)", S_BODY))

    story.append(Paragraph("원인 추정", S_H2))
    story.append(Paragraph("- 데이터 규모(4,342건) 대비 모델 용량(14B, trainable 128.5M)이 충분히 크므로 쉽게 암기 가능", S_BODY))
    story.append(Paragraph("- Dropout 0.05는 정규화 효과가 약함", S_BODY))
    story.append(Paragraph("- 3 Epoch는 이 데이터 규모에서 과다 (1 Epoch이면 충분히 패턴 학습)", S_BODY))

    story.append(Paragraph("영향", S_H2))
    story.append(Paragraph("- load_best_model_at_end=true 덕분에 저장된 어댑터는 Epoch 1 기준 (best)", S_BODY))
    story.append(Paragraph("- 따라서 현재 저장된 모델의 품질에는 영향 없음", S_BODY))
    story.append(Paragraph("- 다만 Epoch 2~3 학습 시간(~36분)은 낭비됨", S_BODY))

    # ================================================================
    # Page 5: 후속 실험 계획
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. 후속 실험 계획", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))

    story.append(Paragraph("실험 2: Qwen3-7B Baseline", S_H2))
    story.append(Paragraph("- 목적: 14B 대비 7B의 성능 차이 확인, 서빙 비용 절반 가능 여부 판단", S_BODY))
    story.append(Paragraph("- 설정: 동일 하이퍼파라미터 (r=32, epoch=3, lr=2e-4)", S_BODY))
    story.append(Paragraph("- 판단 기준: 14B 대비 5% 미만 차이 → 7B 채택, 10% 이상 차이 → 14B 유지", S_BODY))

    story.append(Paragraph("실험 3: 과적합 대응 최적화", S_H2))
    story.append(make_table(
        ["조정 항목", "현재 값", "변경 값", "이유"],
        [
            ["Epochs", "3", "1~2", "Epoch 1이 이미 최적"],
            ["Dropout", "0.05", "0.1", "정규화 강화"],
            ["Learning Rate", "2e-4", "1e-4 (선택)", "덜 공격적인 학습"],
        ],
        col_widths=[38 * mm, 30 * mm, 30 * mm, 72 * mm],
    ))

    story.append(Paragraph("평가 계획 (Phase 6)", S_H2))
    story.append(make_table(
        ["평가 항목", "PASS 기준", "측정 방법"],
        [
            ["JSON 파싱 성공률", ">= 95%", "response 태스크 출력 파싱"],
            ["코멘트 길이", "10~100자 90%+", "comment 필드 길이 검사"],
            ["존댓말 사용률", ">= 95%", "~요/~세요/~습니다 어미"],
            ["이모지 미사용", "== 0%", "response/retry/ending/title"],
            ["금지어 미사용", "== 0%", "설문/조사/서베이 포함률"],
            ["코멘트 톤 적절성", ">= 85%", "수동 30건 블라인드 평가"],
        ],
        col_widths=[50 * mm, 35 * mm, 85 * mm],
    ))

    # ================================================================
    # Page 6: 이슈 및 해결
    # ================================================================
    story.append(Paragraph("5. 이슈 및 해결", S_H1))
    story.append(HRFlowable(width="40%", thickness=1, color=LIGHT_BLUE))
    story.append(SP(4))

    story.append(make_table(
        ["이슈", "원인", "해결"],
        [
            ["set_submodule 에러", "PyTorch 2.4.1과 transformers 5.5.1 호환 문제", "PyTorch 2.6.0으로 업그레이드"],
            ["SFTConfig 파라미터 에러", "trl 1.0.0에서 max_seq_length → max_length 변경", "train.py 파라미터명 수정"],
            ["HF 토큰 만료", ".env.example의 토큰 expired", "새 토큰으로 갱신"],
        ],
        col_widths=[50 * mm, 60 * mm, 60 * mm],
    ))

    # ================================================================
    # 빌드
    # ================================================================
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF 생성 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_report()
