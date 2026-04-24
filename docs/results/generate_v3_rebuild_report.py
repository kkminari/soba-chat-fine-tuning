"""
SOBA 학습 데이터 v3 재구축 보고서 PDF 생성 (v4 — 시간 제거, 용어 설명 추가)
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============================================================
# 폰트
# ============================================================
HOME = str(Path.home())
FONT_DIR = f"{HOME}/Library/Fonts"
pdfmetrics.registerFont(TTFont("KoPub", f"{FONT_DIR}/KoPubDotumLight_0.ttf"))
pdfmetrics.registerFont(TTFont("KoPubMedium", f"{FONT_DIR}/KoPubDotumMedium_0.ttf"))
pdfmetrics.registerFont(TTFont("KoPubBold", f"{FONT_DIR}/KoPubDotumBold_0.ttf"))

FONT_BODY = "KoPub"
FONT_MED = "KoPubMedium"
FONT_BOLD = "KoPubBold"

OUTPUT_PATH = Path(__file__).parent / "2026-04-24-training-data-v3-rebuild-report.pdf"

# ============================================================
# 색
# ============================================================
NAVY = HexColor("#2A3B57")
BLUE = HexColor("#3D5A80")
ACCENT = HexColor("#5F8DC7")
TEXT = HexColor("#2C2C2C")
GRAY_DARK = HexColor("#5A5A5A")
GRAY = HexColor("#8A8A8A")
GRAY_LIGHT = HexColor("#E0E0E5")
BG_SOFT = HexColor("#F7F8FA")
BG_HL = HexColor("#EEF2F8")
BG_PROBLEM = HexColor("#FDF3F3")
BG_SOLUTION = HexColor("#F1F8F1")
BG_EXAMPLE = HexColor("#F7F5EF")
BG_GLOSS = HexColor("#F5F2EC")
RED = HexColor("#B23A3A")
GREEN = HexColor("#2E7D32")
AMBER = HexColor("#B8860B")


# ============================================================
# 스타일
# ============================================================
def style(name, **kw):
    kw.setdefault("fontName", FONT_BODY)
    kw.setdefault("alignment", TA_LEFT)
    return ParagraphStyle(name=name, **kw)


TITLE = style("Title", fontName=FONT_BOLD, fontSize=17, leading=24,
              textColor=NAVY, spaceAfter=4)
META = style("Meta", fontSize=8.5, leading=12, textColor=GRAY, alignment=2)
H1 = style("H1", fontName=FONT_BOLD, fontSize=13, leading=20,
           textColor=NAVY, spaceBefore=14, spaceAfter=8)
H2 = style("H2", fontName=FONT_MED, fontSize=11, leading=16,
           textColor=BLUE, spaceBefore=12, spaceAfter=5)
STEP_TITLE = style("StepTitle", fontName=FONT_BOLD, fontSize=10.5, leading=15,
                   textColor=NAVY, spaceBefore=8, spaceAfter=3)
BODY = style("Body", fontSize=9.5, leading=16, textColor=TEXT, spaceAfter=5)
BODY_SM = style("BodySm", fontSize=8.5, leading=13, textColor=TEXT, spaceAfter=3)
CAPTION = style("Caption", fontSize=8, leading=11, textColor=GRAY, spaceAfter=3)

CELL = style("Cell", fontSize=8.5, leading=12, textColor=TEXT)
CELL_HEAD = style("CellHead", fontName=FONT_BOLD, fontSize=8.5, leading=12,
                  textColor=HexColor("#FFFFFF"))
CELL_STRONG = style("CellStrong", fontName=FONT_MED, fontSize=8.5, leading=12,
                    textColor=NAVY)
CELL_DESC = style("CellDesc", fontSize=8, leading=11, textColor=GRAY_DARK)

LABEL_PROBLEM = style("LabelProb", fontName=FONT_BOLD, fontSize=8.5, leading=12,
                      textColor=RED)
LABEL_SOLUTION = style("LabelSol", fontName=FONT_BOLD, fontSize=8.5, leading=12,
                       textColor=GREEN)
LABEL_EXAMPLE = style("LabelEx", fontName=FONT_BOLD, fontSize=8.5, leading=12,
                      textColor=AMBER)


def P(text, st=BODY):
    return Paragraph(text, st)


def PC(text, st=CELL):
    return Paragraph(text, st)


def rule(color=GRAY_LIGHT, thickness=0.4, space=6):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=space, spaceAfter=space)


def table(header, rows, col_widths):
    data = [[PC(h, CELL_HEAD) for h in header]]
    for r in rows:
        data.append([PC(c, CELL) if not isinstance(c, Paragraph) else c for c in r])
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.3, GRAY_LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(2, len(data), 2):
        ts.append(("BACKGROUND", (0, i), (-1, i), BG_SOFT))
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(ts))
    return t


def problem_solution_box(problem, solution, example=None, width=None):
    w = width or 166 * mm
    rows = [
        [PC("v2 문제", LABEL_PROBLEM), PC(problem)],
        [PC("v3 해결", LABEL_SOLUTION), PC(solution)],
    ]
    if example:
        rows.append([PC("예시", LABEL_EXAMPLE), PC(example)])

    t = Table(rows, colWidths=[22 * mm, w - 22 * mm])
    ts = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BOX", (0, 0), (-1, -1), 0.4, GRAY_LIGHT),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, GRAY_LIGHT),
        ("BACKGROUND", (0, 0), (0, 0), BG_PROBLEM),
        ("BACKGROUND", (0, 1), (0, 1), BG_SOLUTION),
    ]
    if example:
        ts.append(("BACKGROUND", (0, 2), (0, 2), BG_EXAMPLE))
    t.setStyle(TableStyle(ts))
    return t


def callout(title, body):
    content = [
        [PC(f"<b>{title}</b>",
            style("CalloutTitle", fontName=FONT_BOLD, fontSize=9.5, leading=14,
                  textColor=NAVY))],
        [PC(body, style("CalloutBody", fontSize=9, leading=14, textColor=TEXT))],
    ]
    t = Table(content, colWidths=[166 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_HL),
        ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def h1(story, text):
    story.append(P(text, H1))
    story.append(HRFlowable(width="100%", thickness=1.0, color=NAVY,
                            spaceBefore=0, spaceAfter=8))


# ============================================================
# 본문
# ============================================================
def build_story():
    s = []

    s.append(P("SOBA 학습 데이터 v3 재구축 보고서", TITLE))
    s.append(P("2026-04-24 · data/processed_v3", META))
    s.append(rule(color=NAVY, thickness=1.2, space=6))

    # ============================================================
    # 1. 왜 다시 만들었는가
    # ============================================================
    h1(s, "1. 왜 다시 만들었는가")
    s.append(P(
        "파인튜닝 3회(Exp1 14B, Exp2 8B, Exp3a 8B+dropout 0.10) 모두 epoch 1 이후 과적합이 발생했고, "
        "dropout 조정으로도 곡선이 개선되지 않았다. 학습 데이터 자체에 반복 구조가 있다고 판단하여 "
        "전수 진단한 결과 세 가지 원인이 확인되었다.",
        BODY,
    ))

    cause_rows = [
        [PC("① Split 누수", CELL_STRONG),
         PC("같은 base 주제가 train·val·test에 분산되어 eval_loss가 실제보다 낮게 측정됨"),
         PC("val 26% · test 32% 주제 중복")],
        [PC("② 보조 태스크 정형성", CELL_STRONG),
         PC("intro·retry·ending 시작어구가 1~2개 템플릿에 집중되어 짧은 시간에 패턴 암기 완료"),
         PC('intro "안녕하세요" 99.8%')],
        [PC("③ 시드당 샘플 집중", CELL_STRONG),
         PC("(seed, q_idx)당 3개 변형이 어순만 바꾼 near-duplicate라 실효 다양성 1/10 수준"),
         PC("겉보기 3,970 → 실효 400~1,000")],
    ]
    s.append(table(["원인", "핵심 현상", "측정값"], cause_rows,
                   col_widths=[30 * mm, 90 * mm, 45 * mm]))
    s.append(Spacer(1, 6))
    s.append(callout(
        "과적합 메커니즘",
        "학습 가능한 독립 신호가 약 500 step 분량에 그친다. 1 epoch(543 step)에서 이미 포화되므로 "
        "이후 epoch는 암기만 진행되어 train_loss는 떨어지고 eval_loss는 상승한다.",
    ))

    # ============================================================
    # 2. 어떻게 다시 만들었는가
    # ============================================================
    h1(s, "2. 어떻게 다시 만들었는가")
    s.append(P(
        "세 가지 원인에 각각 대응하는 해결책을 단일 축에 몰아넣지 않고, "
        "서로 독립된 6개 축을 곱셈적으로 조합하는 방식으로 설계했다. "
        "생성 후에는 자동 정제 4단계와 Group-aware Split으로 잔여 중복·누수를 제거했다.",
        BODY,
    ))

    # 2.1 6축 개요 (역할 설명 추가)
    s.append(P("2.1 다양성 6축 — 설계 한눈에 보기", H2))
    axis_rows = [
        [PC("Topic (base)", CELL_STRONG),
         PC("178개 고유 주제 — 리서치 대상 도메인 (alcohol 144 + cosmetics 105)"),
         PC("주제 다양성", CELL_DESC)],
        [PC("Persona", CELL_STRONG),
         PC("10개 응답자 프로필 — 연령·성별·직업별 어휘/톤 힌트 명시"),
         PC("응답자 어투 다양성", CELL_DESC)],
        [PC("Occasion", CELL_STRONG),
         PC("4개 상황 축 — 혼자·친구·가족·비즈니스"),
         PC("격식도·맥락 다양성", CELL_DESC)],
        [PC("variation_matrix", CELL_STRONG),
         PC("48조합 — 응답 스타일(4) × 대화 단계(3) × 사용자 감정(4)"),
         PC("response 태스크 세부 다양성", CELL_DESC)],
        [PC("시작어구 Rotation", CELL_STRONG),
         PC("태스크별 6~8개 어구 풀, 호출마다 랜덤 주입"),
         PC("보조 태스크 정형성 완화", CELL_DESC)],
        [PC("생성 파라미터", CELL_STRONG),
         PC("temperature · top_p · frequency/presence penalty 랜덤"),
         PC("GPT 자체 편향 완화", CELL_DESC)],
    ]
    s.append(table(["축", "구성", "역할"], axis_rows,
                   col_widths=[35 * mm, 90 * mm, 40 * mm]))
    s.append(Spacer(1, 4))
    s.append(P(
        "Persona 10 × Occasion 4 = 40 실효 조합으로 분산되어 조합당 약 165 샘플. "
        "이 규모에서는 persona가 학습 중 과적합 신호로 고착되지 않는다.",
        BODY_SM,
    ))

    # 2.2 단계별 상세
    s.append(P("2.2 단계별 상세 — 무엇이 문제였고 어떻게 바꿨는가", H2))

    # Step 1
    s.append(P("① Persona · Occasion 축 신설", STEP_TITLE))
    s.append(problem_solution_box(
        problem=(
            "v2는 시드(topic)만 있고 <b>응답자가 누구이고 어떤 상황인지</b>를 구분하는 축이 없었다. "
            "결과적으로 GPT가 모든 시드에 대해 동일한 '친절한 마케팅 리서처' 톤으로 수렴했다."
        ),
        solution=(
            "응답자의 어휘·톤을 결정하는 <b>Persona 10개</b>와, 격식도를 결정하는 "
            "<b>Occasion 4개</b>를 교차시켜 40 실효 조합을 만들었다. 각 persona에는 "
            "'완전, 레알, ~거든요'(20대 여대생) 같은 구체적 어휘 힌트를 명시하여 "
            "범용 어시스턴트 톤 수렴을 억제했다."
        ),
        example=(
            "같은 시드 '크래프트 맥주'에 대해 <br/>"
            "&nbsp;&nbsp;<b>p03 × o1 (30대 직장인 여성 + 혼자)</b>: "
            '"퇴근하고 혼자 편의점에서 하나 골라서 집에서 마시는 편이에요…"<br/>'
            "&nbsp;&nbsp;<b>p03 × o4 (30대 직장인 여성 + 비즈니스)</b>: "
            '"회식 자리에서는 다 같이 마실 수 있는 대중적인 걸로…"<br/>'
            "→ 같은 인물도 상황이 바뀌면 톤이 확연히 달라지도록 설계"
        ),
    ))

    # Step 2
    s.append(P("② 시드 통합과 base_seed_id 명시", STEP_TITLE))
    s.append(problem_solution_box(
        problem=(
            "v2는 같은 주제의 변형을 <b>_v1, _v1_v2</b> 같은 접미사를 붙여 별도 시드로 취급했다. "
            "이 때문에 Split 단계에서 'alc_005'는 train에, 'alc_005_v1'은 val에, 'alc_005_v1_v2'는 test에 "
            "들어가도 감지되지 않아 누수가 발생했다."
        ),
        solution=(
            "모든 접미사 변이에서 <b>base_seed_id</b>(원본 ID)를 추출하여 명시적 필드로 추가했다. "
            "이 base_seed_id는 이후 Split 단계에서 그룹 키로 사용되어, 같은 base를 가진 "
            "샘플들은 반드시 같은 split에 속하도록 강제된다."
        ),
        example=(
            "<b>alc_005</b>, <b>alc_005_v1</b>, <b>alc_005_v1_v2</b> 모두 "
            "같은 주제 '크래프트 맥주' → 하나의 base_seed_id <b>alc_005</b>로 통합"
        ),
    ))

    # Step 3
    s.append(P("③ 프롬프트 재설계 — 고정 템플릿 제거", STEP_TITLE))
    s.append(problem_solution_box(
        problem=(
            "v2 intro 프롬프트는 <i>'구조: 따뜻한 인사 → 주제 소개 → 소요시간 → 준비 확인'</i> 같은 "
            "고정 구조를 강제 지시했다. 결과적으로 GPT가 '안녕하세요! 오늘은 [주제]에 대해 이야기 "
            "나눠볼까 해요. 5~10분이면 충분해요. 준비되셨나요?' 패턴으로 99.8% 수렴했다."
        ),
        solution=(
            "1) 고정 구조 지시 전면 삭제. "
            "2) <b>starting_phrase</b> 파라미터를 프롬프트에 주입하여 시작어구를 강제 다변화. "
            "3) few-shot 예시를 서로 다른 시작어구·문장 구조로 구성. "
            "4) <b>persona · occasion</b> 인자를 받아 맥락 반영. "
            "5) title 태스크에는 '설문·조사·서베이' 금지어를 명시적으로 차단."
        ),
        example=(
            "<b>v2 intro (99.8% 동일)</b>: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "'안녕하세요! 오늘은 크래프트 맥주에 대해 이야기 나눠볼까 해요…'<br/>"
            "<b>v3 intro (예시 3건)</b>:<br/>"
            "&nbsp;&nbsp;• '반갑습니다. 오늘은 크래프트 맥주 이야기 들려주실 수 있을까요?'<br/>"
            "&nbsp;&nbsp;• '와주셔서 감사해요. 크래프트 맥주 관련해서 편하게 말씀해주시면…'<br/>"
            "&nbsp;&nbsp;• '만나서 반가워요. 크래프트 맥주 자주 드시는 편인가요?'"
        ),
    ))

    s.append(PageBreak())

    # Step 4
    s.append(P("④ (seed, q_idx)당 3건 → 1건 축소", STEP_TITLE))
    s.append(problem_solution_box(
        problem=(
            "v2는 한 번의 API 호출로 <b>3개 변형을 배치 생성</b>했다. 그러나 GPT는 거의 언제나 "
            "어순·어미만 바꾼 near-duplicate 3개를 만들어냈다. 결과적으로 response 2,430건 중 "
            "실효 정보량이 약 1/3 수준."
        ),
        solution=(
            "호출당 <b>1건만 생성</b>하도록 변경. 다양성은 persona × occasion × variation_matrix × "
            "시작어구 × 생성 파라미터 축에서 확보. 각 샘플이 진짜로 독립적."
        ),
        example=(
            "<b>v2 배치 출력 (seed=alc_003, q_idx=4)</b>:<br/>"
            "&nbsp;&nbsp;변형1: '편의점에서 맥주 고르실 때 가격표는 어떻게 확인하세요?'<br/>"
            "&nbsp;&nbsp;변형2: '편의점에서 맥주 살 때, 가격표는 어떻게 체크하세요?'<br/>"
            "&nbsp;&nbsp;변형3: '맥주 고르실 때, 편의점에서 가격표는 어떻게 보세요?'<br/>"
            "→ 세 문장 의미 동일. 모델이 하나 배우면 나머지 둘 자동 맞춤"
        ),
    ))

    # Step 5 — 파라미터 설명 추가
    s.append(P("⑤ 생성 파라미터 다양화", STEP_TITLE))
    s.append(problem_solution_box(
        problem=(
            "v2는 모든 API 호출에서 고정된 기본 파라미터를 사용했다. GPT-4o는 자체적으로 "
            "'안전한 톤'으로 수렴하는 편향이 있어, 프롬프트를 다양화해도 출력이 비슷하게 나왔다."
        ),
        solution=(
            "API 호출마다 아래 4개 파라미터를 랜덤 샘플링하여 적용. 프롬프트 축과 "
            "독립적으로 출력 다양성 +20~30% 확보."
        ),
    ))
    s.append(Spacer(1, 2))

    param_rows = [
        [PC("temperature", CELL_STRONG),
         PC("0.8 / 1.0 / 1.2"),
         PC("다음 토큰 선택의 무작위성 제어. 낮으면 보수적·정답 같은 답, 높으면 창의적·다양한 답.")],
        [PC("top_p", CELL_STRONG),
         PC("0.92 / 0.95"),
         PC("다음 토큰 후보 중 누적 확률 p까지만 고려. 낮으면 상위 단어만, 높으면 드문 단어까지 허용.")],
        [PC("frequency_penalty", CELL_STRONG),
         PC("0.3 / 0.5"),
         PC("이미 많이 등장한 토큰에 비례 페널티. 같은 어절 반복("
            "'조금 더 조금 더')을 억제.")],
        [PC("presence_penalty", CELL_STRONG),
         PC("0.2 / 0.4"),
         PC("한 번이라도 등장한 토큰에 페널티. 새로운 주제·어휘로 넘어가도록 유도.")],
    ]
    s.append(table(["파라미터", "값", "역할"],
                   param_rows,
                   col_widths=[38 * mm, 25 * mm, 102 * mm]))

    # Step 6 — dedup + 방법론 설명
    s.append(P("⑥ 자동 중복 제거 4단 파이프라인", STEP_TITLE))
    s.append(P(
        "생성물에 남아 있을 수 있는 중복·유사 샘플을 단계별로 차단. "
        "표면 중복(어순 차이)부터 의미 중복(어휘 치환)까지 단계별 기법이 다름.",
        BODY_SM,
    ))
    dedup_rows = [
        [PC("1단", CELL_STRONG),
         PC("NFC 정규화 + SHA-256 해시"),
         PC("완전 동일 제거"),
         PC("210건")],
        [PC("2단", CELL_STRONG),
         PC("MinHash + LSH (어절 5-gram · Jaccard 0.7)<br/>"
            "retry는 3-gram, title은 Levenshtein ≤ 2"),
         PC("어순·어미만 다른 표면 중복"),
         PC("143건")],
        [PC("3단", CELL_STRONG),
         PC("OpenAI text-embedding-3-small<br/>cosine ≥ 0.92"),
         PC("의미는 같고 어휘가 다른 중복"),
         PC("20건")],
        [PC("4단", CELL_STRONG),
         PC("v3 ↔ v2 전체 cosine 유사도<br/>max > 0.95 제거"),
         PC("v2와 지나치게 유사한 재생성 차단"),
         PC("44건")],
    ]
    s.append(table(
        ["단계", "방법", "목적", "제거 건수"],
        dedup_rows,
        col_widths=[15 * mm, 70 * mm, 55 * mm, 20 * mm],
    ))
    s.append(Spacer(1, 4))
    s.append(callout(
        "한국어 특화 배려",
        "title처럼 짧은 텍스트(평균 12자)에는 MinHash가 구조적으로 작동하지 않아 Levenshtein 거리를 사용. "
        "임베딩 cosine 임계는 영어 권장치(0.85) 대신 0.92로 높여 한국어 존댓말·어미 차이의 오탐을 회피.",
    ))

    # Step 7
    s.append(P("⑦ Group-aware Split — 누수 원천 차단", STEP_TITLE))
    s.append(problem_solution_box(
        problem=(
            "v2 Split은 태스크별로 랜덤 분할되어, 같은 base_seed_id가 여러 split에 흩어지는 구조. "
            "intro/ending/first_question/title이 특히 심각했다."
        ),
        solution=(
            "scikit-learn의 <b>StratifiedGroupKFold</b>를 사용. 같은 base_seed_id 샘플은 반드시 "
            "같은 split에 들어가도록 강제하면서, task_type 비율은 train/val/test에서 유지. "
            "분할 직후 <b>train_groups ∩ val_groups ∩ test_groups = ∅</b> 를 assertion으로 검증."
        ),
        example=(
            "분할 결과 누수 건수: train ↔ val = <b>0</b>, train ↔ test = <b>0</b>, val ↔ test = <b>0</b>"
        ),
    ))

    # 2.3 Pilot
    s.append(P("2.3 Pilot 사전 검증", H2))
    s.append(P(
        "본 6,000건 생성 전에 태스크별 30건씩 <b>총 180건을 pilot으로 먼저 생성</b>하고 품질 지표를 측정했다. "
        "Pilot 단계에서 intro 시작어구 top-1이 v2의 99.8%에서 26.7%로, response 첫 3어절 고유 비율이 100%로 "
        "나오는 것을 확인한 뒤 본 생성에 진입했다. 프롬프트에 근본적 문제가 있어도 조기에 잡을 수 있는 안전장치다.",
        BODY,
    ))

    s.append(PageBreak())

    # ============================================================
    # 3. 결과
    # ============================================================
    h1(s, "3. 결과")

    s.append(P("3.1 규모 비교", H2))
    size_rows = [
        [PC("총 샘플", CELL_STRONG), PC("4,981"), PC("5,537")],
        [PC("train", CELL_STRONG), PC("3,970"), PC("4,402")],
        [PC("val", CELL_STRONG), PC("478"), PC("569")],
        [PC("test", CELL_STRONG), PC("533"), PC("566")],
        [PC("persona × occasion 조합", CELL_STRONG), PC("-"), PC("40 / 40")],
        [PC("variation_matrix 사용 조합", CELL_STRONG), PC("9 / 48"), PC("16 / 48")],
    ]
    s.append(table(["항목", "v2 (기존)", "v3 (신규)"],
                   size_rows, col_widths=[70 * mm, 45 * mm, 45 * mm]))

    s.append(Spacer(1, 8))
    s.append(P("3.2 품질 지표 (v2 대비)", H2))
    pass_st = style("OK", fontName=FONT_MED, fontSize=8.5, leading=12, textColor=GREEN)
    warn_st = style("WARN", fontName=FONT_MED, fontSize=8.5, leading=12, textColor=AMBER)
    metric_rows = [
        [PC("intro 시작어구 top-1", CELL_STRONG),
         PC("99.8%"), PC("14.25%"), PC("PASS", pass_st)],
        [PC("response 시작어구 top-1", CELL_STRONG),
         PC("3-gram boilerplate"), PC("4.36%"), PC("PASS", pass_st)],
        [PC("Split 누수 (train ↔ val)", CELL_STRONG),
         PC("6 seeds (26%)"), PC("0"), PC("PASS", pass_st)],
        [PC("Split 누수 (train ↔ test)", CELL_STRONG),
         PC("9 seeds (32%)"), PC("0"), PC("PASS", pass_st)],
        [PC("Seed 내 평균 pairwise cosine", CELL_STRONG),
         PC("미측정"), PC("0.32"), PC("PASS", pass_st)],
        [PC("persona × occasion 균형 (std/mean)", CELL_STRONG),
         PC("-"), PC("0.017"), PC("PASS", pass_st)],
        [PC("v3 ↔ v2 cross-sim 평균", CELL_STRONG),
         PC("-"), PC("0.657"), PC("PASS", pass_st)],
        [PC("first_question 시작어구 top-1", CELL_STRONG),
         PC("대부분 '안녕하세요'"), PC("40.4%"), PC("경계", warn_st)],
        [PC("retry 시작어구 top-1", CELL_STRONG),
         PC("단일 템플릿 독점"), PC("21.6%"), PC("경계", warn_st)],
        [PC("ending 시작어구 top-1", CELL_STRONG),
         PC("상위 10개가 29%"), PC("20.3%"), PC("경계", warn_st)],
    ]
    s.append(table(["지표", "v2", "v3", "판정"],
                   metric_rows,
                   col_widths=[65 * mm, 45 * mm, 30 * mm, 20 * mm]))
    s.append(Spacer(1, 4))
    s.append(P(
        "핵심 지표(Split 누수·다양성·cross-sim)는 모두 통과. 보조 태스크 3종의 시작어구 편향은 "
        "기준을 살짝 넘지만 v2의 단일 템플릿 독점 상태와 비교하면 구조적 문제는 해소된 상태.",
        BODY_SM,
    ))

    s.append(Spacer(1, 8))
    s.append(P("3.3 비용", H2))
    cost_rows = [
        [PC("API 호출 수", CELL_STRONG), PC("6,400회 (실패 92회, 성공률 98.6%)")],
        [PC("GPT-4o (response 태스크)", CELL_STRONG), PC("약 $7")],
        [PC("GPT-4o-mini (보조 5개 태스크)", CELL_STRONG), PC("약 $1")],
        [PC("OpenAI 임베딩", CELL_STRONG), PC("약 $0.02")],
        [PC("총합", CELL_STRONG), PC("약 $8 ~ $9")],
    ]
    s.append(table(["항목", "수치"], cost_rows, col_widths=[75 * mm, 85 * mm]))

    # ============================================================
    # 4. 기대 효과와 다음 단계
    # ============================================================
    h1(s, "4. 기대 효과와 다음 단계")

    s.append(P("4.1 기대 효과", H2))
    for t, d in [
        ("과적합 지점이 뒤로 이동",
         "실효 독립 샘플 3배 이상 증가로 epoch 1.5 ~ 2까지 곡선 유지 예상"),
        ("eval_loss의 정직성 확보",
         "Split 누수 0% 보장으로 측정값이 실제 일반화 성능을 그대로 반영"),
        ("모델 출력 다양성 확대",
         "6개 독립 축의 곱셈적 조합으로 v2 대비 훨씬 풍부한 스타일 변주 학습"),
        ("고품질 규모",
         "LIMA(Meta 2023)는 1,000건 고품질로 GPT-4급 alignment를 달성. v3의 5,537건은 충분한 규모"),
    ]:
        s.append(Paragraph(
            f"<b>{t}</b> &nbsp;&nbsp; {d}",
            style("Effect", fontSize=9.5, leading=15, textColor=TEXT,
                  leftIndent=4, spaceAfter=4),
        ))

    s.append(Spacer(1, 6))
    s.append(P("4.2 다음 단계 (Phase 7 진입 전)", H2))
    next_rows = [
        [PC("Exp4 검증 학습", CELL_STRONG),
         PC("RunPod A100에서 v3로 1 epoch 학습. eval_loss가 epoch 1.5 ~ 2까지 유지되는지 확인")],
        [PC("수동 평가 재실행", CELL_STRONG),
         PC("v3 모델 품질이 v2 대비 회귀 없는지 확인 (manual_eval.py 재활용)")],
        [PC("manual_eval.py 버그 수정", CELL_STRONG),
         PC("Image 태스크의 정상 사진 요청에 감점하는 채점 로직 수정")],
        [PC("Phase 7 백엔드 통합", CELL_STRONG),
         PC("vLLM 서빙 환경 구축 + Claude API 호출부를 Qwen3-8B v3 모델로 교체")],
    ]
    s.append(table(["단계", "내용"], next_rows, col_widths=[45 * mm, 120 * mm]))

    # ============================================================
    # 5. 용어 · 방법론 설명
    # ============================================================
    h1(s, "5. 용어 · 방법론 설명")
    s.append(P(
        "본문에서 사용한 주요 변수와 방법론을 간단히 정리한다.",
        BODY_SM,
    ))

    # 5.1 다양성 축 관련 변수
    s.append(P("5.1 다양성 축 관련 변수", H2))
    var_rows = [
        [PC("base_seed_id", CELL_STRONG),
         PC("접미사 변이(_v1, _v1_v2)를 제거한 원본 시드 ID. 같은 주제의 변형이 여러 split에 분산되는 것을 막기 위해 Split 그룹 키로 사용.")],
        [PC("persona_id / occasion_id", CELL_STRONG),
         PC("응답자 프로필 ID(p01 ~ p10)와 상황 ID(o1 ~ o4). 같은 시드도 (persona, occasion) 조합에 따라 다른 톤으로 생성.")],
        [PC("variation_matrix", CELL_STRONG),
         PC("response 태스크의 3개 세부 축 조합. <b>response_style</b>(짧게/상세히/구어체/이모지), "
            "<b>conversation_stage</b>(초반/중반/후반), <b>user_sentiment</b>(긍정/중립/부정/무관심). "
            "4×3×4 = 48조합.")],
        [PC("starting_phrase", CELL_STRONG),
         PC("태스크별 시작어구 풀에서 랜덤 선택되는 첫 어절. 프롬프트에 주입되어 모델이 고정 시작어구로 수렴하는 것을 억제.")],
        [PC("base_seed_id 누수", CELL_STRONG),
         PC("같은 base_seed_id가 train과 val/test에 동시에 존재하는 상태. 모델이 학습 중 본 주제를 평가에서 다시 보게 되어 eval_loss가 왜곡됨.")],
    ]
    s.append(table(["변수", "의미와 역할"], var_rows,
                   col_widths=[40 * mm, 125 * mm]))

    # 5.2 생성 파라미터
    s.append(P("5.2 OpenAI 생성 파라미터", H2))
    param_rows2 = [
        [PC("temperature", CELL_STRONG),
         PC("다음 토큰 확률분포의 sharpness. 0에 가까우면 항상 가장 확률 높은 단어만 선택(보수적), 1 이상이면 확률이 낮은 단어도 자주 선택(창의적).")],
        [PC("top_p (nucleus sampling)", CELL_STRONG),
         PC("다음 토큰 후보 중 누적 확률 p까지만 고려. top_p=0.95면 상위 95% 확률 구간의 단어들만 선택 후보.")],
        [PC("frequency_penalty", CELL_STRONG),
         PC("이미 많이 등장한 토큰에 <b>등장 횟수에 비례하는</b> 페널티. 같은 어절 반복을 억제 (예: '조금 더 조금 더').")],
        [PC("presence_penalty", CELL_STRONG),
         PC("한 번이라도 등장한 토큰에 <b>일정한</b> 페널티. 이미 나온 주제·어휘 대신 새로운 방향으로 전개하도록 유도.")],
    ]
    s.append(table(["파라미터", "의미"], param_rows2,
                   col_widths=[40 * mm, 125 * mm]))

    s.append(PageBreak())

    # 5.3 중복 제거 방법론
    s.append(P("5.3 중복 제거 방법론", H2))
    method_rows = [
        [PC("NFC 정규화", CELL_STRONG),
         PC("Unicode Normalization Form C. 한글은 NFC(조합형)와 NFD(분리형)가 있어 동일 글자도 다른 바이트 열이 될 수 있음. 해시 비교 전 NFC로 통일 필수.")],
        [PC("SHA-256 해시", CELL_STRONG),
         PC("텍스트를 256bit 고유 지문으로 변환. 완전 동일한 텍스트는 동일 해시. 빠르고 false positive 0.")],
        [PC("n-gram (shingle)", CELL_STRONG),
         PC("연속된 n개 어절(또는 문자)을 묶은 조각. 예: '안녕 오늘 은 좋은 날씨'의 3-gram = {'안녕 오늘 은', '오늘 은 좋은', '은 좋은 날씨'}.")],
        [PC("MinHash", CELL_STRONG),
         PC("집합(n-gram 집합)을 고정 길이 시그니처로 요약하는 기법. 두 MinHash 시그니처의 충돌 비율이 Jaccard 유사도를 근사.")],
        [PC("LSH (Locality Sensitive Hashing)", CELL_STRONG),
         PC("유사한 시그니처끼리 같은 버킷에 모이도록 하는 해싱. 수백만 문서 중 유사 후보를 O(1) 시간에 찾을 수 있음.")],
        [PC("Jaccard 유사도", CELL_STRONG),
         PC("두 집합의 교집합 크기를 합집합 크기로 나눈 값(0~1). 1에 가까울수록 유사. 어순·어미 차이에 강건.")],
        [PC("Levenshtein 거리", CELL_STRONG),
         PC("한 문자열을 다른 문자열로 바꾸는 데 필요한 최소 삽입·삭제·교체 연산 수. 짧은 텍스트 비교에 적합.")],
        [PC("임베딩 (embedding)", CELL_STRONG),
         PC("텍스트를 고차원 벡터(여기선 1536차원)로 변환한 것. 의미가 비슷하면 벡터 방향도 비슷하도록 학습됨.")],
        [PC("cosine 유사도", CELL_STRONG),
         PC("두 벡터의 각도 코사인 값(-1~1, 보통 0~1). 벡터 크기 무관하게 <b>방향</b>만 비교. 임베딩 간 의미 유사도 측정의 표준.")],
        [PC("cross-similarity", CELL_STRONG),
         PC("두 데이터셋(v3, v2) 간 유사도. v3 샘플 각각에 대해 v2 전체와의 최대 cosine을 구해 지나치게 유사한 재생성을 차단.")],
        [PC("StratifiedGroupKFold", CELL_STRONG),
         PC("scikit-learn의 분할 기법. <b>같은 그룹</b>(base_seed_id)은 반드시 한 split에 들어가게 하면서 동시에 "
            "<b>라벨(task_type) 비율</b>을 train/val/test에 균등하게 유지.")],
    ]
    s.append(table(["용어", "의미"], method_rows,
                   col_widths=[45 * mm, 120 * mm]))

    # 5.4 평가·학습 용어
    s.append(P("5.4 평가·학습 관련 용어", H2))
    eval_rows = [
        [PC("epoch", CELL_STRONG),
         PC("전체 학습 데이터를 모델이 한 번 훑는 과정. v2 Exp3a 기준 1 epoch ≈ 543 step.")],
        [PC("step (batch)", CELL_STRONG),
         PC("배치 하나를 처리하는 단위. gradient가 한 번 갱신되는 최소 단위.")],
        [PC("train_loss / eval_loss", CELL_STRONG),
         PC("학습 데이터 / 검증 데이터에 대한 손실값. train_loss는 계속 떨어지는데 eval_loss가 오르기 시작하면 과적합 신호.")],
        [PC("과적합 (overfitting)", CELL_STRONG),
         PC("모델이 학습 데이터를 '외우기' 시작하여 일반화 성능이 떨어지는 현상.")],
        [PC("dropout", CELL_STRONG),
         PC("학습 중 일부 뉴런을 무작위로 꺼서 과적합을 완화하는 기법. 0.05~0.10 범위 주로 사용.")],
        [PC("ChatML 형식", CELL_STRONG),
         PC("OpenAI/Qwen 계열의 대화 학습 포맷. system/user/assistant 역할별 메시지 리스트 구조.")],
        [PC("(seed, q_idx)", CELL_STRONG),
         PC("하나의 시드(topic)와 그 안의 질문 인덱스(0~7) 쌍. TPOLEACI 8질문 프레임워크에서 현재 질문-다음 질문 쌍을 식별.")],
    ]
    s.append(table(["용어", "의미"], eval_rows,
                   col_widths=[40 * mm, 125 * mm]))

    s.append(Spacer(1, 10))
    s.append(rule())
    s.append(P(
        "관련 산출물 · data/processed_v3/{train,val,test}.jsonl (ChatML) "
        "· data_generation/v3/ (생성·정제·검증 스크립트) "
        "· docs/plans/2026-04-23-data-regeneration-plan.md "
        "· docs/plans/2026-04-23-deduplication-methodology.md "
        "· docs/results/2026-04-23-data-quality-diagnosis.md",
        CAPTION,
    ))

    return s


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_BODY, 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="SOBA 학습 데이터 v3 재구축 보고서",
        author="SOBA Fine-tuning Team",
    )
    story = build_story()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
