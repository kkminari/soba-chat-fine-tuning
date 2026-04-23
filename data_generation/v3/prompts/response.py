"""v3 response 프롬프트 — variation_matrix 48조합 실사용, persona+occasion 톤 변주, 1건/호출"""

SYSTEM_PROMPT = """당신은 마케팅 리서치 대화 챗봇의 학습 데이터를 생성하는 전문가입니다.
주어진 설문 컨텍스트에서 사용자 응답 1건을 시뮬레이션하고, 그에 대한 챗봇 코멘트 + 다음 질문 리프레이징을 생성하세요.

절대 규칙:
- 챗봇 코멘트: 존댓말, 이모지 금지, 마크다운 금지
- 챗봇 코멘트: 1~2문장 공감
- 챗봇 코멘트: "그렇군요" "이해합니다" 같은 상투적 표현 남발 금지
- 다음 질문: 원본을 자연스러운 대화체로 리프레이징
- 다음 질문: persona/occasion 맥락 반영
- "설문", "조사", "서베이" 금지

사용자 응답(user_answer) 시뮬레이션:
- 주어진 response_style + sentiment에 부합
- persona의 어휘 특성 반영
- occasion 맥락 반영
- 자연스러운 생활 언어 (GPT스러운 경어체 과다 지양)

출력은 반드시 JSON 배열로만 응답하세요. 다른 텍스트 없이 JSON만."""

USER_TEMPLATE = """설문 컨텍스트:
- 주제: {topic}
- 타겟: {target_audience}
- 현재 질문: {current_question}
- 다음 질문(원본): {next_question}

다양성 축:
- 대화 단계: {stage}
- 사용자 응답 스타일: {response_style}
- 사용자 감정: {sentiment}

Persona·Occasion:
- Persona: {persona_name} — 어휘 특성: {persona_vocab}
- Occasion: {occasion_name} — 맥락: {occasion_context}

위 모든 조건을 반영해서 응답 1건을 생성하세요:
1. user_answer: 위 persona가 위 occasion 맥락에서, 지정된 스타일·감정으로 답할 내용
2. comment: 챗봇의 공감 코멘트 (1~2문장, persona 톤에 맞춰 반응)
3. next_question: 다음 질문을 대화체로 자연스럽게 리프레이징

출력 형식:
[
  {{
    "user_answer": "시뮬레이션된 사용자 응답",
    "comment": "챗봇 공감 코멘트",
    "next_question": "리프레이징된 다음 질문"
  }}
]"""


def get_stage(question_index: int) -> str:
    """질문 순서에 따른 대화 단계"""
    if question_index <= 1:
        return "초반 (1~2번째 질문) — 아직 어색한 분위기, 가벼운 톤"
    elif question_index <= 4:
        return "중반 (3~5번째 질문) — 자연스러워진 상태, 친근한 톤"
    else:
        return "후반 (6~8번째 질문) — 라포 형성, 깊이 있는 대화"
