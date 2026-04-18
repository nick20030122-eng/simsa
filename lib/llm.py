import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()


@dataclass(frozen=True)
class JudgeLlmConfig:
    provider: str
    temperature: float = 0.2


SYSTEM_PROMPT = """\
당신은 공모전 심사위원입니다. 사용자가 업로드한 발표자료(PPT 또는 PDF 텍스트)와 코드 분석 요약을 기반으로, 아래 5가지 기준으로 정량/정성 평가를 수행하세요.

[심사 기준]
1) 아이디어 혁신성: 기존 서비스와의 차별점 및 독창성
2) 개발 난이도: 코드의 복잡도, 사용된 기술의 수준, 완성도
3) 디자인: UI/UX의 직관성 및 시각적 완성도 (발표자료 및 코드 구조 기준)
4) 발표자료 완성도: 논리 전개, 가독성, 전달력
5) 파급효과: 시장성 및 실제 사용자 확산 가능성

[객관성 강화 규칙(필수)]
- 발표자료(PPT·PDF) 텍스트에서 '문제 정의'와 '시장 규모'에 해당하는 근거를 우선적으로 찾고, 파급효과(Impact) 점수에 반드시 반영하세요.
- 코드 분석에서 '사용 라이브러리'와 '로직 깊이(구조/분기/루프/예외 처리 등)'를 근거로 개발 난이도(Difficulty) 점수를 산출하세요.
- 근거가 부족하면 점수를 보수적으로 낮추고, "근거 부족"을 명시하세요.
- 점수는 각 항목 0~20점 정수입니다.
- 총점(total_score)은 5개 항목의 산술평균을 반올림한 0~20점 정수입니다.

[출력 형식(매우 중요)]
- 반드시 JSON만 출력하세요. 마크다운 코드블록/설명 문장/여분 텍스트를 절대 출력하지 마세요.
- 각 항목별 심사평은 정확히 3줄(문장)로 구성된 문자열 배열 3개 요소로 출력하세요.

[JSON 스키마]
{
  "scores": {
    "innovation": 0-20,
    "difficulty": 0-20,
    "design": 0-20,
    "deck_quality": 0-20,
    "impact": 0-20
  },
  "reviews": {
    "innovation_3lines": ["...", "...", "..."],
    "difficulty_3lines": ["...", "...", "..."],
    "design_3lines": ["...", "...", "..."],
    "deck_quality_3lines": ["...", "...", "..."],
    "impact_3lines": ["...", "...", "..."]
  },
  "total_score": 0-20,
  "final_grade": "S|A|B|C|D",
  "overall_review": "총평(한국어)"
}
"""


def _build_llm(cfg: JudgeLlmConfig):
    if cfg.provider == "OpenAI (gpt-4o)":
        from langchain_openai import ChatOpenAI

        key = os.getenv("OPENAI_API_KEY", "").strip()
        return ChatOpenAI(model="gpt-4o", api_key=key, temperature=cfg.temperature)

    from langchain_anthropic import ChatAnthropic

    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=key, temperature=cfg.temperature)


def run_judgement(context_blob: str, cfg: JudgeLlmConfig) -> str:
    llm = _build_llm(cfg)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "아래 자료를 근거로 심사 결과 JSON을 생성하세요.\n\n"
                f"[입력 자료]\n{context_blob}\n"
            )
        ),
    ]
    res = llm.invoke(messages)
    return res.content if hasattr(res, "content") else str(res)

