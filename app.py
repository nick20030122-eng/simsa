import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from lib.analysis import analyze_code_files, extract_pdf_text, extract_ppt_signals, extract_ppt_text
from lib.llm import JudgeLlmConfig, run_judgement
from lib.schema import JudgeResult


load_dotenv()


@dataclass(frozen=True)
class UploadedBundle:
    pptx_files: List
    pdf_files: List
    code_files: List


def _grade_from_total(total: int) -> str:
    if total >= 18:
        return "S"
    if total >= 16:
        return "A"
    if total >= 14:
        return "B"
    if total >= 12:
        return "C"
    return "D"


def _radar_chart(scores: dict) -> go.Figure:
    """5개 항목 점수 레이더(방사형) 차트."""
    labels = [
        "아이디어 혁신성",
        "개발 난이도",
        "디자인",
        "발표자료 완성도",
        "파급효과",
    ]
    values = [int(scores.get(k, 0)) for k in labels]
    values_closed = values + values[:1]
    labels_closed = labels + labels[:1]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                fillcolor="rgba(45, 106, 159, 0.22)",
                line=dict(color="#1e4d7b", width=2),
                name="점수",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(255,255,255,0.5)",
            radialaxis=dict(
                visible=True,
                range=[0, 20],
                tickvals=[0, 5, 10, 15, 20],
                gridcolor="rgba(30, 77, 123, 0.12)",
                linecolor="rgba(30, 77, 123, 0.2)",
            ),
            angularaxis=dict(linecolor="rgba(30, 77, 123, 0.15)", gridcolor="rgba(30, 77, 123, 0.08)"),
        ),
        font=dict(family="system-ui, 'Malgun Gothic', sans-serif", color="#1a3550", size=11),
        showlegend=False,
        margin=dict(l=24, r=24, t=12, b=12),
        height=300,
    )
    return fig


def _result_sheet_html(total: int, grade: str, score_map: dict) -> str:
    """한 번에 출력해 Streamlit에서 깨지는 div 래핑(빈 긴 박스)을 방지합니다."""
    rows = _score_table_html(score_map)
    return (
        '<div class="ic-card ic-result-sheet">'
        '<div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:baseline;margin-bottom:1rem;">'
        '<div><span style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;color:#1e4d7b;text-transform:uppercase;">총점</span>'
        f'<div style="font-size:2rem;font-weight:800;color:#153a5c;line-height:1.1;">{total}<span style="font-size:1rem;font-weight:600;opacity:0.55">/20</span></div></div>'
        '<div style="padding:0.35rem 0.85rem;border-radius:999px;background:linear-gradient(135deg,#e0f2fe,#f0f9ff);border:1px solid rgba(125,211,252,0.5);">'
        '<span style="font-size:0.72rem;color:#4a6b82;">등급</span> '
        f'<strong style="font-size:1.1rem;color:#153a5c;">{grade}</strong></div></div>'
        '<p class="ic-panel-title" style="margin-bottom:0.5rem;">항목별 점수</p>'
        f"{rows}"
        "</div>"
    )


def _collect_uploads() -> UploadedBundle:
    uploaded = st.file_uploader(
        "PPT(.pptx), PDF(.pdf) 또는 코드 파일을 여기에 놓거나 선택하세요",
        type=[
            "pptx",
            "pdf",
            "py",
            "js",
            "ts",
            "tsx",
            "jsx",
            "java",
            "kt",
            "go",
            "rs",
            "sql",
            "md",
            "txt",
            "json",
            "yml",
            "yaml",
            "toml",
        ],
        accept_multiple_files=True,
    )
    pptx_files: List = []
    pdf_files: List = []
    code_files: List = []
    for f in uploaded or []:
        name = (f.name or "").lower()
        if name.endswith(".pptx"):
            pptx_files.append(f)
        elif name.endswith(".pdf"):
            pdf_files.append(f)
        else:
            code_files.append(f)
    return UploadedBundle(pptx_files=pptx_files, pdf_files=pdf_files, code_files=code_files)


def _require_api_key(provider: str) -> Tuple[bool, str]:
    if provider == "OpenAI (gpt-4o)":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        return (bool(key), "OPENAI_API_KEY")
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    return (bool(key), "ANTHROPIC_API_KEY")


def _score_table_html(score_map: dict) -> str:
    parts: List[str] = []
    for label, val in score_map.items():
        parts.append(
            f'<div class="ic-score-row"><span>{label}</span><strong>{int(val)}/20</strong></div>'
        )
    return "".join(parts)


def _truncate_sections(sections: Iterable[Tuple[str, str]], max_total_chars: int) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    used = 0
    for title, content in sections:
        if used >= max_total_chars:
            break
        remaining = max_total_chars - used
        cut = content[:remaining]
        out.append((title, cut))
        used += len(cut)
    return out


_THIS_DIR = os.path.dirname(__file__)
_LOGO_PATH = os.path.abspath(os.path.join(_THIS_DIR, "..", "이노커브 로고.png"))
_LOGO_MOF_PATH = os.path.abspath(os.path.join(_THIS_DIR, "..", "재정경제부_로고.png"))


@st.cache_data(show_spinner=False)
def _load_logo_cutout(path: str, display_width_px: int, pixel_ratio: float = 5.0) -> Image.Image:
    """
    로고 PNG 누끼 후, 화면 표시 너비는 display_width_px 이지만
    내부 비트맵은 display_width_px * pixel_ratio 로 렌더해 두었다가
    st.image(..., width=display_width_px)로 축소 표시 → 훨씬 선명하게 보입니다.
    """
    img = Image.open(path).convert("RGBA")
    w, h = img.size

    def _is_bg(r: int, g: int, b: int) -> bool:
        if r >= 245 and g >= 245 and b >= 245:
            return True
        if r >= 220 and g >= 238 and b >= 250:
            return True
        return False

    pixels = list(img.getdata())
    cut: List[Tuple[int, int, int, int]] = []
    for r, g, b, a in pixels:
        if _is_bg(r, g, b):
            cut.append((r, g, b, 0))
        else:
            cut.append((r, g, b, a))
    img.putdata(cut)

    if w <= 0 or h <= 0:
        return img
    render_w = max(1, int(display_width_px * max(2.0, pixel_ratio)))
    render_w = min(render_w, 4000)
    ratio = render_w / float(w)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def _combine_logos_side_by_side(left: Image.Image, right: Image.Image, gap_px: int) -> Image.Image:
    """두 로고를 가로로 붙임(세로는 높은 쪽 기준 중앙 정렬). gap_px는 합성 비트맵 기준 픽셀."""
    gap_px = max(0, gap_px)
    h = max(left.height, right.height)
    w = left.width + gap_px + right.width
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y_l = (h - left.height) // 2
    y_r = (h - right.height) // 2
    out.paste(left, (0, y_l), left)
    out.paste(right, (left.width + gap_px, y_r), right)
    return out


st.set_page_config(page_title="인공지능 심사위원", page_icon="🧑‍⚖️", layout="wide")
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
<style>
  :root {
    --ic-navy: #153a5c;
    --ic-navy-soft: #1e4d7b;
    --ic-sky-50: #f0f9ff;
    --ic-sky-100: #e0f2fe;
    --ic-sky-200: #bae6fd;
    --ic-sky-300: #7dd3fc;
    --ic-text: #0f2940;
    --ic-muted: #4a6b82;
    --ic-card: rgba(255, 255, 255, 0.92);
    --ic-border: rgba(125, 211, 252, 0.45);
    --ic-shadow: 0 12px 40px rgba(21, 58, 92, 0.08);
  }
  .stApp {
    background: radial-gradient(1200px 600px at 50% -10%, #dbeafe 0%, transparent 55%),
                linear-gradient(180deg, #e8f4fc 0%, #f5fbff 38%, #fafdff 100%);
  }
  div.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1100px;
  }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5fbff 0%, #eef8ff 100%) !important;
    border-right: 1px solid rgba(125, 211, 252, 0.35);
  }
  section[data-testid="stSidebar"] > div {
    background: transparent !important;
  }
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    font-family: "DM Sans", "Malgun Gothic", sans-serif !important;
    color: var(--ic-navy) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
  }
  h1, h2, h3 {
    font-family: "DM Sans", "Malgun Gothic", sans-serif !important;
    color: var(--ic-text) !important;
    letter-spacing: -0.03em;
  }
  .ic-hero {
    text-align: center;
    padding: 0.5rem 0 1.25rem 0;
  }
  .ic-hero-spacer { height: 56px; }
  .ic-logo-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 0 1rem 0;
  }
  .ic-logo-to-title-gap {
    height: 2.5rem;
  }
  .ic-main-title {
    font-family: "DM Sans", "Malgun Gothic", sans-serif;
    font-size: clamp(1.85rem, 4.2vw, 2.45rem);
    font-weight: 800;
    color: var(--ic-navy);
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.045em;
    line-height: 1.2;
    text-align: center;
  }
  .ic-tagline {
    font-family: "DM Sans", "Malgun Gothic", sans-serif;
    font-size: 0.95rem;
    color: var(--ic-muted);
    margin: 0 auto;
    max-width: 36rem;
    line-height: 1.55;
  }
  .ic-panel {
    background: var(--ic-card);
    border: 1px solid var(--ic-border);
    border-radius: 20px;
    padding: 1.35rem 1.5rem;
    box-shadow: var(--ic-shadow);
    margin-bottom: 1.1rem;
  }
  .ic-panel-title {
    font-family: "DM Sans", "Malgun Gothic", sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ic-navy-soft);
    margin: 0 0 0.85rem 0;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, #f8fcff 0%, #ffffff 100%) !important;
    border: 2px dashed rgba(125, 211, 252, 0.85) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(30, 77, 123, 0.45) !important;
    background: #ffffff !important;
  }
  div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a4a73 0%, #2d6a9f 100%) !important;
    border: none !important;
    color: #fff !important;
    font-family: "DM Sans", "Malgun Gothic", sans-serif !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 8px 20px rgba(30, 77, 123, 0.22) !important;
    width: 100%;
  }
  div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 10px 26px rgba(30, 77, 123, 0.28) !important;
    filter: brightness(1.03);
  }
  div.stButton > button[kind="primary"]:disabled {
    opacity: 0.45 !important;
    box-shadow: none !important;
  }
  .ic-stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
  }
  .ic-stat {
    flex: 1;
    min-width: 120px;
    background: linear-gradient(145deg, #f0f9ff 0%, #ffffff 100%);
    border: 1px solid rgba(125, 211, 252, 0.4);
    border-radius: 14px;
    padding: 0.65rem 0.85rem;
    text-align: center;
  }
  .ic-stat-num {
    font-family: "DM Sans", "Malgun Gothic", sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--ic-navy);
  }
  .ic-stat-label {
    font-size: 0.72rem;
    color: var(--ic-muted);
    margin-top: 0.15rem;
  }
  .ic-card {
    background: var(--ic-card);
    border: 1px solid var(--ic-border);
    border-radius: 20px;
    padding: 1.25rem 1.35rem;
    box-shadow: var(--ic-shadow);
  }
  .ic-score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(125, 211, 252, 0.25);
    font-size: 0.92rem;
  }
  .ic-score-row:last-child { border-bottom: none; }
  .ic-review-block {
    margin: 0.85rem 0;
    padding-left: 0.85rem;
    border-left: 3px solid rgba(45, 106, 159, 0.35);
  }
  .ic-review-title {
    font-weight: 600;
    color: var(--ic-navy);
    font-size: 0.9rem;
    margin-bottom: 0.35rem;
  }
  .ic-overall {
    margin-top: 1rem;
    padding: 1rem;
    background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
    border-radius: 14px;
    border: 1px solid rgba(125, 211, 252, 0.35);
    line-height: 1.65;
    color: var(--ic-text);
  }
  .ic-result-sheet {
    max-width: 520px;
    margin-left: auto;
    margin-right: auto;
  }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 옵션")
    provider = st.selectbox("모델", ["OpenAI (gpt-4o)", "Anthropic (claude-3.5-sonnet)"])
    max_files = st.number_input("최대 파일 수", min_value=1, max_value=50, value=20, step=1)
    max_chars = st.number_input("최대 전달 글자 수", min_value=10_000, max_value=250_000, value=120_000, step=5_000)

st.markdown('<div class="ic-hero">', unsafe_allow_html=True)
st.markdown('<div class="ic-hero-spacer"></div>', unsafe_allow_html=True)
_has_ic = os.path.exists(_LOGO_PATH)
_has_mof = os.path.exists(_LOGO_MOF_PATH)
# 표시 너비(px). 내부는 pixel_ratio배 렌더 후 st.image width로 축소(선명도)
_LOGO_DISPLAY_DUAL = 150
_LOGO_DISPLAY_SINGLE = 300
_LOGO_PIXEL_RATIO = 5.0
_LOGO_PAIR_GAP_RENDER_PX = 6

if _has_ic and _has_mof:
    _, mid, _ = st.columns([2, 6, 2])
    with mid:
        pil_mof = _load_logo_cutout(
            _LOGO_MOF_PATH,
            display_width_px=_LOGO_DISPLAY_DUAL,
            pixel_ratio=_LOGO_PIXEL_RATIO,
        )
        pil_ic = _load_logo_cutout(
            _LOGO_PATH,
            display_width_px=_LOGO_DISPLAY_DUAL,
            pixel_ratio=_LOGO_PIXEL_RATIO,
        )
        combined = _combine_logos_side_by_side(pil_mof, pil_ic, _LOGO_PAIR_GAP_RENDER_PX)
        pair_display_w = _LOGO_DISPLAY_DUAL * 2 + 8
        m1, m2, m3 = st.columns([1, 3, 1])
        with m2:
            st.image(combined, width=pair_display_w)
    st.markdown(
        '<p style="text-align:center;font-size:0.85rem;color:#4a6b82;margin:0.4rem 0 0 0;'
        'letter-spacing:0.03em;font-weight:500;">재정경제부 · Innocurve AI</p>',
        unsafe_allow_html=True,
    )
elif _has_ic:
    st.markdown('<div class="ic-logo-wrap">', unsafe_allow_html=True)
    st.image(
        _load_logo_cutout(
            _LOGO_PATH,
            display_width_px=_LOGO_DISPLAY_SINGLE,
            pixel_ratio=_LOGO_PIXEL_RATIO,
        ),
        width=_LOGO_DISPLAY_SINGLE,
    )
    st.markdown("</div>", unsafe_allow_html=True)
elif _has_mof:
    st.markdown('<div class="ic-logo-wrap">', unsafe_allow_html=True)
    st.image(
        _load_logo_cutout(
            _LOGO_MOF_PATH,
            display_width_px=_LOGO_DISPLAY_SINGLE,
            pixel_ratio=_LOGO_PIXEL_RATIO,
        ),
        width=_LOGO_DISPLAY_SINGLE,
    )
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="ic-logo-to-title-gap"></div>', unsafe_allow_html=True)
st.markdown('<h1 class="ic-main-title">인공지능 심사위원</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="ic-tagline">PPT(.pptx)·PDF(.pdf)·코드를 올리면 5개 항목(각 20점)으로 채점하고, 총점·항목별 점수·레이더 차트로 확인할 수 있습니다.</p>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<p class="ic-panel-title" style="margin-top:0.5rem;">자료 업로드</p>', unsafe_allow_html=True)
bundle = _collect_uploads()

ok, key_name = _require_api_key(provider)
st.markdown('<div class="ic-stat-row">', unsafe_allow_html=True)
st.markdown(
    f'<div class="ic-stat"><div class="ic-stat-num">{len(bundle.pptx_files)}</div>'
    f'<div class="ic-stat-label">PPT (.pptx)</div></div>'
    f'<div class="ic-stat"><div class="ic-stat-num">{len(bundle.pdf_files)}</div>'
    f'<div class="ic-stat-label">PDF</div></div>'
    f'<div class="ic-stat"><div class="ic-stat-num">{len(bundle.code_files)}</div>'
    f'<div class="ic-stat-label">코드·기타</div></div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

if not ok:
    st.error(f"{key_name} 키가 필요합니다.")
run_clicked = st.button("심사 시작", type="primary", disabled=not ok, use_container_width=True)


if run_clicked:
    if not bundle.pptx_files and not bundle.pdf_files and not bundle.code_files:
        st.warning("파일을 업로드해 주세요.")
        st.stop()

    with st.spinner("파일에서 텍스트/신호를 추출하고 있어요..."):
        ppt_texts = [extract_ppt_text(f) for f in bundle.pptx_files[: int(max_files)]]
        pdf_texts: List[str] = []
        for f in bundle.pdf_files[: int(max_files)]:
            try:
                pdf_texts.append(extract_pdf_text(f))
            except ImportError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.warning(f"PDF 텍스트 추출 실패 ({f.name}): {e}")
        ppt_merged = "\n\n".join(t for t in ppt_texts if t).strip()
        pdf_merged = "\n\n".join(t for t in pdf_texts if t).strip()
        deck_merged = "\n\n".join(x for x in [ppt_merged, pdf_merged] if x).strip()
        ppt_signals = extract_ppt_signals(deck_merged) if deck_merged else {}

        code_analysis = analyze_code_files(bundle.code_files[: int(max_files)])

        sections = [
            ("[발표자료 전체 텍스트 (PPT·PDF)]", deck_merged),
            ("[발표자료 파급효과 우선 신호(문제 정의/시장 규모)]", json.dumps(ppt_signals, ensure_ascii=False)),
            ("[코드 분석(라이브러리/복잡도/구조)]", json.dumps(code_analysis, ensure_ascii=False)),
        ]
        sections = _truncate_sections(sections, int(max_chars))
        context_blob = "\n\n".join(f"{t}\n{c}" for t, c in sections if c)

    with st.spinner("LLM이 심사 중이에요..."):
        cfg = JudgeLlmConfig(provider=provider, temperature=0.1)
        raw = run_judgement(context_blob=context_blob, cfg=cfg)

    with st.spinner("결과를 구조화하고 있어요..."):
        try:
            result = JudgeResult.model_validate_json(raw)
        except Exception:
            try:
                result = JudgeResult.model_validate(json.loads(raw))
            except Exception:
                st.error("모델 출력 형식을 해석하지 못했습니다. 잠시 후 다시 시도해 주세요.")
                st.stop()

    # 총점은 20점 만점(5개 항목 평균). 모델이 실수로 100점 스케일로 내보내도 방어합니다.
    item_scores = [
        int(result.scores.innovation),
        int(result.scores.difficulty),
        int(result.scores.design),
        int(result.scores.deck_quality),
        int(result.scores.impact),
    ]
    avg20 = int(round(sum(item_scores) / 5.0))
    total = int(result.total_score)
    if total > 20:
        total = avg20
    total = max(0, min(20, total))
    grade = result.final_grade or _grade_from_total(total)

    st.divider()
    st.subheader("심사 결과")

    score_map = {
        "아이디어 혁신성": result.scores.innovation,
        "개발 난이도": result.scores.difficulty,
        "디자인": result.scores.design,
        "발표자료 완성도": result.scores.deck_quality,
        "파급효과": result.scores.impact,
    }

    st.markdown(_result_sheet_html(total, grade, score_map), unsafe_allow_html=True)

    st.caption("레이더 차트(방사형) · 5개 항목 균형")
    _, col_radar, _ = st.columns([1, 2, 1])
    with col_radar:
        fig = _radar_chart(score_map)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

