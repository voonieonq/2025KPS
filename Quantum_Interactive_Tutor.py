import os
import streamlit as st
import textwrap
from pathlib import Path
import sys
import importlib
import importlib.util
import runpy
import types
from textwrap import dedent
import streamlit.components.v1 as components
from urllib.parse import quote

APP_ROOT = Path(__file__).resolve().parent
PAGES_DIR = APP_ROOT / "pages"

# 1) Streamlit secrets -> 환경변수 주입
if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ.setdefault("LLM_MODEL", str(st.secrets.get("LLM_MODEL", "gpt-4o-mini")))
os.environ.setdefault("LLM_MAX_TOKENS", str(st.secrets.get("LLM_MAX_TOKENS", 180)))

# 2) 세션별 호출 제한
MAX_FREE = int(st.secrets.get("MAX_FREE_CALLS", 15))
if "call_count" not in st.session_state:
    st.session_state.call_count = 0

# 3) 한도 초과 시 사용자 키 받기
if st.session_state.call_count >= MAX_FREE and not st.session_state.get("user_key_ok"):
    st.info(
        "무료 사용 한도를 초과했어요. 본인 OpenAI API 키를 입력하면 계속 이용할 수 있어요.\n"
        "입력한 키는 브라우저 세션에서만 사용되며 서버에 저장하지 않습니다."
    )
    user_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")

    if user_key:
        st.session_state["user_api_key"] = user_key.strip()   # 세션 전용 저장
        st.session_state["user_key_ok"] = True
        os.environ["OPENAI_API_KEY"] = st.session_state["user_api_key"]

        try:
            import importlib
            import quantum_ai as _qai_reload
            importlib.reload(_qai_reload)
            globals()["HAVE_QAI"] = True
        except Exception:
            pass

        st.success("개인 키가 설정되었습니다. 계속 이용할 수 있어요!")


SESSION_DEFAULTS = {
    "user_profile": {},
    "quiz_history": {},
    "review_mode": False,
    "current_keyword": None,

    # 오답 복습용 상태
    "review_all_active": False,
    "review_all_stack": [],
    "review_all_idx": 0,
    "review_all_last_answer": None,

    # 단계 승급(초급자→중급자→전문가)
    "level_state": {},  # {canonical: {"stage":"초급자","streak":0}}
}

SESSION_DEFAULTS.update({
    "review_kw_active": False,
    "review_kw_groups": {},
    "review_kw_keys": [],
    "review_kw_selected": None,
    "review_kw_idx": 0,
    "review_kw_last_answer": None,
})

for k, v in SESSION_DEFAULTS.items():
    st.session_state.setdefault(k, v)

# =========================
# 환경/경로
# =========================
# 모듈 탐색 경로 고정
sys.path.insert(0, os.path.dirname(__file__))

# ── 개념/키워드 모듈 ──
import quantum_modules as qm
concepts = qm.concepts
keyword_map = qm.keyword_map
SIM_REGISTRY = qm.SIM_REGISTRY

# ── AI 모듈 임포트 ──
try:
    import quantum_ai as qai
    from quantum_ai import assess_user_level_ai_wrapped
    HAVE_QAI = True
    QAI_PATH = getattr(qai, "__file__", "?")
    QAI_VERSION = getattr(qai, "QAI_VERSION", "unknown")
except Exception as e:
    HAVE_QAI = False
    QAI_PATH = "import-failed"
    QAI_VERSION = "unknown"
    QAI_IMPORT_ERR = str(e)

# === 단계 승급 설정 ===
PROMO_STREAK = 3  # 3연속 정답 시 승급

# 세션 초기화
if "level_state" not in st.session_state:
    # per-topic: {"stage": "초급자"|"중급자"|"전문가", "streak": int}
    st.session_state.level_state = {}

def _get_stage(canonical: str) -> str:
    st_level = st.session_state.level_state.setdefault(canonical, {"stage":"초급자","streak":0})
    return st_level["stage"]

def _get_streak(canonical: str) -> int:
    st_level = st.session_state.level_state.setdefault(canonical, {"stage":"초급자","streak":0})
    return st_level["streak"]

def _record_result_and_maybe_promote(canonical: str, is_correct: bool) -> None:
    st_level = st.session_state.level_state.setdefault(canonical, {"stage":"초급자","streak":0})
    if is_correct:
        st_level["streak"] += 1
        if st_level["stage"] == "초급자" and st_level["streak"] >= PROMO_STREAK:
            st_level["stage"], st_level["streak"] = "중급자", 0
        elif st_level["stage"] == "중급자" and st_level["streak"] >= PROMO_STREAK:
            st_level["stage"], st_level["streak"] = "전문가", 0
    else:
        st_level["streak"] = 0  # 강등 없음

# =========================
# 유틸
# =========================

def _overall_level_from_ratio(r: float) -> str:
    """간단 기준: 0~59 초급자, 60~89 중급자, 90~100 전문가"""
    if r >= 0.90:
        return "전문가"
    if r >= 0.60:
        return "중급자"
    return "초급자"

def assess_level(canonical: str):
    """가능하면 AI 분석, 실패 시 로컬 보정."""
    if HAVE_QAI:
        try:
            return assess_user_level_ai_wrapped(
                canonical,
                st.session_state.user_profile,
                st.session_state.quiz_history,
            )
        except Exception:
            pass
    prof = st.session_state.user_profile.get(canonical, {})
    return {"level_hint": prof.get("level_text", "중급자"), "weak_points": []}

def generate_quiz_adaptive(canonical: str):
    """현재 단계(stage)에 맞춘 1문항 생성."""
    st.session_state["ai_last_error"] = None
    st.session_state["ai_last_len"] = 0

    if not HAVE_QAI:
        st.session_state["ai_last_error"] = f"quantum_ai import 실패: {globals().get('QAI_IMPORT_ERR','?')}"
        return []

    try:
        level_hint = _get_stage(canonical)  # ★ 초급자/중급자/전문가
        data = concepts.get(canonical, {})
        concept_blob = data if isinstance(data, dict) else {"개념 설명": str(data)}
        items = qai.generate_quiz_ai(  # assess 호출 안 함!
            canonical,
            concept_blob,
            level_hint,
            st.session_state.quiz_history,
        )
        st.session_state["ai_last_len"] = len(items) if isinstance(items, list) else 0
        return items if isinstance(items, list) else []
    except Exception as e:
        st.session_state["ai_last_error"] = f"{type(e).__name__}: {e}"
        return []

def _norm(s: str) -> str:
    return "".join((s or "").strip().lower().split())

def resolve_keyword(raw: str):
    norm_map = {_norm(k): v for k, v in keyword_map.items()}
    k = _norm(raw or "")
    if k in norm_map:
        return norm_map[k]
    for c in concepts.keys():
        if _norm(c) == k:
            return c
    return None

def _md(text: str) -> str:
    s = textwrap.dedent(text or "").strip()
    return s.replace("\n", "  \n")

def _sanitize_quiz_items(items):
    """AI에서 온 문제 리스트를 안전형으로 보정 (항상 객관식, 최대 4지)."""
    safe = []
    if not isinstance(items, (list, tuple)):
        return safe
    for i, q in enumerate(items):
        if not isinstance(q, dict):
            continue
        question = (q.get("question") or "").strip()
        options = q.get("options") or []
        answer = q.get("answer")
        qid = q.get("id") or f"aiq_{i+1}"
        if not isinstance(options, (list, tuple)):
            options = [str(options)]
        options = [str(o).strip() for o in options if str(o).strip()][:4]
        if not question or len(options) == 0:
            continue
        if answer is None or str(answer) not in options:
            answer = options[0]
        safe.append({
            "question": question,
            "options": options,
            "answer": str(answer),
            "id": qid,
        })
    return safe


# =========================
# 스타일: 관련 개념 칩
# =========================
st.markdown("""
<style>
#related-chips .stButton { display:inline-block; margin: .25rem .35rem .25rem 0; }
#related-chips .stButton > button{
  border:1px solid rgba(255,255,255,.18);
  background:rgba(255,255,255,.06);
  border-radius:9999px;
  padding:.35rem .75rem;
  font-size:.95rem;
  line-height:1.2;
  display:inline-block;
}
#related-chips .stButton > button:hover{
  background:rgba(255,255,255,.12);
  border-color:rgba(255,255,255,.35);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
  [data-testid="stSidebarNav"] { display: none; }
  [data-testid="stSidebarNavSeparator"] { display: none; }
</style>
""", unsafe_allow_html=True)

# =========================
# UI: 타이틀/키워드 입력
# =========================
st.title("AI 기반 양자역학 튜터")
st.subheader("학습 키워드를 입력해보세요!")

if "kw_input" not in st.session_state:
    st.session_state.kw_input = ""

if "kw_jump" in st.session_state:
    st.session_state.kw_input = st.session_state.kw_jump
    st.session_state.current_keyword = st.session_state.kw_jump
    del st.session_state["kw_jump"]

if not st.session_state.get("review_mode"):
    keyword = st.text_input("예: 양자 얽힘, 양자 터널링, 양자 컴퓨터 ···", key="kw_input")
    if keyword:
        st.session_state.current_keyword = keyword
else:
    keyword = st.session_state.current_keyword

# =========================
# 사이드바: 내 학습 리포트
# =========================
with st.sidebar:
    st.header("내 학습 리포트")

    profile = st.session_state.get("user_profile", {})
    history = st.session_state.get("quiz_history", {})

    if profile:
        total_q = sum(int(v.get("total", 0)) for v in profile.values())
        total_correct = sum(int(v.get("score", 0)) for v in profile.values())
        overall = (total_correct / total_q) if total_q else 0.0

        c1, c2 = st.columns([2, 1])
        with c1:
            st.metric("전체 정답률", f"{overall*100:.0f}%")
            st.progress(overall)
        with c2:
            st.write("**내 수준**")
            if total_q >= 3:
                st.subheader(_overall_level_from_ratio(overall))
            else:
                st.caption("3문제 이상 풀면 표시됩니다.")

        st.markdown("---")
        st.subheader("키워드별 결과")

        # ── 키워드별 점수/정답률만 표시
        for canonical_key, v in profile.items():
            score = int(v.get("score", 0))
            total = int(v.get("total", 0))
            pct = (score / total) if total else 0.0

            st.markdown(f"**{canonical_key}**")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"- 점수: {score} / {total}")
            with col2:
                st.write(f"{pct*100:.0f}%")
            st.progress(pct)

        # ── 전체 AI 분석 요약
        from collections import Counter

        st.markdown("---")
        st.subheader("AI 분석 요약")

        all_levels = []
        all_wp, all_nf = [], []

        for canonical_key in profile.keys():
            try:
                lvl = assess_level(canonical_key) or {}
            except Exception:
                lvl = {}
            if lvl.get("level_hint"):
                all_levels.append(lvl["level_hint"])
            all_wp.extend(lvl.get("weak_points") or [])

        if all_levels:
            level_summary = Counter(all_levels).most_common(1)[0][0]
        else:
            level_summary = "-"

        # 약점
        weak_all = [w for w in (all_wp + all_nf) if w]
        weak_all = list(dict.fromkeys(weak_all))  # preserve order & dedup

        if weak_all:
            st.write("· 약점: " + ", ".join(weak_all))
        else:
            st.write("데이터를 더 모으면 약점을 분석할 수 있어요")

        st.markdown("---")
        b1, b2 = st.columns(2)

        with b1:
            if st.button("오답 복습"):
                groups = {}
                for cano, qlist in history.items():
                    for idx, q in enumerate(qlist):
                        if q.get("is_correct"):
                            continue
                        opts = q.get("options") or []
                        ca = q.get("correct_answer")
                        groups.setdefault(cano, []).append({
                            "question": q.get("question", ""),
                            "options": opts or ([ca] if ca else []),
                            "answer": ca,
                            "id": f"kwrev_{cano}_{idx}",
                        })

                if not groups:
                    st.info("푼 퀴즈의 오답이 없습니다.")
                else:
                    st.session_state.review_kw_active = True
                    st.session_state.review_kw_groups = groups
                    st.session_state.review_kw_keys = list(groups.keys())
                    st.session_state.review_kw_selected = st.session_state.review_kw_keys[0]
                    st.session_state.review_kw_idx = 0
                    st.session_state.review_kw_last_answer = None
                    st.session_state.review_mode = False
                    st.rerun()

        with b2:
            if st.button("초기화"):
                st.session_state.user_profile = {}
                st.session_state.quiz_history = {}
                st.session_state.review_mode = False

                # 복습 상태 리셋
                st.session_state.review_kw_active = False
                st.session_state.review_kw_groups = {}
                st.session_state.review_kw_keys = []
                st.session_state.review_kw_selected = None
                st.session_state.review_kw_idx = 0
                st.session_state.review_kw_last_answer = None

                st.session_state.level_state = {}
                st.rerun()

    else:
        st.info("아직 푼 퀴즈가 없어요.  \n키워드를 입력하고 퀴즈를 풀면 내 학습 리포트가 나타납니다.")

# =========================
# 오답 복습 UI
# =========================
def render_review_by_keyword():
    groups = st.session_state.get("review_kw_groups", {})
    keys = st.session_state.get("review_kw_keys", [])
    if not groups or not keys:
        st.info("복습할 오답이 없습니다.")
        st.session_state.review_kw_active = False
        st.session_state.review_kw_selected = None
        st.session_state.review_kw_idx = 0
        return

    # 키워드 선택
    current_key = st.session_state.get("review_kw_selected") or keys[0]
    current_key = st.selectbox(
        "키워드 선택",
        options=keys,
        index=keys.index(current_key) if current_key in keys else 0,
        key="rev_kw_select"
    )
    st.session_state.review_kw_selected = current_key

    stack = groups.get(current_key, [])
    if not stack:
        st.info("이 키워드에서 복습할 오답이 없습니다.")
        return

    idx = max(0, min(st.session_state.get("review_kw_idx", 0), len(stack) - 1))
    st.session_state.review_kw_idx = idx

    q = stack[idx]
    question = q.get("question", "")
    options = q.get("options") or []
    answer = q.get("answer")

    st.markdown(f"### 오답 복습 — **{current_key}** ({idx+1}/{len(stack)})")
    st.write(f"**문제:** {question}")

    if not options:
        options = [answer] if answer else ["(선택지 없음)"]
    if len(options) == 1 and answer and answer not in options:
        options = list(dict.fromkeys(options + [answer]))

    with st.form(f"review_kw_form_{current_key}_{idx}"):
        sel = st.radio("선택지", options, key=f"rev_kw_sel_{current_key}_{idx}")
        submitted = st.form_submit_button("정답 확인")
        if submitted:
            st.session_state.review_kw_last_answer = sel

    last = st.session_state.get("review_kw_last_answer")
    if last is not None:
        if last == answer:
            _ = st.success("정답입니다!")
        else:
            _ = st.error(f"아쉬워요. 정답은 '{answer}' 입니다.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("⬅ 이전 문제"):
            st.session_state.review_kw_idx = (idx - 1) % len(stack)
            st.session_state.review_kw_last_answer = None
            st.rerun()
    with c2:
        if st.button("다음 문제 ➡"):
            st.session_state.review_kw_idx = (idx + 1) % len(stack)
            st.session_state.review_kw_last_answer = None
            st.rerun()
    with c3:
        # 이전 키워드
        if st.button("◀ 이전 키워드"):
            pos = keys.index(current_key)
            st.session_state.review_kw_selected = keys[(pos - 1) % len(keys)]
            st.session_state.review_kw_idx = 0
            st.session_state.review_kw_last_answer = None
            st.rerun()
    with c4:
        # 다음 키워드
        if st.button("다음 키워드 ▶"):
            pos = keys.index(current_key)
            st.session_state.review_kw_selected = keys[(pos + 1) % len(keys)]
            st.session_state.review_kw_idx = 0
            st.session_state.review_kw_last_answer = None
            st.rerun()

    st.markdown("---")
    if st.button("복습 종료"):
        st.session_state.review_kw_active = False
        st.session_state.review_kw_selected = None
        st.session_state.review_kw_idx = 0
        st.session_state.review_kw_last_answer = None
        st.success("오답 복습을 종료했습니다.")

# =========================
# 리뷰 모드 우선 처리
# =========================
if st.session_state.get("review_kw_active", False):
    render_review_by_keyword()
    st.stop()

# =========================
# 키워드 처리 / 개념 설명
# =========================
if not (st.session_state.get("current_keyword") or ""):
    st.stop()

keyword = st.session_state.current_keyword
st.markdown(f"### 학습 키워드: {keyword}")
canonical = resolve_keyword(keyword)
if not canonical:
    st.warning(f"'{keyword}'(을)를 찾지 못했어요.")
    st.stop()

# 개념 설명 섹션
cdata = concepts.get(canonical, {})
concept_parts = {"개념 설명": cdata} if isinstance(cdata, str) else (cdata or {})

with st.expander("개념 설명", expanded=True):
    st.markdown(_md(concept_parts.get("개념 설명")))
if concept_parts.get("예시"):
    with st.expander("예시"):
        st.markdown(_md(concept_parts.get("예시")))
if concept_parts.get("특징 정리"):
    with st.expander("특징 정리"):
        st.markdown(_md(concept_parts.get("특징 정리")))

# (5) 관련 개념 섹션 복구 + 칩 스타일
related = concept_parts.get("관련 개념")
if related:
    with st.expander("관련 개념"):
        if isinstance(related, str):
            items = [s.strip() for s in related.replace("，", ",").split(",") if s.strip()]
            pairs = [(x, x) for x in items]
        elif isinstance(related, (list, tuple, set)):
            pairs = [(str(x), str(x)) for x in related]
        elif isinstance(related, dict):
            # label → target 매핑 지원
            pairs = [(str(k), str(v)) for k, v in related.items()]
        else:
            pairs = []

        st.markdown("<div id='related-chips'>", unsafe_allow_html=True)
        for i, (label, target) in enumerate(pairs):
            # 버튼 라벨에 타깃이 다르면 '라벨 (→타깃)' 형태로 힌트 부여
            show = label if label == target else f"{label} (→ {target})"
            if st.button(show, key=f"rel_{canonical}_{i}"):
                goto = resolve_keyword(target) or target
                st.session_state.kw_jump = goto
                st.session_state.review_mode = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 시뮬레이터 (멀티페이지 링크 방식)
# =========================

def _normalize_item(item):
    if isinstance(item, (list, tuple)):
        if len(item) == 2:
            return item[0], item[1], ""
        elif len(item) >= 3:
            return item[0], item[1], item[2]
    raise ValueError("Invalid SIM_REGISTRY item: expected 2- or 3-tuple")

def _sim_rel_path(mod_name: str) -> str | None:
    """앱 루트 기준 상대경로('pages/xxx.py')를 반환"""
    rel = Path("pages") / f"{mod_name}.py"
    # 실제 파일 존재 확인은 __file__ 기준으로
    if (Path(__file__).parent / rel).exists():
        return str(rel).replace("\\", "/")  # Windows 대비
    return None

# === 모든 시뮬레이터 공통 실행 유틸 ===

def _canon(name: str) -> str:
    # 공백/대시 → 언더스코어로 표준화
    return name.strip().replace("-", "_").replace(" ", "_")

def _try_import_by_name(name: str) -> types.ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def _try_import_by_path(pyfile: Path) -> types.ModuleType | None:
    if not pyfile.is_file():
        return None
    spec = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    except Exception as e:
        st.exception(e)
        return None

def _run_entry(mod) -> bool:
    """모듈 내부의 엔트리 함수를 찾아 실행. 없으면 파일 자체를 스크립트처럼 실행."""
    # 1) 엔트리 함수 우선
    for fn in ("main", "app", "run", "render", "st_app"):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            getattr(mod, fn)()
            return True

    # 2) 엔트리 함수가 없으면, 파일 경로를 얻어서 스크립트 실행
    from pathlib import Path
    import runpy

    mod_path = Path(getattr(mod, "__file__", ""))
    if mod_path.is_file():
        # __name__을 '__main__'으로 둔 상태로 실행 → if __name__ == "__main__": 블록도 동작
        runpy.run_path(str(mod_path))
        return True

    return False

def run_external_sim(mod_name: str, title: str, desc: str = "") -> None:
    """
    시뮬레이터 범용 실행기:
    - 이름으로 import 시도
    - 실패하면 루트/src/tools에서 .py 파일 직접 로드
    - 엔트리 함수(main/app/run/render/st_app) 있으면 호출, 없으면 top-level 실행
    """
    name = _canon(mod_name)

    # 1) 패키지/모듈 import
    mod = _try_import_by_name(name)
    if mod is None:
        # 2) 파일 경로로 로드 시도
        candidates = [
            APP_ROOT / f"{name}.py",          # 프로젝트 루트
            APP_ROOT / "src" / f"{name}.py",  # src 폴더
            APP_ROOT / "tools" / f"{name}.py" # tools 폴더
        ]
        for p in candidates:
            mod = _try_import_by_path(p)
            if mod is not None:
                break

    if mod is not None:
        with st.container(border=True):
            st.markdown(f"### {title}")
            if desc:
                st.markdown(desc)

        ran = _run_entry(mod)
        if not ran:
            st.error("시뮬레이터 엔트리를 찾지 못했고, 스크립트 실행에도 실패했습니다.")
        return

    # 3) 끝으로 runpy로 스크립트 실행
    for p in [
        APP_ROOT / f"{name}.py",
        APP_ROOT / "src" / f"{name}.py",
        APP_ROOT / "tools" / f"{name}.py",
    ]:
        if p.is_file():
            with st.container(border=True):
                st.markdown(f"### {title}")
                if desc:
                    st.markdown(desc)
            try:
                runpy.run_path(str(p))
                return
            except Exception as e:
                st.exception(e)
                return

    st.error(f"시뮬레이터 '{mod_name}'를 찾을 수 없습니다. "
             f"({APP_ROOT}/{{{name}.py, src/{name}.py, tools/{name}.py}} 확인)")




def render_sim_entry(mod_name: str, title: str, desc: str):
    abs_page = PAGES_DIR / f"{mod_name}.py"
    btn_label = f"{title} (새 탭)" if abs_page.is_file() else f"▶ {title} (직접 실행)"
    if abs_page.is_file():
        newtab_button_for_page(mod_name, btn_label)
    else:
        if st.button(btn_label, key=f"btn_{mod_name}", use_container_width=True):
            run_external_sim(mod_name, title, desc="")
    if desc:
        st.markdown(dedent(desc).strip().replace("\n", "  \n"))
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def _router_label_from_file(filename="00_router.py") -> str:
    # "00_router.py" -> "00 router"
    stem = (PAGES_DIR / filename).stem
    return stem.replace("_", " ")

def newtab_button_for_page(mod_name: str, label: str, *, height: int = 54) -> None:
    import streamlit.components.v1 as components
    from urllib.parse import quote

    base = "window.location.origin + window.location.pathname"
    if (PAGES_DIR / "00_router.py").is_file():
        router_label = _router_label_from_file("00_router.py")
        # 라우터 성공 또는 메인 우회 모두 잡도록 두 파라미터 동시 세팅
        url_js = f'{base} + "?page={quote(router_label)}&target={quote(mod_name)}&run={quote(mod_name)}"'
    else:
        # 라우터 없어도 동작
        url_js = f'{base} + "?run={quote(mod_name)}"'

    btn_id = f"open_{mod_name}"
    components.html(f"""
      <div>
        <button id="{btn_id}"
                style="width:100%;padding:.6rem 1rem;border-radius:10px;
                       background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.2);
                       cursor:pointer;">
          {label}
        </button>
        <script>
          (function(){{
            const url = {url_js};
            document.getElementById("{btn_id}").addEventListener("click", function(){{
              window.open(url, "_blank", "noopener");
            }});
          }})();
        </script>
      </div>
    """, height=height)

def _get_qp():
    try:
        return st.query_params
    except Exception:
        return st.experimental_get_query_params()

def _switch_or_run_direct(target: str):
    """pages/<target>.py 가 있으면 라우팅, 없으면 외부 실행기로 직접 실행"""
    if (PAGES_DIR / f"{target}.py").is_file():
        try:
            st.switch_page(f"pages/{target}.py")
            return
        except Exception:
            pass
    # pages에 없으면 프로젝트 루트/src/tools에서 직접 실행
    st.markdown(f"### {target} 실행 (직접 모드)")
    run_external_sim(target, target, desc="")
    st.stop()

# 새 탭 진입 시 바로 처리 (라우터 없어도 동작)
_qp = _get_qp()
_direct = (_qp.get("run") or [""])[0] or (_qp.get("target") or [""])[0]
if _direct:
    _switch_or_run_direct(_direct)
    st.stop()

with st.expander("시뮬레이터", expanded=False):
    sims = SIM_REGISTRY.get(canonical, []) or []
    if not sims:
        st.caption("이 키워드는 준비된 시뮬레이터가 없어요.")
    else:
        for it in sims:
            mod_name, title, desc = _normalize_item(it)  # (2/3튜플 정규화 함수)
            render_sim_entry(mod_name, title, desc)

# =========================
# 개념 퀴즈 (객관식 1문항씩)
# =========================
st.markdown("### 개념 퀴즈 <span style='font-size:13px; color:gray;'>(Powered by OpenAI GPT-4o-mini)</span>", unsafe_allow_html=True)
_current_q_key = f"current_q_{canonical}"

def _load_new_question():
    items = generate_quiz_adaptive(canonical)
    items = _sanitize_quiz_items(items)
    st.session_state[_current_q_key] = items[0] if items else None

# AI가 실제로 새 문항을 만들었다면 1회 사용으로 카운트
    if items:
        st.session_state["call_count"] = st.session_state.get("call_count", 0) + 1

if _current_q_key not in st.session_state or st.button("새 문제 받기", key=f"regen_{canonical}", use_container_width=True):
    _load_new_question()

q = st.session_state.get(_current_q_key)
if not q:
    st.info("퀴즈가 준비되지 않았습니다. ‘새 문제 받기’를 눌러보세요.")

    last_len = st.session_state.get("ai_last_len")
    last_err = st.session_state.get("ai_last_error")
    st.caption(f"AI 모듈: {QAI_VERSION} ({QAI_PATH}) | HAVE_QAI={HAVE_QAI} | last_len={last_len} | last_err={last_err}")

    if HAVE_QAI and hasattr(qai, "debug_last"):
        dbg = qai.debug_last()
        with st.expander("모델 응답 디버그", expanded=False):
            st.write(f"stage = {dbg.get('stage')}, model = {dbg.get('model')}, ok = {dbg.get('ok')}")
            if dbg.get("err"):
                st.error(f"error: {dbg['err']}")
            st.text_area("raw (model output)", dbg.get("raw") or "", height=160)
            st.text_area("text (after strip_to_json_like)", dbg.get("text") or "", height=160)
else:
    picked = st.radio(q["question"], q["options"], key=f"ans_{q['id']}")
    if st.button("제출", key=f"submit_{q['id']}", use_container_width=True):
        correct = (picked == q["answer"])
        _ = st.success("정답입니다!") if correct else st.error(f"오답입니다. 정답: '{q['answer']}' 입니다.")


        # 히스토리 누적
        hist_list = st.session_state.quiz_history.setdefault(canonical, [])
        hist_list.append({
            "question": q["question"],
            "options": q["options"],
            "correct_answer": q["answer"],
            "user_answer": picked,
            "is_correct": correct,
        })

        # 프로필 누적 갱신
        prof = st.session_state.user_profile.setdefault(canonical, {"score": 0, "total": 0})
        prof["total"] += 1
        if correct:
            prof["score"] += 1
        ratio = (prof["score"] / prof["total"]) if prof["total"] else 0.0
        prof["level_percent"] = int(ratio * 100)
        prof["level_text"] = _overall_level_from_ratio(ratio)

        # 단계 승급 로직 반영
        _record_result_and_maybe_promote(canonical, correct)

        # 다음 문제 즉시 로드
        _load_new_question()
        st.rerun()

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 13px;
        color: gray;
    }
    </style>
    <div class="footer">
        Developed by Kim Yoonjeong & Ji Hyeeun<br>
        Department of Physics, Soongsil University
    </div>
    """,
    unsafe_allow_html=True
)
