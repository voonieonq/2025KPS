
import os, json, time, random
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import logging
import random
from quantum_modules import SIM_REGISTRY

QAI_VERSION = "single-question-1.1"

# === debug state ===
_DEBUG_STATE = {"ok": None, "model": None, "raw": None, "text": None, "err": None, "stage": None}
def _set_debug(**kw): _DEBUG_STATE.update(kw)
def debug_last(): return dict(_DEBUG_STATE)

def _extract_output_text(resp) -> str:
    """OpenAI Responses 객체에서 텍스트를 안전 추출."""
    try:
        t = getattr(resp, "output_text", None)
        if t:
            return str(t)
    except Exception:
        pass
    try:
        return resp.output[0].content[0].text.value
    except Exception:
        return ""

def _strip_to_json_like(s: str) -> str:
    """코드펜스/프리앰블 제거 후 JSON 본문만 남긴다."""
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    firsts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if firsts:
        f = min(firsts)
        if f > 0:
            s = s[f:]
    last = max(s.rfind("}"), s.rfind("]"))
    if last != -1:
        s = s[: last + 1]
    return s.strip()

# === logging ===
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
LOG_FILE  = os.getenv("APP_LOG_FILE")  # 예: "app.log"
_handlers = [logging.StreamHandler()]
if LOG_FILE:
    _handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger("quantum_ai")
os.environ.setdefault("OPENAI_LOG", os.getenv("OPENAI_LOG", ""))  # "debug" 주면 상세

# ─────────────────────────────────────────────
# (1) 주제 힌트 사전: canonical → 설명
# ─────────────────────────────────────────────
TOPIC_HINTS: Dict[str, str] = {
    "entanglement": "양자 얽힘(벨 상태, CHSH 부등식, 비국소 상관)",
    "wave_particle_duality": "입자-파동 이중성(이중 슬릿 간섭, 경로 정보와 간섭의 상쇄)",
    "tunneling": "양자 터널링(포텐셜 장벽, 투과확률, 파동함수 감쇠)",
    "quantum_teleportation": "양자 텔레포테이션(벨 측정, 클래식 통신, 상태 전송)",
    "quantum_computer": "양자 컴퓨팅(큐비트, 양자 회로, 게이트: X/H/CNOT 등)",
    "superposition": "양자 중첩(기저중첩, 위상, 측정시 붕괴)",
    "uncertainty": "불확정성(표준편차, 위치-운동량, 측정 한계)",
    "bloch_sphere": "블로흐 구(단일 큐비트 상태, 극좌표, 위상)",
    "quantum_gate": "양자 게이트(단일/이큐비트, 유니터리, X/H/CNOT, 회전 게이트)",
}

# ─────────────────────────────────────────────
# (2) 난이도 스펙: level_hint → 생성 가이드
# ─────────────────────────────────────────────
LEVEL_SPECS: Dict[str, Dict[str, Any]] = {
    "초급자": {
        "difficulty_tag": "easy",
        "verbs": ["정의", "식별", "기본 개념"],
        "constraints": [
            "수식 계산 요구 금지",
            "일상적 맥락 예시 허용",
            "오답은 흔한 오개념으로 구성"
        ],
        "good_patterns": ["개념 정의 고르기", "간단한 예/비례 관계 식별"],
        "question_types": ["정의형", "기본 개념 확인", "간단한 예시 식별"],
        "distractor_strategy": "흔히 착각하는 개념을 오답으로 포함",
        "max_tokens": 120,
        "require_explanation": False,
    },
    "중급자": {
        "difficulty_tag": "medium",
        "verbs": ["적용", "추론", "정성적 비교"],
        "constraints": [
            "간단한 수치/순서 비교 허용(고1~대1)",
            "실험 설정 변화에 따른 현상 예측"
        ],
        "good_patterns": ["변수 변화에 따른 결과 판단", "측정/간섭/위상과의 연관성"],
        "question_types": ["상황 적용", "정성적 추론", "간단한 수치 비교"],
        "distractor_strategy": "조건을 일부 바꾼 상황을 오답으로 구성",
        "max_tokens": 180,
        "require_explanation": False,
    },
    "전문가": {
        "difficulty_tag": "hard",
        "verbs": ["분석", "계량적 평가", "개념 통합"],
        "constraints": [
            "기본 공식 활용한 근사 계산 허용",
            "상태표현/행렬/부등식(예: CHSH)까지 허용"
        ],
        "good_patterns": ["경계조건/한계사례 분석", "다중 개념 연동(예: 블로흐구 ↔ 게이트)"],
        "question_types": ["계산 문제", "조건 변화 분석", "복합 개념 적용"],
        "distractor_strategy": "수학적 계산 오류나 잘못된 전제에서 나온 plausible 오답",
        "max_tokens": 250,
        "require_explanation": True,
    },
}

def level_to_guide(level_hint: str) -> Dict[str, Any]:
    level = (level_hint or "중급자").strip()
    if level not in LEVEL_SPECS:
        level = "중급자"
    return LEVEL_SPECS[level]

# ─────────────────────────────────────────────
# (3) LLM 클라이언트
# ─────────────────────────────────────────────
# 싱글톤 클라이언트
_CLIENT_SINGLETON = None
def get_client():
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None:
        _CLIENT_SINGLETON = LLMClient()
    return _CLIENT_SINGLETON


class LLMClient:
    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_output_tokens = int(os.getenv("LLM_MAX_TOKENS", "180"))
        self.err: str | None = None
        self._setup()

    def _setup(self):
        self.ok = False
        try:
            # 1) 세션 키 우선 (BYOK)
            user_key = None
            try:
                import streamlit as st
                user_key = st.session_state.get("user_api_key")
            except Exception:
                pass

            # 2) env / secrets 백업
            env_key = os.getenv("OPENAI_API_KEY")
            sec_key = None
            try:
                import streamlit as st
                sec_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass

            api_key = user_key or env_key or sec_key
            if not api_key:
                self.err = "NO_API_KEY"
                logger.error(" OPENAI_API_KEY missing")
                return

            kw = {"api_key": api_key}
            if os.getenv("OPENAI_BASE_URL"): kw["base_url"] = os.getenv("OPENAI_BASE_URL")
            if os.getenv("OPENAI_ORG"):      kw["organization"] = os.getenv("OPENAI_ORG")

            self._openai = OpenAI(**kw)
            self.ok = True
            logger.info(" OpenAI client ready (model=%s)", self.model)
        except Exception as e:
            self.err = f"INIT_FAIL: {e}"
            logger.exception(" OpenAI client init failed")




    def _responses_json(self, prompt: str) -> Tuple[bool, Any]:
        """chat.completions 먼저 → 실패 시 responses 백업. 디버그 기록."""
        if not self.ok:
            _set_debug(ok=False, err=self.err, stage="none")
            return False, None

        # 1) Chat Completions (안정 경로)
        try:
            resp = self._openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            text = _strip_to_json_like(raw)
            data = json.loads(text)
            _set_debug(ok=True, model=self.model, raw=raw, text=text, err=None, stage="chat")
            return True, data
        except Exception as e:
            logger.warning("chat.completions failed → fallback to responses: %s", e)
            last_err = e

        # 2) Responses API (백업 경로)
        models = [self.model] + [m for m in ["gpt-4o-mini", "gpt-4.1-mini"] if m != self.model]
        payload = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        last_raw, last_text = "", ""
        for m in models:
            for attempt in range(2):
                try:
                    resp = self._openai.responses.create(
                        model=m,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_format={"type": "json_object"},
                        input=payload,
                    )
                    raw = _extract_output_text(resp)
                    text = _strip_to_json_like(raw)
                    if not text:
                        raise RuntimeError("EMPTY_RESPONSE")
                    data = json.loads(text)
                    _set_debug(ok=True, model=m, raw=raw, text=text, err=None, stage="responses")
                    return True, data
                except Exception as e:
                    last_err = e
                    last_raw, last_text = (locals().get("raw") or ""), (locals().get("text") or "")
                    time.sleep(0.5 * (2 ** attempt) + random.random() * 0.2)

        self.err = f"REQ_FAIL: {last_err}"
        _set_debug(ok=False, model=models[0], raw=last_raw, text=last_text, err=str(last_err), stage="none")
        return False, None

    def complete_json(self, prompt: str) -> Any:
        ok, data = self._responses_json(prompt)
        return data if ok else None

# ─────────────────────────────────────────────
# (4) 내부 유틸 (응답 정규화/보정)
# ─────────────────────────────────────────────
def _pick_list_from_json(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ["items", "questions", "quizzes", "result", "data"]:
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        for v in data.values():
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []

def _clean_options(canonical: str, options: List[str]) -> List[str]:
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    seen, dedup = set(), []
    for o in opts:
        if o not in seen:
            dedup.append(o); seen.add(o)
    opts = dedup[:4]
    while len(opts) < 4:
        if canonical == "entanglement":
            candidates = [
                "(|00⟩ + |11⟩)/√2", "(|00⟩ − |11⟩)/√2",
                "(|01⟩ + |10⟩)/√2", "(|01⟩ − |10⟩)/√2",
            ]
        elif canonical == "wave_particle_duality":
            candidates = ["경로 정보 획득", "광원 단색성 부족", "검출기 해상도 한계", "슬릿 간격 변화"]
        elif canonical == "tunneling":
            candidates = ["파동함수 감쇠", "에너지 < 장벽 높이", "확률적 투과", "정지 에너지 증가"]
        else:
            candidates = ["모두 옳다", "옳지 않다", "해당 없음", "정의와 모순"]
        for c in candidates:
            if c not in opts:
                opts.append(c); break
        if not candidates:
            break
    return opts[:4] if len(opts) >= 2 else opts

# ─────────────────────────────────────────────
# (5) 수준 평가
# ─────────────────────────────────────────────
def assess_user_level_ai(canonical_or_keyword: str, user_profile: Dict[str, Any], quiz_history: Dict[str, Any]) -> Dict[str, Any]:
    canonical = (canonical_or_keyword or "").strip()
    client = get_client()
    fallback = {"level_hint": "중급자", "weak_points": []}
    if not client.ok:
        return fallback

    prof = user_profile.get(canonical, {})
    hist = quiz_history.get(canonical, [])[:30]

    prompt = f"""
역할: 양자역학 튜터.
학습 주제: '{canonical}' ({TOPIC_HINTS.get(canonical, canonical)})

다음은 학습자의 누적 리포트와 최근 퀴즈 기록입니다.
- 요약: {json.dumps(prof, ensure_ascii=False)}
- 히스토리(최대 30): {json.dumps(hist, ensure_ascii=False)}

아래 JSON 스키마로만 출력:
{
  "level_hint": "초급자|중급자|전문가",
  "weak_points": ["..."],
}

제한:
- weak_points는 최대 3개까지만
- 각 항목은 12자 이내의 짧은 구(문장부호·불릿 금지)
- weak_points는 핵심명사구 위주(예: '개념 정의 혼동', '위상 해석 약함')로 작성
""".strip()

    data = client.complete_json(prompt)
    if not isinstance(data, dict):
        return fallback
    return {
        "level_hint": (data.get("level_hint") or "중급자"),
        "weak_points": data.get("weak_points", []) or [],
    }

# ─────────────────────────────────────────────
# (6) 객관식 1문항 생성 (키워드·수준 반영)
# ─────────────────────────────────────────────
def generate_quiz_ai(
    canonical_or_keyword: str,
    concept_blob: Dict[str, str],
    level_hint: str,
    quiz_history: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """항상 '객관식 1문항'을 생성해 리스트로 반환."""
    canonical = (canonical_or_keyword or "").strip()

    client = get_client()
    if not client.ok:
        return []

    topic_desc = TOPIC_HINTS.get(canonical, canonical)
    concept_text = "\n".join([
        concept_blob.get("개념 설명", ""),
        concept_blob.get("예시", ""),
        concept_blob.get("특징 정리", ""),
    ]).strip()

    # ── 시뮬레이터 설명 합치기 ──
    sim_descs = []
    for sim in SIM_REGISTRY.get(canonical, []):
        if len(sim) >= 3:
            sim_descs.append(sim[2])  # 세 번째 요소가 설명 텍스트

    if sim_descs:
        concept_text += "\n\n" + "\n\n".join(sim_descs)

    guide = level_to_guide(level_hint)
    verbs = ", ".join(guide["verbs"])
    constraints = "\n- ".join(["난이도 제약:" ] + guide["constraints"]) if guide.get("constraints") else ""
    good_patterns = ", ".join(guide.get("good_patterns", []))

    base = f"""
역할: 한국어로 출제하는 양자역학 튜터.
주제: '{canonical}' — {topic_desc}
대상: 고등학생~학부 초반. 현재 수준: {level_hint} (핵심 동사: {verbs})

요구사항:
- 반드시 **객관식 1문항(4지선다)**만 생성 (서술형 금지)
- 보기 간 의미 중복 금지, 문장은 간결하게
- **정답은 보기 중 하나와 '완전 동일' 문자열이어야 함**
- 출력은 **JSON만** 허용 (코드펜스/설명 금지)
- 문제는 반드시 상기 주제와 직접 관련 (주제 이탈 금지)
- 권장 문항 형태(예시): {good_patterns if good_patterns else '개념 확인'}
{constraints}

허용 스키마(둘 중 하나):
{{
  "question":"...",
  "options":["...","...","...","..."],
  "answer":"...",
  "id":"aiq1"
}}
또는
[
  {{
    "question":"...",
    "options":["...","...","...","..."],
    "answer":"...",
    "id":"aiq1"
  }}
]

개념 설명(발췌):
\"\"\"{concept_text[:800]}\"\"\"

최근 퀴즈(중복 방지 참고):
{json.dumps(quiz_history.get(canonical, [])[:10], ensure_ascii=False)}
""".strip()

    # 1차
    data1 = client.complete_json(base)
    items = _pick_list_from_json(data1)
    if not items and isinstance(data1, dict):
        items = [data1]

    # 2차(형식 더 엄격)
    if len(items) < 1:
        tight = base + "\n\n주의: 반드시 JSON만 출력하고, 문제는 1개만 생성하세요."
        data2 = client.complete_json(tight)
        cand = _pick_list_from_json(data2) or ([data2] if isinstance(data2, dict) else [])
        if len(cand) > len(items):
            items = cand

    # 3차(초간단)
    if len(items) < 1:
        mini = (
            '객관식 1문항을 한국어 JSON으로만 출력:\n'
            '{"question":"...", "options":["...","...","...","..."], "answer":"...", "id":"aiq1"}'
        )
        data3 = client.complete_json(mini)
        items = _pick_list_from_json(data3) or ([data3] if isinstance(data3, dict) else [])

    # 보정 & 반환
    out: List[Dict[str, Any]] = []
    for q in items:
        if not isinstance(q, dict):
            continue
        question = (q.get("question") or "").strip()
        opts = _clean_options(canonical, q.get("options") or [])
        if not question or len(opts) < 2:
            continue
        ans = (q.get("answer") or "").strip()
        if ans not in opts:
            ans = opts[0]

        # --- 보기 순서를 랜덤 섞기 ---
        random.shuffle(opts)

        qid = (q.get("id") or "aiq1").strip()
        out.append({
            "question": question,
            "options": opts[:4],   # 섞인 보기
            "answer": ans,         # 문자열 그대로 유지 → 정답 비교시 문제 없음
            "id": qid
        })
        break

    if not out:
        _set_debug(ok=False, err="NO_ITEMS_FROM_MODEL", stage=_DEBUG_STATE.get("stage"))
        out = [{
            "question": f"다음 중 '{topic_desc}'에 가장 부합하는 설명은?",
            "options": ["정의", "예시", "반례", "해당 없음"],
            "answer": "정의",
            "id": "aiq_fallback",
        }]
    return out

# ─────────────────────────────────────────────
# (7) 외부 래퍼
# ─────────────────────────────────────────────
def assess_user_level_ai_wrapped(canonical_or_keyword: str, user_profile: Dict[str, Any], quiz_history: Dict[str, Any]):
    hist = quiz_history.get(canonical_or_keyword, [])
    if len(hist) < 3:
        return {"level_hint": "중급자", "weak_points": []}
    return assess_user_level_ai(canonical_or_keyword, user_profile, quiz_history)

def generate_quiz_ai_wrapped(
    canonical_or_keyword: str,
    concept_blob: Dict[str, str],
    user_profile: Dict[str, Any],
    quiz_history: Dict[str, Any],
    rule_based_fallback,  # 시그니처 유지용(미사용)
):
    level = assess_user_level_ai_wrapped(canonical_or_keyword, user_profile, quiz_history)
    items = generate_quiz_ai(canonical_or_keyword, concept_blob, level.get("level_hint", "중급자"), quiz_history)
    return items or []
