# tools/page_boot.py  (pages 밖!)
import streamlit as st
import importlib, importlib.util, runpy
from pathlib import Path

# 사이드바 기본 멀티페이지 내비게이션 숨김 (모든 페이지에 적용)
_HIDE_NAV_CSS = """
<style>
  [data-testid="stSidebarNav"] { display: none; }
  [data-testid="stSidebarNavSeparator"] { display: none; }
</style>
"""
def _hide_nav():
    st.markdown(_HIDE_NAV_CSS, unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 루트

def _try_import_by_name(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def _try_import_by_path(pyfile: Path):
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
    for fn in ("st_app", "main", "app", "run", "render"):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            getattr(mod, fn)()
            return True
    mod_path = Path(getattr(mod, "__file__", ""))
    if mod_path.is_file():
        runpy.run_path(str(mod_path))
        return True
    return False

def run_sim_page(mod_name: str, title: str, desc: str = ""):
    st.set_page_config(page_title=title, layout="wide")
    _hide_nav()  # ← 사이드바 기본 내비 숨김
    st.title(title)
    if desc:
        st.markdown(desc)

    # 1) import by name
    mod = _try_import_by_name(mod_name)

    # 2) common paths fallbacks
    if mod is None:
        for p in [ROOT / f"{mod_name}.py", ROOT / "tools" / f"{mod_name}.py", ROOT / "src" / f"{mod_name}.py"]:
            mod = _try_import_by_path(p)
            if mod is not None:
                break

    if mod is None:
        st.error(f"시뮬레이터 모듈을 찾을 수 없습니다: {mod_name}")
        return

    if not _run_entry(mod):
        st.error("시뮬레이터 엔트리를 찾지 못했고, 스크립트 실행에도 실패했습니다.")
