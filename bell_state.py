#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bell state 3D visualization
- Figure 1: 3D bar chart of |rho| (4x4)
- Figure 2: 3D surface of correlation E(theta_a, theta_b)
"""
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 등록용)

# ----- Pauli matrices -----
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# Computational basis |00>, |01>, |10>, |11>
e00 = np.array([1, 0, 0, 0], dtype=complex)
e01 = np.array([0, 1, 0, 0], dtype=complex)
e10 = np.array([0, 0, 1, 0], dtype=complex)
e11 = np.array([0, 0, 0, 1], dtype=complex)

def bell_state(name: str) -> np.ndarray:
    if name == "Φ+":
        psi = (e00 + e11) / np.sqrt(2)
    elif name == "Φ−":
        psi = (e00 - e11) / np.sqrt(2)
    elif name == "Ψ+":
        psi = (e01 + e10) / np.sqrt(2)
    elif name == "Ψ−":
        psi = (e01 - e10) / np.sqrt(2)
    else:
        raise ValueError("Unknown Bell state")
    return psi

def density_matrix(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, np.conjugate(psi))

def n_sigma(theta_deg: float) -> np.ndarray:
    """σ·n for n in X–Z plane at polar angle θ (deg) from +Z toward +X."""
    th = math.radians(theta_deg)
    return math.sin(th) * SX + math.cos(th) * SZ

def correlation(theta_a: float, theta_b: float, rho: np.ndarray) -> float:
    """E(θ_a, θ_b) = Tr[(σ·n_a ⊗ σ·n_b) ρ], real-valued."""
    A = n_sigma(theta_a)
    B = n_sigma(theta_b)
    return float(np.real(np.trace(np.kron(A, B) @ rho)))

def compute_surface(rho: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Return E grid of shape (len(thetas), len(thetas)) for all θ_a, θ_b."""
    N = len(thetas)
    E = np.empty((N, N), dtype=float)
    Abufs = [n_sigma(float(a)) for a in thetas]
    Bbufs = [n_sigma(float(b)) for b in thetas]
    for i, A in enumerate(Abufs):
        for j, B in enumerate(Bbufs):
            E[i, j] = float(np.real(np.trace(np.kron(A, B) @ rho)))
    return E

def _draw_rho_bars(ax, rho_mat):
    ax.clear()
    ax.view_init(elev=25, azim=-60)
    # 4x4 grid
    xs, ys = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')
    xpos = xs.flatten()
    ypos = ys.flatten()
    zpos = np.zeros_like(xpos, dtype=float)
    dx = dy = 0.8 * np.ones_like(xpos, dtype=float)
    dz = np.abs(rho_mat).flatten().real

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
    ax.set_zlim(0, 1.0)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["00", "01", "10", "11"])
    ax.set_yticklabels(["00", "01", "10", "11"])
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("|ρ| entry")
    ax.set_title("|ρ| for Bell state")

def render(
    state: str = "Φ+",
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    theta_step: float = 2.0,
):
    """
    Streamlit에서 호출할 엔트리 포인트.
    전달된 파라미터로 그림(2개)을 생성해서 반환한다.
    - state: "Φ+","Φ−","Ψ+","Ψ−"
    - theta_*: θ 범위/해상도
    반환: [fig_rho, fig_surf]
    """
    # 상태/밀도행렬
    psi = bell_state(state)
    rho = density_matrix(psi)

    # θ grid
    thetas = np.arange(theta_min, theta_max + 1e-9, float(theta_step))

    # --- Figure 1: |rho| bars ---
    fig_rho = plt.figure()
    ax_rho = fig_rho.add_subplot(111, projection='3d')
    _draw_rho_bars(ax_rho, rho)
    fig_rho.suptitle(f"|ρ| (magnitude) — state {state}", y=0.98)
    fig_rho.tight_layout()

    # --- Figure 2: E(theta_a, theta_b) surface ---
    fig_surf = plt.figure()
    ax_surf = fig_surf.add_subplot(111, projection='3d')
    ax_surf.view_init(elev=30, azim=-60)

    Egrid = compute_surface(rho, thetas)
    X, Y = np.meshgrid(thetas, thetas, indexing='ij')
    surf = ax_surf.plot_surface(X, Y, Egrid, linewidth=0, antialiased=True)
    ax_surf.set_xlim(theta_min, theta_max)
    ax_surf.set_ylim(theta_min, theta_max)
    ax_surf.set_zlim(-1.05, 1.05)
    ax_surf.set_xlabel("θ_a (deg)")
    ax_surf.set_ylabel("θ_b (deg)")
    ax_surf.set_zlabel("E(θ_a, θ_b)")
    ax_surf.set_title(f"Correlation surface — state {state}")
    cb = fig_surf.colorbar(surf, ax=ax_surf, shrink=0.7, pad=0.1)
    cb.set_label("E")
    fig_surf.tight_layout()

    # Streamlit 쪽에서 st.pyplot(...) 하므로 여기선 show/pyplot 호출하지 않음
    return [fig_rho, fig_surf]

# ---- 캐시: 같은 파라미터면 재계산 생략 ----
@st.cache_data(show_spinner=False)
def _cached_render(state: str, tmin: float, tmax: float, tstep: float):
    return render(state=state, theta_min=tmin, theta_max=tmax, theta_step=tstep)

def st_app() -> None:
    """Streamlit 페이지에서 호출되는 인터랙티브 엔트리."""
    st.header("벨 상태 시뮬레이터")

    # --- 컨트롤 UI ---
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        state = st.selectbox("상태 선택", ["Φ+", "Φ−", "Ψ+", "Ψ−"], index=0)
    with c2:
        theta_range = st.slider("θ 범위 (deg)", 0, 180, (0, 180), step=1)
        tmin, tmax = theta_range
    with c3:
        tstep = st.number_input("θ 간격 (deg)", min_value=1.0, max_value=30.0, value=2.0, step=1.0)

    # 과도한 격자 방지 (대충 120×120 이하)
    n_pts = int((tmax - tmin) / max(tstep, 1e-9)) + 1
    if n_pts > 200:
        st.warning(f"격자 점수가 {n_pts}개입니다. θ 간격을 키워 보세요.")
    
    # --- 그림 생성 & 렌더 ---
    figs = _cached_render(state, float(tmin), float(tmax), float(tstep))
    fig_rho, fig_surf = figs

    st.pyplot(fig_rho, use_container_width=True)
    st.pyplot(fig_surf, use_container_width=True)

    # 메모리 누수 방지 (옵션)
    import matplotlib.pyplot as _plt
    for _f in figs:
        _plt.close(_f)

if __name__ == "__main__":
    # 로컬에서 단독 실행 시에도 동작
    st_app()
