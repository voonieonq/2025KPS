#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Streamlit 실시간 렌더 버전 (이미지/GIF 금지, 프레임 갱신)
import time, random, numpy as np, streamlit as st
import matplotlib
matplotlib.use("Agg")  # 헤드리스 백엔드
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---- 상태 벡터/기본 설정 ----
up = np.array([1.0+0j, 0.0+0j])
down = np.array([0.0+0j, 1.0+0j])
# |Ψ⁻> = (|↑↓> - |↓↑>)/√2
psi_entangled = (np.kron(up, down) - np.kron(down, up))/np.sqrt(2.0)

def _projectors(basis):
    if basis == "z":
        plus, minus = up, down
        labels = ("up", "down")
    else:  # 'x'
        plus  = (up + down)/np.sqrt(2.0)
        minus = (up - down)/np.sqrt(2.0)
        labels = ("plus", "minus")
    Pp = plus[:, None] @ plus[None, :]
    Pm = minus[:, None] @ minus[None, :]
    return (Pp, Pm), labels, plus, minus

def measure_spin(state, which="first", basis="z"):
    (Pp, Pm), labels, plus, minus = _projectors(basis)
    I = np.eye(2)
    if which == "first":
        Pp_full, Pm_full = np.kron(Pp, I), np.kron(Pm, I)
    else:
        Pp_full, Pm_full = np.kron(I, Pp), np.kron(I, Pm)

    v_p, v_m = Pp_full @ state, Pm_full @ state
    p_p = float(np.real(np.vdot(v_p, v_p)))
    p_m = float(np.real(np.vdot(v_m, v_m)))
    probs = {labels[0]: p_p, labels[1]: p_m}

    outcome = random.choices([labels[0], labels[1]], weights=[p_p, p_m])[0]
    single = (plus if outcome == labels[0] else minus)  # 측정된 큐빗의 단일 상태
    return outcome, probs, single

def bloch_coords(s):
    a, b = s[0], s[1]
    sx = 2*np.real(np.conjugate(a)*b)
    sy = 2*np.imag(np.conjugate(a)*b)
    sz = np.abs(a)**2 - np.abs(b)**2
    return float(sx), float(sy), float(sz)

def _draw(prob, basis, measured, vec, azim):
    sx, sy, sz = vec
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # 확률 막대
    ax1.bar(list(prob.keys()), list(prob.values()))
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Measurement probabilities (basis: {basis})")
    ax1.set_ylabel("Probability")
    ax1.grid(alpha=0.3)

    # Bloch sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_surface(x, y, z, alpha=0.08, edgecolor="gray")

    ax2.quiver(0, 0, 0, sx, sy, sz, linewidth=2)
    ax2.set(xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1], xlabel="X", ylabel="Y", zlabel="Z")
    ax2.view_init(elev=30, azim=azim)
    ax2.set_title(f"A measured as: {measured}")
    fig.tight_layout()
    return fig

def st_app():
    st.subheader("양자 얽힘 시뮬레이터")

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        basis = st.radio("측정 기준", ["z", "x"], horizontal=True, index=0)
    with c2:
        which = st.radio("어느 큐빗 측정?", ["first", "second"], horizontal=True, index=0)
    with c3:
        live = st.radio("표시 방식", ["실시간 회전", "정지 뷰"], horizontal=True, index=0)

    seed_on = st.checkbox("난수 고정(재현)", value=False)
    if seed_on:
        seed = st.number_input("seed", value=42, step=1)
        random.seed(int(seed))

    # 측정 1회(재실행시 갱신)
    measured, prob_dict, single = measure_spin(psi_entangled, which=which, basis=basis)
    vec = bloch_coords(single)

    ph = st.empty()

    if live == "정지 뷰":
        az = st.slider("보기 각도(azim)", 0, 360, 300, 5)
        fig = _draw(prob_dict, basis, measured, vec, az)
        ph.pyplot(fig, clear_figure=True); plt.close(fig)
        return

    # === 실시간 회전 ===
    if "run_ent" not in st.session_state:
        st.session_state.run_ent = False
    if "ent_azim" not in st.session_state:
        st.session_state.ent_azim = 300

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("▶ 실행/정지"):
            st.session_state.run_ent = not st.session_state.run_ent
    with colB:
        speed = st.slider("회전 속도(deg/frame)", 1, 15, 6)

    az = st.session_state.ent_azim
    fig = _draw(prob_dict, basis, measured, vec, az)
    ph.pyplot(fig, clear_figure=True); plt.close(fig)

    while st.session_state.run_ent:
        az = (az + speed) % 360
        st.session_state.ent_azim = az
        fig = _draw(prob_dict, basis, measured, vec, az)
        ph.pyplot(fig, clear_figure=True); plt.close(fig)
        time.sleep(1/30)

if __name__ == "__main__":
    st_app()
