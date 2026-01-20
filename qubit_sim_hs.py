# qubit_sim_hs.py
# 고등학생 눈높이 큐비트 시뮬레이터
# - |ψ> = cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>
# - θ, φ 슬라이더로 상태를 바꾸면 확률이 실시간 갱신
# - [Measure (Animate)] 클릭 시 2초간 회전하는 확률 바퀴 애니메이션 후 결과 결정(확률적으로)
# - [Measure 100 times]으로 빈도 수렴 보기, [Reset]으로 초기화
# - [Apply H (Change Basis)] 체크 → 측정 전에 Hadamard(기저 변경) 적용
import streamlit as st
import matplotlib
matplotlib.use("TkAgg")  # 윈도우/VSCode에서 대화형 보장

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons

# --------- 선형대수 기본 ---------
H = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1,-1]], dtype=complex)

def ket(theta, phi):
    """|ψ> = [cos(θ/2), e^{iφ} sin(θ/2)]"""
    return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)

def probs_from_state(psi, apply_H=False):
    """(P0, P1) calculate. H|ψ> will be measured."""
    v = H @ psi if apply_H else psi
    p0 = float(np.abs(v[0])**2)
    p1 = float(np.abs(v[1])**2)
    s = p0 + p1
    if s == 0: p0, p1 = 1.0, 0.0
    else: p0, p1 = p0/s, p1/s
    return p0, p1

# --------- 전역 상태 ---------
rng = np.random.default_rng(7)
theta0 = np.pi/3
phi0   = np.pi/6

counts = {0:0, 1:0}  # 누적 측정 횟수

# --------- Figure 레이아웃 ---------
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(3, 3, height_ratios=[3.0, 1.2, 0.8])

ax_wheel = fig.add_subplot(gs[0, 0:2])  # 회전 바퀴
ax_bar   = fig.add_subplot(gs[0, 2])    # 빈도 막대
ax_text  = fig.add_subplot(gs[1, 0:3])  # ← 오타 수정: add_subplot

# 슬라이더/버튼 영역
ax_theta = fig.add_subplot(gs[2, 0])
ax_phi   = fig.add_subplot(gs[2, 1])
ax_ctrl  = fig.add_subplot(gs[2, 2])
ax_ctrl.axis("off")

# --------- 회전 바퀴 빌드 ---------
wheel = {
    "w0": Wedge((0,0), 1.0, 0, 180, facecolor="#4CAF50", edgecolor="black", lw=1.8),  # |0>
    "w1": Wedge((0,0), 1.0, 180, 360, facecolor="#F44336", edgecolor="black", lw=1.8) # |1>
}
for w in wheel.values():
    ax_wheel.add_patch(w)

hub = Circle((0,0), 0.08, color="black")
ax_wheel.add_patch(hub)
# 고정 포인터(위쪽)
ax_wheel.plot([0, 0], [1.05, 1.28], lw=4, color='black')
ax_wheel.text(0, 1.33, "▲", ha="center", va="center", fontsize=16)
# 타이틀
title_txt = ax_wheel.text(0, -1.22, "properties wheel (direction of the point is the result))", ha="center", fontsize=12)

# 바퀴 프레임 설정
ax_wheel.set_aspect('equal')
ax_wheel.set_xlim(-1.3, 1.3); ax_wheel.set_ylim(-1.45, 1.25)
ax_wheel.axis("off")

# 중앙 상태 텍스트 (측정 전 '?' / 후 결과)
center_txt = ax_wheel.text(0, 0, "?", ha="center", va="center",
                           fontsize=28, color="white", weight="bold",
                           bbox=dict(boxstyle="circle,pad=0.35", fc="#00000088", ec="none"))

# --------- 빈도 막대 ---------
bar_rects = ax_bar.bar(["0", "1"], [0, 0], color=["#4CAF50", "#F44336"])
ax_bar.set_ylim(0, 1)
ax_bar.set_title("real vs ideal probabilities")
theory_txt = ax_bar.text(0.5, 1.02, "", ha="center", va="bottom", transform=ax_bar.transAxes)

# --------- θ, φ 슬라이더 ---------
theta_slider = Slider(ax_theta, "θ", 0.0, np.pi, valinit=theta0)
phi_slider   = Slider(ax_phi,   "φ", 0.0, 2*np.pi, valinit=phi0)

# --------- 버튼/체크 (영문) ---------
btn_measure_ax = fig.add_axes([0.73, 0.18, 0.10, 0.05])
btn_many_ax    = fig.add_axes([0.84, 0.18, 0.10, 0.05])
btn_reset_ax   = fig.add_axes([0.62, 0.18, 0.10, 0.05])
chk_ax         = fig.add_axes([0.62, 0.25, 0.32, 0.10])

btn_measure = Button(btn_measure_ax, "Measure (Animate)")
btn_many    = Button(btn_many_ax,    "Measure 100 times")
btn_reset   = Button(btn_reset_ax,   "Reset")
chk = CheckButtons(chk_ax, ["Apply H (Change Basis)"], [False])

# --------- 텍스트/수식 ---------
ax_text.axis("off")
info_txt = ax_text.text(0.01, 0.75,
    "Qubit quick guide\n"
    "• A qubit can be a superposition of |0⟩ and |1⟩.\n"
    "• Use sliders θ, φ to set |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.\n"
    "• Measuring yields 0 or 1 according to probabilities P(0), P(1).\n"
    "• Turning on H applies Hadamard before measurement (basis change).\n",
    fontsize=11, va="top")

state_txt = ax_text.text(0.01, 0.15, "", fontsize=11, va="top", family="monospace")

# --------- 유틸 함수 ---------
def update_theory_and_text():
    theta = theta_slider.val
    phi   = phi_slider.val
    apply_H_flag = bool(chk.get_status()[0])

    psi = ket(theta, phi)
    p0, p1 = probs_from_state(psi, apply_H_flag)

    theory_txt.set_text(f"P(0) = {p0:.3f},  P(1) = {p1:.3f}")
    state_txt.set_text(
        f"|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩\n"
        f"θ = {theta:.2f} rad, φ = {phi:.2f} rad\n"
        + ("Apply H before measure: Yes" if apply_H_flag else "Apply H before measure: No")
    )
    set_wheel_sizes(p0, p1)

def set_wheel_sizes(p0, p1):
    deg0 = 360.0 * p0
    deg1 = 360.0 - deg0
    wheel["w0"].theta1, wheel["w0"].theta2 = 0.0, deg0
    wheel["w1"].theta1, wheel["w1"].theta2 = deg0, deg0 + deg1
    fig.canvas.draw_idle()

def set_wheel_angle(angle_rad):
    a = np.degrees(angle_rad) % 360.0
    for w in (wheel["w0"], wheel["w1"]):
        span = (w.theta2 - w.theta1) % 360.0
        w.theta1 = a % 360.0
        w.theta2 = (a + span) % 360.0
        a = (a + span) % 360.0
    fig.canvas.draw_idle()

def update_bar():
    total = counts[0] + counts[1]
    vals = [0, 0] if total == 0 else [counts[0]/total, counts[1]/total]
    for rect, v in zip(bar_rects, vals):
        rect.set_height(v)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel(f"total times: {total}")
    fig.canvas.draw_idle()

def sample_once():
    theta = theta_slider.val
    phi   = phi_slider.val
    psi   = ket(theta, phi)
    p0, p1 = probs_from_state(psi, bool(chk.get_status()[0]))
    return int(rng.choice([0,1], p=[p0, p1]))

# --------- 애니메이션(측정) ---------
spinning = False
spin_angle = 0.0
spin_speed = 8.0    # rad/s 기본 회전 속도
meas_duration = 2.0 # 초: 회전 후 멈추는 애니메이션 길이
meas_timer = 0.0
meas_target_state = None

def start_measure_animation():
    global spinning, meas_timer, meas_target_state
    if spinning:
        return
    meas_target_state = sample_once()
    center_txt.set_text("?")
    meas_timer = 0.0
    spinning = True

def measure_many(n=100):
    for _ in range(n):
        outcome = sample_once()
        counts[outcome] += 1
    update_bar()

def reset_all(_evt=None):
    global counts, spinning, meas_timer, meas_target_state, spin_angle
    counts = {0:0, 1:0}
    spinning = False
    meas_timer = 0.0
    meas_target_state = None
    spin_angle = 0.0
    center_txt.set_text("?")
    update_theory_and_text()
    update_bar()

def on_theta(val): update_theory_and_text()
def on_phi(val):   update_theory_and_text()
def on_check(label): update_theory_and_text()

theta_slider.on_changed(on_theta)
phi_slider.on_changed(on_phi)
chk.on_clicked(on_check)

btn_measure.on_clicked(lambda evt: start_measure_animation())
btn_many.on_clicked(lambda evt: measure_many(100))
btn_reset.on_clicked(reset_all)

# --------- 메인 애니메이션 루프 ---------
def ease_out_cubic(t):  # 0→1에서 끝에 감속
    return 1 - (1 - t)**3

def animate(frame):
    global spin_angle, spinning, meas_timer, meas_target_state

    # 바퀴는 항상 조금씩 돌아가게 해서 '살아있는' 느낌
    spin_angle = (spin_angle + 0.8 * (1/60)) % (2*np.pi)

    if spinning:
        dt = 1/60
        meas_timer += dt
        t = min(meas_timer / meas_duration, 1.0)
        factor = 1.0 - ease_out_cubic(t)  # 1→0 감속
        spin_angle = (spin_angle + (spin_speed * factor) * dt) % (2*np.pi)

        if t >= 1.0:
            spinning = False
            center_txt.set_text(str(meas_target_state))
            counts[meas_target_state] += 1
            update_bar()

            # 결과 조각 중심에 포인터가 오도록 스냅
            theta = theta_slider.val; phi = phi_slider.val
            p0, p1 = probs_from_state(ket(theta, phi), bool(chk.get_status()[0]))
            w0_span  = 2*np.pi * p0
            if meas_target_state == 0:
                target = (np.pi/2) - (0.0 + w0_span/2)
            else:
                w1_start = 0.0 + w0_span
                w1_span  = 2*np.pi - w0_span
                target = (np.pi/2) - (w1_start + w1_span/2)
            spin_angle = target % (2*np.pi)

    set_wheel_angle(spin_angle)
    return []

ani = FuncAnimation(fig, animate, interval=1000/60, blit=False)

# 초기 반영
def init_all():
    update_theory_and_text()
    update_bar()
init_all()

plt.tight_layout()
st.pyplot(fig)
