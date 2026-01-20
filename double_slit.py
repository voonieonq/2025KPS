# double_slit_sim.py
# Double-slit simulation (Fraunhofer, particle build-up)
# - 상단: 이론적 강도분포 I(x)
# - 하단: 입자 누적 히스토그램(샘플링으로 점차 간섭무늬가 나타나는 모습)
# - 슬라이더: 파장 λ, 슬릿폭 a, 슬릿간격 d, 스크린거리 L, 코히런스 γ (0=완전 소실, 1=완전 간섭)
# - 토글: Single-slit / Double-slit, Which-path(측정) = γ=0
# - 버튼: Emit 1 / 100 / 1000, Reset
#
# Fraunhofer 근사: I(x) ∝ sinc^2(β) * [1 + γ·cos(2α)]  (double-slit),
#   β = π a x /(λ L), α = π d x /(λ L)
#   γ∈[0,1]는 가간섭도(코히런스). which-path 측정 시 γ≈0로 모델링.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from numpy.random import default_rng

rng = default_rng()

# -----------------------------
# 물리/수치 파라미터
# -----------------------------
# 화면 좌표 범위 (m)
X_RANGE = 0.02  # ±2 cm
N_SAMPLES = 5001  # x 격자
BINS = 200       # 히스토그램 bins

# 초기값 (단위: m)
lambda0 = 650e-9   # 650 nm
a0      = 50e-6    # slit width 50 μm
d0      = 200e-6   # slit separation (center-to-center) 200 μm
L0      = 1.5      # screen distance 1.5 m
gamma0  = 1.0      # full coherence

# -----------------------------
# 유틸
# -----------------------------
def sinc(x):
    # numpy의 normalized sinc와 달리 여기선 sin(x)/x
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out

def intensity_profile(x, lam, a, d, L, gamma, mode='double'):
    """
    이론 강도 I(x) (Before normalize). Fraunhofer approximation.
    mode='double' 또는 'single'
    """
    k = 2*np.pi/lam
    beta = np.pi * a * x / (lam * L)    # 회절항
    env = sinc(beta)**2                 # sinc^2(β)
    if mode == 'single':
        return env                      # 단일 슬릿: 간섭항 없음
    # double-slit: 1 + γ cos(2α)
    alpha = np.pi * d * x / (lam * L)
    return env * (1.0 + gamma * np.cos(2.0 * alpha))

def normalized_pdf(x, lam, a, d, L, gamma, mode):
    I = intensity_profile(x, lam, a, d, L, gamma, mode)
    I = np.clip(I, 0.0, None)
    area = np.trapz(I, x)
    if area <= 0:
        return np.ones_like(I)/len(I)
    return I / area

def sample_positions(num, x_grid, pdf):
    # 연속분포 샘플링: CDF 역함수 보간
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    # 중복 방지용 작은 단조 증가 보정
    eps = 1e-12
    cdf = np.maximum.accumulate(cdf + eps*np.arange(len(cdf)))
    u = rng.random(num) * cdf[-1]
    xs = np.interp(u, cdf, x_grid)
    return xs

# -----------------------------
# 초기 격자/상태
# -----------------------------
x = np.linspace(-X_RANGE, X_RANGE, N_SAMPLES)
mode = 'double'   # 'double' or 'single'
which_path = False

# 누적 히스토그램 상태
counts, edges = np.histogram([], bins=BINS, range=(-X_RANGE, X_RANGE))
centers = 0.5*(edges[:-1] + edges[1:])

# -----------------------------
# Figure & Axes
# -----------------------------
plt.close('all')
fig = plt.figure(figsize=(11, 7))

ax_top = fig.add_axes([0.08, 0.58, 0.84, 0.35])
ax_bot = fig.add_axes([0.08, 0.15, 0.84, 0.35])

# 슬라이더 영역
ax_lam = fig.add_axes([0.10, 0.08, 0.20, 0.03])  # λ
ax_a   = fig.add_axes([0.35, 0.08, 0.20, 0.03])  # a
ax_d   = fig.add_axes([0.60, 0.08, 0.20, 0.03])  # d

ax_L   = fig.add_axes([0.10, 0.04, 0.20, 0.03])  # L
ax_gam = fig.add_axes([0.35, 0.04, 0.20, 0.03])  # γ

# 버튼 영역
ax_emit1   = fig.add_axes([0.60, 0.04, 0.08, 0.03])
ax_emit100 = fig.add_axes([0.69, 0.04, 0.08, 0.03])
ax_emit1k  = fig.add_axes([0.78, 0.04, 0.08, 0.03])
ax_reset   = fig.add_axes([0.87, 0.04, 0.05, 0.03])

# 모드 토글 버튼
ax_tgl_mode = fig.add_axes([0.87, 0.08, 0.05, 0.03])
ax_tgl_wp   = fig.add_axes([0.81, 0.08, 0.05, 0.03])

# 라인/바 핸들
line_theory, = ax_top.plot([], [], lw=2)
bars = None

# -----------------------------
# 슬라이더 생성 (matplotlib 3.9+ 안전: 키워드 사용)
# -----------------------------
s_lambda = Slider(ax_lam, label="λ (nm)",  valmin=380.0,   valmax=780.0,   valinit=lambda0*1e9)
s_a      = Slider(ax_a,   label="a (μm)",  valmin=10.0,    valmax=200.0,   valinit=a0*1e6)
s_d      = Slider(ax_d,   label="d (μm)",  valmin=20.0,    valmax=500.0,   valinit=d0*1e6)

s_L      = Slider(ax_L,   label="L (m)",   valmin=0.3,     valmax=5.0,     valinit=L0)
s_gamma  = Slider(ax_gam, label="γ (coherence)", valmin=0.0, valmax=1.0,   valinit=gamma0)

b_emit1   = Button(ax_emit1,   "Emit 1")
b_emit100 = Button(ax_emit100, "Emit 100")
b_emit1k  = Button(ax_emit1k,  "Emit 1000")
b_reset   = Button(ax_reset,   "Reset")

b_tgl_mode = Button(ax_tgl_mode, "Mode")
b_tgl_wp   = Button(ax_tgl_wp,   "W-Path")

# -----------------------------
# 그리기 루틴
# -----------------------------
def compute_theory():
    lam = s_lambda.val * 1e-9
    a   = s_a.val      * 1e-6
    d   = s_d.val      * 1e-6
    L   = s_L.val
    gam = 0.0 if which_path else s_gamma.val
    return intensity_profile(x, lam, a, d, L, gam, mode)

def update_theory_plot():
    I = compute_theory()
    # 보기 좋게 최대값 1로 정규화
    if I.max() > 0:
        I = I / I.max()
    line_theory.set_data(x*1000, I)  # x축 mm 단위로 표시
    ax_top.set_xlim([-X_RANGE*1000, X_RANGE*1000])
    ax_top.set_ylim([0, 1.05])
    ttl = "Double-slit" if mode == 'double' else "Single-slit"
    ttl += "  |  which-path: ON" if which_path else "  |  which-path: OFF"
    ax_top.set_title(f"Theoretical Intensity (normalized)  —  {ttl}")
    ax_top.set_xlabel("Screen position x (mm)")
    ax_top.set_ylabel("I(x) / I_max")

def update_hist_plot():
    global bars
    ax_bot.cla()
    # 히스토그램을 확률밀도가 아니라 '카운트'로 보여줌
    bars = ax_bot.bar(centers*1000, counts, width=(edges[1]-edges[0])*1000, align='center')
    ax_bot.set_xlim([-X_RANGE*1000, X_RANGE*1000])
    # y범위 자동
    ymax = max(1, counts.max())
    ax_bot.set_ylim([0, ymax*1.1])
    ax_bot.set_xlabel("Screen position x (mm)")
    ax_bot.set_ylabel("Counts")
    ax_bot.set_title("Particle Build-up (counts)")

def redraw():
    update_theory_plot()
    update_hist_plot()
    fig.canvas.draw_idle()

def emit(n):
    global counts
    lam = s_lambda.val * 1e-9
    a   = s_a.val      * 1e-6
    d   = s_d.val      * 1e-6
    L   = s_L.val
    gam = 0.0 if which_path else s_gamma.val
    pdf = normalized_pdf(x, lam, a, d, L, gam, mode)
    xs = sample_positions(n, x, pdf)
    new_counts, _ = np.histogram(xs, bins=BINS, range=(-X_RANGE, X_RANGE))
    counts = counts + new_counts
    update_hist_plot()
    fig.canvas.draw_idle()

def reset():
    global counts, which_path, mode
    counts[:] = 0
    which_path = False
    mode = 'double'
    s_lambda.reset(); s_a.reset(); s_d.reset(); s_L.reset(); s_gamma.reset()
    redraw()

# -----------------------------
# 콜백
# -----------------------------
def on_param_change(val):
    # 파라미터 바뀌면 이론곡선 갱신 (히스토그램은 유지)
    update_theory_plot()
    fig.canvas.draw_idle()

def on_emit1(evt):   emit(1)
def on_emit100(evt): emit(100)
def on_emit1k(evt):  emit(1000)
def on_reset(evt):   reset()

def on_toggle_mode(evt):
    global mode
    mode = 'single' if mode == 'double' else 'double'
    on_param_change(None)

def on_toggle_wp(evt):
    global which_path
    which_path = not which_path
    on_param_change(None)

# 이벤트 연결
for s in (s_lambda, s_a, s_d, s_L, s_gamma):
    s.on_changed(on_param_change)

b_emit1.on_clicked(on_emit1)
b_emit100.on_clicked(on_emit100)
b_emit1k.on_clicked(on_emit1k)
b_reset.on_clicked(on_reset)

b_tgl_mode.on_clicked(on_toggle_mode)
b_tgl_wp.on_clicked(on_toggle_wp)

# 첫 렌더
redraw()
st.pyplot(fig)
