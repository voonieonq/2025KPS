
# uncertainty.py
# Heisenberg 불확정성(σ_x · σ_p >= 1/2) 시뮬레이터
# 단위 스케일링: ħ = 1

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -----------------------------
# 유틸 함수
# -----------------------------
def gaussian_packet(x, sigma_x, x0, k0):

    """가우시안 파동함수 ψ(x). Var[x] = sigma_x^2, 정규화 포함."""
    norm = (1.0 / (2 * np.pi * sigma_x**2)) ** 0.25
    psi = norm * np.exp(-(x - x0) ** 2 / (4 * sigma_x**2)) * np.exp(1j * k0 * x)
    return psi

def fft_momentum(psi, dx):
    """φ(k) = FFT[ψ(x)] with unitary-like scaling (ħ=1 → p=k)."""
    phi = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2 * np.pi)
    return phi

def mean_std(grid, pdf, dgrid):
    """정규화된 확률밀도 pdf 에 대한 평균/표준편차."""

    Z = np.sum(pdf) * dgrid
    if Z <= 0:
        return 0.0, 0.0
    pdf = pdf / Z
    mu = np.sum(grid * pdf) * dgrid

    var = np.sum(((grid - mu) ** 2) * pdf) * dgrid


    var = max(var, 0.0)
    return mu, np.sqrt(var)

def build_grid(L, N):

    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    k = np.fft.fftshift(k)
    dk = k[1] - k[0]
    return x, k, dx, dk

# -----------------------------
# 초기 파라미터 / 전역 상태
# 초기 파라미터
# -----------------------------
L_init   = 100.0     # 공간 길이
N        = 4096      # 격자점
sigma0   = 2.0       # 초기 σ_x
x0_init  = 0.0
k0_init  = 0.0

x, k, dx, dk = build_grid(L_init, N)

# 측정(국소화) 상태
measured      = False
meas_center   = 0.0
meas_strength = 1.5  # measurement_width = meas_strength * sigma_x

# 초기 파동함수
psi_base = gaussian_packet(x, sigma0, x0_init, k0_init)
psi      = psi_base.copy()

def apply_measurement(psi, sigma_x, x_meas, strength):
    """국소화(측정) 가우시안 윈도우를 적용."""
    meas_sigma = max(1e-6, strength * sigma_x)
    window = np.exp(-(x - x_meas) ** 2 / (2 * meas_sigma ** 2))
    psi2 = psi * window
    # 재정규화
    norm = np.sqrt(np.sum(np.abs(psi2) ** 2) * dx)

    if norm > 0:
        psi2 /= norm
    return psi2

# -----------------------------

# 독립 실행용(별도 창) 애플리케이션
# -----------------------------
def run_matplotlib_app():
    """Matplotlib 위젯으로 데스크톱 창을 띄워 인터랙션."""
    global x, k, dx, dk, psi_base, psi, measured, meas_center, meas_strength
    global N, sigma0, x0_init, k0_init, L_init

    plt.figure(figsize=(11, 6))
    ax_pos = plt.axes([0.08, 0.40, 0.40, 0.50])  # 위치분포
    ax_mom = plt.axes([0.56, 0.40, 0.40, 0.50])  # 모멘텀분포

    axcolor = '0.95'
    ax_sigma = plt.axes([0.10, 0.26, 0.36, 0.03], facecolor=axcolor)
    ax_x0    = plt.axes([0.10, 0.21, 0.36, 0.03], facecolor=axcolor)
    ax_k0    = plt.axes([0.10, 0.16, 0.36, 0.03], facecolor=axcolor)
    ax_L     = plt.axes([0.10, 0.11, 0.36, 0.03], facecolor=axcolor)

    ax_xm    = plt.axes([0.58, 0.26, 0.36, 0.03], facecolor=axcolor)
    ax_str   = plt.axes([0.58, 0.21, 0.36, 0.03], facecolor=axcolor)

    btn_measure_ax = plt.axes([0.58, 0.12, 0.17, 0.05])
    btn_reset_ax   = plt.axes([0.77, 0.12, 0.17, 0.05])

    # 텍스트 박스(불확정성 지표)
    info_text = plt.gcf().text(0.08, 0.02, "", fontsize=11)

    # 슬라이더/버튼
    s_sigma = Slider(ax_sigma, r'σ_x', 0.1, 10.0, valinit=sigma0, valstep=0.1)
    s_x0    = Slider(ax_x0,    'x0', -L_init/4, L_init/4, valinit=x0_init, valstep=0.1)
    s_k0    = Slider(ax_k0,    'k0 (≡⟨p⟩)', -5.0, 5.0, valinit=k0_init, valstep=0.1)
    s_L     = Slider(ax_L,     'L (length)', 20.0, 200.0, valinit=L_init, valstep=5.0)

    s_xm    = Slider(ax_xm,    'center of measurement point x_meas', -L_init/4, L_init/4, valinit=0.0, valstep=0.1)
    s_str   = Slider(ax_str,   'intensity of the measurement point', 0.1, 5.0, valinit=meas_strength, valstep=0.1)

    btn_measure = Button(btn_measure_ax, 'measure')
    btn_reset   = Button(btn_reset_ax,   'reset')

    # 초기 플롯
    line_pos, = ax_pos.plot([], [])
    line_mom, = ax_mom.plot([], [])
    ax_pos.set_title("|ψ(x)|²  (position space)")
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("|ψ|²")
    ax_pos.grid(True, alpha=0.3)

    ax_mom.set_title("|φ(k)|²  (momentum space, ħ=1 → p=k)")
    ax_mom.set_xlabel("k (≡ p)")
    ax_mom.set_ylabel("|φ|²")
    ax_mom.grid(True, alpha=0.3)

    # 업데이트 루틴
    def recompute(_=None):
        nonlocal line_pos, line_mom, info_text

        # 격자 길이 변경 반영
        L_new = s_L.val
        s_x0.ax.set_xlim(-L_new/4, L_new/4)
        s_xm.ax.set_xlim(-L_new/4, L_new/4)

        if L_new != (x[-1] - x[0] + (x[1] - x[0])):
            x_new, k_new, dx_new, dk_new = build_grid(L_new, N)
            sigma_x = s_sigma.val
            x0 = s_x0.val
            k0 = s_k0.val
            base = gaussian_packet(x_new, sigma_x, x0, k0)
            cur = base.copy()
            if measured:
                cur = apply_measurement(cur, sigma_x, meas_center, meas_strength)

            # 교체
            globals()['x'], globals()['k']   = x_new, k_new
            globals()['dx'], globals()['dk'] = dx_new, dk_new
            globals()['psi_base'], globals()['psi'] = base, cur
        else:
            sigma_x = s_sigma.val
            x0 = s_x0.val
            k0 = s_k0.val
            base = gaussian_packet(x, sigma_x, x0, k0)
            cur = base.copy()
            if measured:
                cur = apply_measurement(cur, sigma_x, meas_center, meas_strength)
            globals()['psi_base'], globals()['psi'] = base, cur

        # 분포/지표
        cur = globals()['psi']
        rho_x = np.abs(cur) ** 2
        phi   = fft_momentum(cur, dx)
        rho_k = np.abs(phi) ** 2

        x_mean, sx = mean_std(x, rho_x, dx)
        k_mean, sk = mean_std(k, rho_k, dk)
        sp = sk
        product = sx * sp
        bound = 0.5

        # 그리기
        line_pos.set_data(x, rho_x)
        line_mom.set_data(k, rho_k)

        def autoscale(ax, grid, pdf):
            y_max = float(pdf.max()) if pdf.size else 1.0
            if y_max <= 0:
                y_max = 1.0
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(0, y_max * 1.1)

        autoscale(ax_pos, x, rho_x)
        autoscale(ax_mom, k, rho_k)

        status = "σₓ·σ_p ≥ 1/2 충족" if product >= bound - 1e-4 else "측정(국소화) 상태"
        info_text.set_text(
            f"⟨x⟩ = {x_mean:.4f}   σₓ = {sx:.4f}    "
            f"⟨p⟩ = {k_mean:.4f}   σ_p = {sp:.4f}    "
            f"σₓ·σ_p = {product:.5f}   (경계 1/2)   {status}"
        )
        plt.draw()

    # 콜백
    s_sigma.on_changed(recompute)
    s_x0.on_changed(recompute)
    s_k0.on_changed(recompute)
    s_L.on_changed(recompute)
    s_xm.on_changed(recompute)
    s_str.on_changed(recompute)

    def on_measure(_):
        global measured, meas_center, meas_strength, psi
        measured = True
        meas_center = s_xm.val
        meas_strength = s_str.val
        sigma_x = s_sigma.val
        psi = apply_measurement(psi, sigma_x, meas_center, meas_strength)
        recompute()

    def on_reset(_):
        global measured, psi, psi_base
        measured = False
        psi_base = gaussian_packet(x, s_sigma.val, s_x0.val, s_k0.val)
        psi = psi_base.copy()
        recompute()

    btn_measure.on_clicked(on_measure)
    btn_reset.on_clicked(on_reset)

    # 첫 렌더
    recompute()
    plt.tight_layout()
    
def render(**params):
    # 1) 새 그림을 만들거나
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    # ax.plot(...), ax.imshow(...), 등등
    return fig

# 독립 실행
if __name__ == "__main__":
    run_matplotlib_app()

# -----------------------------
# Streamlit용 렌더러
# -----------------------------
def render():
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    st.subheader("하이젠베르크 불확정성 시뮬레이터 (ħ = 1)")

    # 상태(측정 유지)
    ss = st.session_state
    ss.setdefault("unc_measured", False)
    ss.setdefault("unc_meas_center", 0.0)
    ss.setdefault("unc_meas_strength", 1.5)

    # ⊞ UI
    col1, col2, col3 = st.columns(3)
    sigma_x = col1.slider("σₓ", 0.1, 10.0, 2.0, 0.1)
    x0      = col2.slider("x₀", -10.0, 10.0, 0.0, 0.1)
    k0      = col3.slider("k₀ (≡⟨p⟩)", -5.0, 5.0, 0.0, 0.1)
    L       = st.slider("L (길이)", 20.0, 200.0, 100.0, 5.0)

    c1, c2  = st.columns(2)
    x_meas   = c1.slider("측정 중심 x_meas", -L/4, L/4, float(ss.unc_meas_center), 0.1)
    strength = c2.slider("측정 강도", 0.1, 5.0, float(ss.unc_meas_strength), 0.1)

    b1, b2, _ = st.columns(3)
    if b1.button("측정 적용"):
        ss.unc_measured     = True
        ss.unc_meas_center  = float(x_meas)
        ss.unc_meas_strength = float(strength)
    if b2.button("리셋"):
        ss.unc_measured = False

    # 계산
    N = 2048
    x, k, dx, dk = build_grid(L, N)
    psi = gaussian_packet(x, sigma_x, x0, k0)
    if ss.unc_measured:
        psi = apply_measurement(psi, sigma_x, ss.unc_meas_center, ss.unc_meas_strength)

    rho_x = np.abs(psi) ** 2
    phi   = fft_momentum(psi, dx)
    rho_k = np.abs(phi) ** 2

    mx, sx = mean_std(x, rho_x, dx)
    mp, sp = mean_std(k, rho_k, dk)  # ħ=1 → p=k
    prod   = sx * sp

    # 플롯
    fig1, ax1 = plt.subplots()
    ax1.plot(x, rho_x)
    ax1.set_xlabel("x"); ax1.set_ylabel("|ψ(x)|²")
    ax1.set_title(f"위치분포  σₓ≈{sx:.3f}")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(k, rho_k)
    ax2.set_xlabel("p(=k)"); ax2.set_ylabel("|φ(p)|²")
    ax2.set_title(f"운동량분포  σₚ≈{sp:.3f}")
    st.pyplot(fig2)

    st.info(f"σₓ·σₚ ≈ {prod:.3f}  (이론 경계: 1/2)")


#3차원 시각화(만약 3차원으로 할 거면 아래의 코드로 바꾸)
'''
# uncertainty.py
# Heisenberg 불확정성(σ_x · σ_p >= 1/2) 시뮬레이터
# 단위 스케일링: ħ = 1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D   # 3D 플롯 등록용

# -----------------------------
# 유틸 함수
# -----------------------------
def gaussian_packet(x, sigma_x, x0, k0):
    """가우시안 파동함수 ψ(x). Var[x] = sigma_x^2, 정규화 포함."""
    norm = (1.0 / (2 * np.pi * sigma_x**2)) ** 0.25
    psi = norm * np.exp(-(x - x0) ** 2 / (4 * sigma_x**2)) * np.exp(1j * k0 * x)
    return psi

def fft_momentum(psi, dx):
    """φ(k) = FFT[ψ(x)] with unitary-like scaling (ħ=1 → p=k)."""
    phi = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2 * np.pi)
    return phi

def mean_std(grid, pdf, dgrid):
    """정규화된 확률밀도 pdf 에 대한 평균/표준편차."""
    Z = np.sum(pdf) * dgrid
    if Z <= 0:
        return 0.0, 0.0
    pdf = pdf / Z
    mu = np.sum(grid * pdf) * dgrid
    var = np.sum(((grid - mu) ** 2) * pdf) * dgrid
    var = max(var, 0.0)
    return mu, np.sqrt(var)

def build_grid(L, N):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    k = np.fft.fftshift(k)
    dk = k[1] - k[0]
    return x, k, dx, dk

# -----------------------------
# 초기 파라미터 / 전역 상태
# -----------------------------
L_init   = 100.0     # 공간 길이
N        = 4096      # 격자점
sigma0   = 2.0       # 초기 σ_x
x0_init  = 0.0
k0_init  = 0.0

x, k, dx, dk = build_grid(L_init, N)

# 측정(국소화) 상태
measured      = False
meas_center   = 0.0
meas_strength = 1.5  # measurement_width = meas_strength * sigma_x

# 초기 파동함수
psi_base = gaussian_packet(x, sigma0, x0_init, k0_init)
psi      = psi_base.copy()

def apply_measurement(psi, sigma_x, x_meas, strength):
    """국소화(측정) 가우시안 윈도우를 적용."""
    meas_sigma = max(1e-6, strength * sigma_x)
    window = np.exp(-(x - x_meas) ** 2 / (2 * meas_sigma ** 2))
    psi2 = psi * window
    # 재정규화
    norm = np.sqrt(np.sum(np.abs(psi2) ** 2) * dx)
    if norm > 0:
        psi2 /= norm
    return psi2

# -----------------------------
# 독립 실행용 Matplotlib 앱
# -----------------------------
def run_matplotlib_app():
    global x, k, dx, dk, psi_base, psi, measured, meas_center, meas_strength
    global N, sigma0, x0_init, k0_init, L_init

    # 메인(2D) Figure
    plt.figure(figsize=(11, 6))
    ax_pos = plt.axes([0.08, 0.40, 0.40, 0.50])  # 위치분포
    ax_mom = plt.axes([0.56, 0.40, 0.40, 0.50])  # 모멘텀분포

    axcolor = '0.95'
    ax_sigma = plt.axes([0.10, 0.26, 0.36, 0.03], facecolor=axcolor)
    ax_x0    = plt.axes([0.10, 0.21, 0.36, 0.03], facecolor=axcolor)
    ax_k0    = plt.axes([0.10, 0.16, 0.36, 0.03], facecolor=axcolor)
    ax_L     = plt.axes([0.10, 0.11, 0.36, 0.03], facecolor=axcolor)

    ax_xm    = plt.axes([0.58, 0.26, 0.36, 0.03], facecolor=axcolor)
    ax_str   = plt.axes([0.58, 0.21, 0.36, 0.03], facecolor=axcolor)

    btn_measure_ax = plt.axes([0.58, 0.12, 0.17, 0.05])
    btn_reset_ax   = plt.axes([0.77, 0.12, 0.17, 0.05])

    info_text = plt.gcf().text(0.08, 0.02, "", fontsize=11)

    # 슬라이더/버튼
    s_sigma = Slider(ax_sigma, r'σ_x', 0.1, 10.0, valinit=sigma0, valstep=0.1)
    s_x0    = Slider(ax_x0,    'x0', -L_init/4, L_init/4, valinit=x0_init, valstep=0.1)
    s_k0    = Slider(ax_k0,    'k0 (≡⟨p⟩)', -5.0, 5.0, valinit=k0_init, valstep=0.1)
    s_L     = Slider(ax_L,     'L (length)', 20.0, 200.0, valinit=L_init, valstep=5.0)

    s_xm    = Slider(ax_xm,    'center of measurement point x_meas', -L_init/4, L_init/4, valinit=0.0, valstep=0.1)
    s_str   = Slider(ax_str,   'intensity of the measurement point', 0.1, 5.0, valinit=meas_strength, valstep=0.1)

    btn_measure = Button(btn_measure_ax, 'measure')
    btn_reset   = Button(btn_reset_ax,   'reset')

    # 초기 플롯(2D)
    line_pos, = ax_pos.plot([], [])
    line_mom, = ax_mom.plot([], [])
    ax_pos.set_title("|ψ(x)|²  (position space)")
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("|ψ|²")
    ax_pos.grid(True, alpha=0.3)

    ax_mom.set_title("|φ(k)|²  (momentum space, ħ=1 → p=k)")
    ax_mom.set_xlabel("k (≡ p)")
    ax_mom.set_ylabel("|φ|²")
    ax_mom.grid(True, alpha=0.3)

    # ---- 3D Figure: 한 번만 생성 (여기가 핵심) ----
    fig3 = plt.figure("Phase-space 3D", figsize=(7, 5))
    ax3  = fig3.add_subplot(111, projection="3d")
    # -----------------------------------------------

    def recompute(_=None):
        nonlocal line_pos, line_mom, info_text, ax3, fig3

        # 격자 길이 변경 반영
        L_new = s_L.val
        s_x0.ax.set_xlim(-L_new/4, L_new/4)
        s_xm.ax.set_xlim(-L_new/4, L_new/4)

        if L_new != (x[-1] - x[0] + (x[1] - x[0])):
            x_new, k_new, dx_new, dk_new = build_grid(L_new, N)
            sigma_x = s_sigma.val
            x0 = s_x0.val
            k0 = s_k0.val
            base = gaussian_packet(x_new, sigma_x, x0, k0)
            cur = base.copy()
            if measured:
                cur = apply_measurement(cur, sigma_x, meas_center, meas_strength)
            globals()['x'], globals()['k']   = x_new, k_new
            globals()['dx'], globals()['dk'] = dx_new, dk_new
            globals()['psi_base'], globals()['psi'] = base, cur
        else:
            sigma_x = s_sigma.val
            x0 = s_x0.val
            k0 = s_k0.val
            base = gaussian_packet(x, sigma_x, x0, k0)
            cur = base.copy()
            if measured:
                cur = apply_measurement(cur, sigma_x, meas_center, meas_strength)
            globals()['psi_base'], globals()['psi'] = base, cur

        # 분포/지표
        cur = globals()['psi']
        rho_x = np.abs(cur) ** 2
        phi   = fft_momentum(cur, dx)
        rho_k = np.abs(phi) ** 2

        x_mean, sx = mean_std(x, rho_x, dx)
        k_mean, sk = mean_std(k, rho_k, dk)
        sp = sk
        product = sx * sp
        bound = 0.5

        # 2D 갱신
        line_pos.set_data(x, rho_x)
        line_mom.set_data(k, rho_k)

        def autoscale(ax, grid, pdf):
            y_max = float(pdf.max()) if pdf.size else 1.0
            if y_max <= 0:
                y_max = 1.0
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(0, y_max * 1.1)

        autoscale(ax_pos, x, rho_x)
        autoscale(ax_mom, k, rho_k)

        status = "σₓ·σ_p ≥ 1/2 충족" if product >= bound - 1e-4 else "측정(국소화) 상태"
        info_text.set_text(
            f"⟨x⟩ = {x_mean:.4f}   σₓ = {sx:.4f}    "
            f"⟨p⟩ = {k_mean:.4f}   σ_p = {sp:.4f}    "
            f"σₓ·σ_p = {product:.5f}   (경계 1/2)   {status}"
        )

        # ---- 3D 갱신 (새 창 생성 금지: 기존 축 재사용) ----
        # 너무 큰 격자를 그대로 띄우면 매우 느려지므로 다운샘플링
        MAX_N3D = 200
        step_x = max(1, len(x) // MAX_N3D)
        step_k = max(1, len(k) // MAX_N3D)
        x3 = x[::step_x]
        k3 = k[::step_k]
        Z  = np.outer(rho_x[::step_x], rho_k[::step_k])
        X, P = np.meshgrid(x3, k3)

        ax3.cla()  # 이전 내용 지우기
        ax3.plot_surface(X, P, Z, cmap="plasma", rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax3.set_xlabel("x")
        ax3.set_ylabel("p")
        ax3.set_zlabel("ρ(x,p)")
        fig3.canvas.draw_idle()
        # ---------------------------------------------------

        plt.draw()

    # 콜백
    s_sigma.on_changed(recompute)
    s_x0.on_changed(recompute)
    s_k0.on_changed(recompute)
    s_L.on_changed(recompute)
    s_xm.on_changed(recompute)
    s_str.on_changed(recompute)

    def on_measure(_):
        global measured, meas_center, meas_strength, psi
        measured = True
        meas_center = s_xm.val
        meas_strength = s_str.val
        sigma_x = s_sigma.val
        psi = apply_measurement(psi, sigma_x, meas_center, meas_strength)
        recompute()

    def on_reset(_):
        global measured, psi, psi_base
        measured = False
        psi_base = gaussian_packet(x, s_sigma.val, s_x0.val, s_k0.val)
        psi = psi_base.copy()
        recompute()

    btn_measure.on_clicked(on_measure)
    btn_reset.on_clicked(on_reset)

    # 첫 렌더
    recompute()
    plt.tight_layout()
    # st.pyplot(fig)는 여기서 '한 번만' 호출 → 두 Figure(2D, 3D)가 함께 뜸
    st.pyplot(fig)

# 독립 실행
if __name__ == "__main__":
    run_matplotlib_app()

# -----------------------------
# Streamlit 렌더러
# -----------------------------
def render():
    import streamlit as st
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    st.subheader("하이젠베르크 불확정성 시뮬레이터 (ħ = 1)")

    ss = st.session_state
    ss.setdefault("unc_measured", False)
    ss.setdefault("unc_meas_center", 0.0)
    ss.setdefault("unc_meas_strength", 1.5)

    col1, col2, col3 = st.columns(3)
    sigma_x = col1.slider("σₓ", 0.1, 10.0, 2.0, 0.1)
    x0      = col2.slider("x₀", -10.0, 10.0, 0.0, 0.1)
    k0      = col3.slider("k₀ (≡⟨p⟩)", -5.0, 5.0, 0.0, 0.1)
    L       = st.slider("L (길이)", 20.0, 200.0, 100.0, 5.0)

    c1, c2  = st.columns(2)
    x_meas   = c1.slider("측정 중심 x_meas", -L/4, L/4, float(ss.unc_meas_center), 0.1)
    strength = c2.slider("측정 강도", 0.1, 5.0, float(ss.unc_meas_strength), 0.1)

    b1, b2, _ = st.columns(3)
    if b1.button("측정 적용"):
        ss.unc_measured      = True
        ss.unc_meas_center   = float(x_meas)
        ss.unc_meas_strength = float(strength)
    if b2.button("리셋"):
        ss.unc_measured = False

    # 계산
    N = 2048
    x, k, dx, dk = build_grid(L, N)
    psi = gaussian_packet(x, sigma_x, x0, k0)
    if ss.unc_measured:
        psi = apply_measurement(psi, sigma_x, ss.unc_meas_center, ss.unc_meas_strength)

    rho_x = np.abs(psi) ** 2
    phi   = fft_momentum(psi, dx)
    rho_k = np.abs(phi) ** 2

    mx, sx = mean_std(x, rho_x, dx)
    mp, sp = mean_std(k, rho_k, dk)  # ħ=1 → p=k
    prod   = sx * sp

    # 2D 플롯들
    fig1, ax1 = plt.subplots()
    ax1.plot(x, rho_x)
    ax1.set_xlabel("x"); ax1.set_ylabel("|ψ(x)|²")
    ax1.set_title(f"위치분포  σₓ≈{sx:.3f}")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(k, rho_k)
    ax2.set_xlabel("p(=k)"); ax2.set_ylabel("|φ(p)|²")
    ax2.set_title(f"운동량분포  σₚ≈{sp:.3f}")
    st.pyplot(fig2)

    # 3D 플롯 (다운샘플링 적용)
    MAX_N3D = 200
    step_x = max(1, len(x) // MAX_N3D)
    step_k = max(1, len(k) // MAX_N3D)
    x3 = x[::step_x]
    k3 = k[::step_k]
    Z  = np.outer(rho_x[::step_x], rho_k[::step_k])
    X, P = np.meshgrid(x3, k3)

    fig3 = plt.figure()
    ax3  = fig3.add_subplot(111, projection="3d")
    ax3.plot_surface(X, P, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax3.set_xlabel("x")
    ax3.set_ylabel("p")
    ax3.set_zlabel("ρ(x,p)")
    st.pyplot(fig3)

    st.info(f"σₓ·σₚ ≈ {prod:.3f}  (이론 경계: 1/2)")
    '''