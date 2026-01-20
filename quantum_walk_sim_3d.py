# quantum_walk_sim_3d_auto.py
# Classical Random Walk vs Coined Quantum Walk (1D) — AutoPlay + Boundary + 3D viz
# - 오토플레이: 실행 즉시 진행, 스페이스바(⎵) 토글, R로 리셋
# - 슬라이더: θ(coin), φ(initial coin phase; 리셋 후 반영), N3D(3D 표시 길이),
#             SPF(steps per frame), MAX steps(자동정지 한계)
# - 라디오: Boundary = absorb/reflect, 3D 대상 = Quantum/Classical
# - 상단: 3D 표면(최근 N3D 스텝), 하단: 현재 분포(양자/고전), 좌하단 메트릭

import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# 설정
# -----------------------------
MAX_POS   = 300
GRID_SIZE = 2*MAX_POS + 1
X = np.arange(-MAX_POS, MAX_POS+1)

theta0 = np.pi/4
phi0   = np.pi/2

# 히스토리 버퍼
HIST_MAX = 2000

# -----------------------------
# 상태/유틸
# -----------------------------
def reset_states(theta, phi):
    global psi_L, psi_R, P_classical, steps
    global Pq_hist, Pc_hist, hist_len
    psi_L = np.zeros(GRID_SIZE, dtype=complex)
    psi_R = np.zeros(GRID_SIZE, dtype=complex)
    psi_L[MAX_POS] = 1/np.sqrt(2)
    psi_R[MAX_POS] = np.exp(1j*phi)/np.sqrt(2)
    P_classical = np.zeros(GRID_SIZE, dtype=float); P_classical[MAX_POS] = 1.0
    Pq_hist = np.zeros((HIST_MAX, GRID_SIZE), dtype=float)
    Pc_hist = np.zeros((HIST_MAX, GRID_SIZE), dtype=float)
    hist_len = 0
    steps = 0
    push_history()
    update_plots()

def push_history():
    global Pq_hist, Pc_hist, hist_len
    Pq = quantum_prob(); Pc = P_classical.copy()
    if Pq.sum() > 0: Pq /= Pq.sum()
    if Pc.sum() > 0: Pc /= Pc.sum()
    if hist_len < HIST_MAX:
        Pq_hist[hist_len, :] = Pq; Pc_hist[hist_len, :] = Pc; hist_len += 1
    else:
        Pq_hist[:-1, :] = Pq_hist[1:, :]; Pc_hist[:-1, :] = Pc_hist[1:, :]
        Pq_hist[-1, :] = Pq; Pc_hist[-1, :] = Pc

def coin_cs(theta):
    return np.cos(theta), np.sin(theta)

def step_quantum(theta, boundary_mode):
    global psi_L, psi_R
    c, s = coin_cs(theta)
    Lc = c*psi_L + s*psi_R
    Rc = s*psi_L - c*psi_R
    new_L = np.zeros_like(psi_L); new_R = np.zeros_like(psi_R)
    new_L[1:]  += Lc[:-1]   # L -> x-1
    new_R[:-1] += Rc[1:]    # R -> x+1
    if boundary_mode == 'reflect':
        new_R[0]  += Lc[0]    # 왼끝 반사
        new_L[-1] += Rc[-1]   # 오른끝 반사
    psi_L, psi_R = new_L, new_R

def step_classic(theta, boundary_mode):
    global P_classical
    p = np.sin(theta)**2
    old = P_classical; new = np.zeros_like(old)
    new[1:-1] = p*old[0:-2] + (1-p)*old[2:]
    if boundary_mode == 'absorb':
        new[0]  += (1-p)*old[1]; new[-1] += p*old[-2]
    else:
        new[0]  += (1-p)*old[1]; new[-1] += p*old[-2]
        new[1]  += (1-p)*old[0]; new[-2] += p*old[-1]
    P_classical = new

def quantum_prob():
    return np.abs(psi_L)**2 + np.abs(psi_R)**2

def stats_from_pdf(x, p):
    Z = p.sum()
    if Z <= 0: return 0.0, 0.0
    p = p/Z; m = (x*p).sum(); v = ((x-m)**2*p).sum()
    return m, float(np.sqrt(max(v, 0.0)))

# -----------------------------
# 플롯/위젯
# -----------------------------
plt.close('all')
fig = plt.figure(figsize=(12.8, 7.6))

ax3d = fig.add_axes([0.07, 0.55, 0.86, 0.38], projection='3d')
ax_q = fig.add_axes([0.07, 0.28, 0.41, 0.20])
ax_c = fig.add_axes([0.52, 0.28, 0.41, 0.20])

ax_th  = fig.add_axes([0.07, 0.18, 0.25, 0.03])
ax_ph  = fig.add_axes([0.37, 0.18, 0.25, 0.03])
ax_n3d = fig.add_axes([0.67, 0.18, 0.26, 0.03])
ax_spf = fig.add_axes([0.07, 0.12, 0.25, 0.03])
ax_mxs = fig.add_axes([0.37, 0.12, 0.25, 0.03])

ax_radio_b = fig.add_axes([0.67, 0.09, 0.12, 0.10])  # boundary
ax_radio_3 = fig.add_axes([0.81, 0.09, 0.12, 0.10])  # 3D target

txt = fig.text(0.07, 0.03, "", fontsize=10, family='monospace')

# 슬라이더 (키워드 인자)
s_theta = Slider(ax_th,  label="θ (coin angle)",         valmin=0.0,    valmax=np.pi/2, valinit=theta0)
s_phi   = Slider(ax_ph,  label="φ (initial coin phase)", valmin=-np.pi, valmax=np.pi,   valinit=phi0)
s_n3d   = Slider(ax_n3d, label="3D steps shown (N3D)",   valmin=20,     valmax=HIST_MAX, valinit=200, valstep=10)
s_spf   = Slider(ax_spf, label="SPF (steps per frame)",  valmin=1,      valmax=50,      valinit=5,    valstep=1)
s_mxs   = Slider(ax_mxs, label="MAX steps (auto-stop)",  valmin=100,    valmax=HIST_MAX, valinit=1000, valstep=50)

radio_boundary = RadioButtons(ax_radio_b, labels=("absorb", "reflect"), active=0)
radio_3dtarget = RadioButtons(ax_radio_3, labels=("Quantum", "Classical"), active=0)

# -----------------------------
# 렌더
# -----------------------------
def draw_2d_panels():
    Pq = quantum_prob(); Pc = P_classical.copy()
    if Pq.sum() > 0: Pq /= Pq.sum()
    if Pc.sum() > 0: Pc /= Pc.sum()
    ax_q.cla(); ax_q.plot(X, Pq, lw=2)
    ax_q.set_xlim([-60, 60]); ax_q.set_ylim([0, max(1e-6, Pq.max())*1.1])
    ax_q.set_title("Quantum Walk: P_Q(x)"); ax_q.set_xlabel("x"); ax_q.set_ylabel("prob")
    ax_c.cla(); ax_c.plot(X, Pc, lw=2)
    ax_c.set_xlim([-60, 60]); ax_c.set_ylim([0, max(1e-6, Pc.max())*1.1])
    ax_c.set_title("Classical Walk: P_C(x)"); ax_c.set_xlabel("x"); ax_c.set_ylabel("prob")

def draw_3d_surface():
    ax3d.cla()
    N3D = int(s_n3d.val); N = min(hist_len, N3D)
    if N <= 1:
        ax3d.set_title("3D surface (waiting...)"); return
    target = radio_3dtarget.value_selected
    H = Pq_hist if target == "Quantum" else Pc_hist
    Z = H[max(0, hist_len-N):hist_len, :]
    T = np.arange(Z.shape[0])
    Xg, Tg = np.meshgrid(X, T)
    ax3d.plot_surface(Xg, Tg, Z, linewidth=0, antialiased=True, rstride=2, cstride=8)
    ax3d.set_xlabel("position x"); ax3d.set_ylabel("time (recent)"); ax3d.set_zlabel("probability")
    ax3d.set_title(f"3D Surface — {target} (last {N} steps)")
    ax3d.view_init(elev=30, azim=-60)

def update_metrics_text():
    Pq = quantum_prob(); Pc = P_classical.copy()
    if Pq.sum() > 0: Pq /= Pq.sum()
    if Pc.sum() > 0: Pc /= Pc.sum()
    mq, sq = stats_from_pdf(X, Pq); mc, sc = stats_from_pdf(X, Pc)
    p = np.sin(s_theta.val)**2
    txt.set_text(
        f"steps = {steps:4d}/{int(s_mxs.val):d} | θ={s_theta.val:.4f} rad (Hadamard=π/4) | φ={s_phi.val:.4f} rad | p_classic=sin^2θ={p:.3f} | "
        f"boundary={radio_boundary.value_selected} | SPF={int(s_spf.val)}\n"
        f"Quantum  : <x>={mq:+.3f}, σ={sq:.3f}\n"
        f"Classical: <x>={mc:+.3f}, σ={sc:.3f}   (quantum: ~linearly spread, classic: ~√t)"
    )

def update_plots():
    draw_2d_panels(); draw_3d_surface(); update_metrics_text()
    fig.canvas.draw_idle()

# -----------------------------
# 오토플레이 타이머
# -----------------------------
running = True
timer = fig.canvas.new_timer(interval=50)  # 20 FPS 정도
def on_timer():
    global steps, running
    if not running: return
    if steps >= int(s_mxs.val):
        running = False
        return
    theta = s_theta.val
    bmode = radio_boundary.value_selected
    n = int(s_spf.val)
    for _ in range(n):
        step_quantum(theta, bmode)
        step_classic(theta, bmode)
        steps += 1
        push_history()
    update_plots()
timer.add_callback(on_timer)
timer.start()

# -----------------------------
# 이벤트
# -----------------------------
def on_theta(val):      pass          # 다음 스텝부터 반영
def on_phi(val):        pass          # 리셋 후 반영
def on_n3d(val):        update_plots()
def on_spf(val):        None          # 속도는 다음 틱부터 반영
def on_maxsteps(val):   None
def on_radio_b(label):  update_plots()
def on_radio_3(label):  update_plots()

def on_key(event):
    global running
    if event.key == ' ':
        running = not running
    elif event.key in ('r', 'R'):
        reset_states(s_theta.val, s_phi.val)
        running = True
    elif event.key in ('left',):
        # 한 프레임만 뒤로는 구현 복잡 → 대신 일시정지
        running = False

fig.canvas.mpl_connect('key_press_event', on_key)

s_theta.on_changed(on_theta); s_phi.on_changed(on_phi)
s_n3d.on_changed(on_n3d); s_spf.on_changed(on_spf); s_mxs.on_changed(on_maxsteps)
radio_boundary.on_clicked(on_radio_b); radio_3dtarget.on_clicked(on_radio_3)

# 시작
reset_states(theta0, phi0)
st.pyplot(fig)
