# wavefunction_focus_sim.py (refined)
# Wavefunction-focused 1D TDSE simulator (ħ=1, m=1)
# - Split-Operator FFT (Strang) time evolution
# - Focus on ψ: |ψ|², phase arg(ψ), probability current j(x)=Im(ψ*∂xψ)
# - Continuity check: dρ/dt + ∂x j ≈ 0 (RMS)
# - Initial modes: Single / Two(interference) / HO n=0 / HO n=1
# - Global phase φ_g slider (visual only; physics invariant)
# - Autoplay: Space (toggle), R (reset)
# - Complex Absorbing Potential (CAP) for edges
# - Matplotlib >= 3.7 (Slider keyword args)
#
# Changes vs your draft:
# - φ_g, dt, CAP, potential params now update without resetting ψ
# - Barrier width/center sliders added
# - HO mode auto-switches the potential radio to "Harmonic"
# - Single timer instance; no multiple windows/timers


#폰트사이즈 줄여도 겹침 문제 해결이 안됨.. 근데 일단 넣어놓긴했음
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 5 #기본 폰트 크기 12


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (backend registration)

# ==========================
# Grid / base parameters
# ==========================
L = 160.0
N = 2**12
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1]-x[0]

k = 2*np.pi*np.fft.fftfreq(N, d=dx)
k = np.fft.fftshift(k)
dk = k[1]-k[0]

dt0    = 0.04
cap0   = 0.02
phi_g0 = 0.0  # global phase (display only)

# ==========================
# Utils / math
# ==========================
def normalize(psi):
    n = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi if n == 0 else psi / n

def fft_psi_to_phi(psi):
    # φ(k) = FFT[ψ(x)] * dx/√(2π)
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2*np.pi)

def ifft_phi_to_psi(phi):
    # ψ(x) = IFFT[φ(k)] * √(2π)/dx
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(phi))) * np.sqrt(2*np.pi) / dx

def gaussian_packet(x, x0, k0, sigma):
    A = (1/(2*np.pi*sigma**2))**0.25
    return A * np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)

def grad_x(f):
    return np.gradient(f, dx, edge_order=2)

def prob_current(psi):
    # j = Im(ψ* ∂x ψ) / m (m=1, ħ=1)
    dpsi = grad_x(psi)
    return np.imag(np.conj(psi) * dpsi)

def moments_x(psi):
    ρ = np.abs(psi)**2; Z = np.sum(ρ)*dx
    if Z <= 0: return 0.0, 0.0
    ρ = ρ/Z
    μ = np.sum(x*ρ)*dx
    σ2 = np.sum((x-μ)**2 * ρ)*dx
    return μ, np.sqrt(max(σ2,0.0))

# ==========================
# Potentials
# ==========================
def V_free():
    return np.zeros_like(x)

def V_harmonic(omega=0.05):
    return 0.5*(omega**2)*x**2

def V_barrier(height=0.06, width=10.0, center=0.0):
    return height * (np.abs(x-center) < width/2)

def build_V(kind, p1):  # kept for compatibility (height / ω only)
    if kind == "Free":      return V_free()
    if kind == "Harmonic":  return V_harmonic(p1)
    if kind == "Barrier":   return V_barrier(height=p1)
    return V_free()

# current, slider-aware potential constructor
def build_V_current():
    if pot_kind == "Free":
        return V_free()
    elif pot_kind == "Harmonic":
        return V_harmonic(s_pot.val)
    elif pot_kind == "Barrier":
        return V_barrier(height=s_pot.val, width=s_bw.val, center=s_bc.val)
    else:
        return V_free()

# ==========================
# HO eigenstates (approx)
# ==========================
def ho_ground(omega):
    a = (omega/np.pi)**0.25
    return a * np.exp(-0.5*omega*x**2)

def ho_first(omega):
    a = (omega/np.pi)**0.25
    return np.sqrt(2)* (omega**0.5) * x * a * np.exp(-0.5*omega*x**2)

# ==========================
# CAP (complex absorbing boundary)
# ==========================
def cap_profile(strength):
    eta = np.zeros_like(x)
    edge = 0.35*L
    maskL = x < -edge
    maskR = x >  edge
    # smooth quadratic ramp
    eta[maskL] = ((-x[maskL]-edge)/(L/2 - edge))**2
    eta[maskR] = (( x[maskR]-edge)/(L/2 - edge))**2
    return strength * eta

# ==========================
# Split-Operator (Strang)
# ==========================
def kinetic_phase(dt):
    return np.exp(-0.5j * (k**2) * dt)  # m=1

def propagate(psi, Vx, dt, cap_s):
    W = Vx - 1j*cap_profile(cap_s)
    # half V
    psi = psi * np.exp(-1j*W*dt/2)
    # full T
    phi = fft_psi_to_phi(psi)
    phi = phi * kinetic_phase(dt)
    psi = ifft_phi_to_psi(phi)
    # half V
    psi = psi * np.exp(-1j*W*dt/2)
    return psi

# ==========================
# Initial state modes
# ==========================
mode_init = "Single"
pot_kind  = "Free"
omega0    = 0.05  # for HO / harmonic

# default initial (Single/Two)
x0_1, k0_1, s1 = -35.0,  1.1, 6.0
x0_2, k0_2, s2 = +35.0, -1.1, 6.0
alpha2, dphi = 0.8, 0.0

def make_initial_state():
    global pot_kind
    if mode_init == "Single":
        psi = gaussian_packet(x, s_x01.val, s_k01.val, s_s1.val)
        Vx  = build_V_current()
    elif mode_init == "Two":
        psi1 = gaussian_packet(x, s_x01.val, s_k01.val, s_s1.val)
        psi2 = gaussian_packet(x, s_x02.val, s_k02.val, s_s2.val) * (s_a2.val*np.exp(1j*s_dphi.val))
        psi  = psi1 + psi2
        Vx   = build_V_current()
    elif mode_init == "HO n=0":
        # ensure harmonic potential is selected in UI
        pot_radio.set_active(1)  # 0:Free, 1:Harmonic, 2:Barrier
        pot_kind = "Harmonic"
        Vx  = build_V_current()
        psi = ho_ground(s_pot.val)
    elif mode_init == "HO n=1":
        pot_radio.set_active(1)
        pot_kind = "Harmonic"
        Vx  = build_V_current()
        psi = ho_first(s_pot.val)
    else:
        psi = gaussian_packet(x, s_x01.val, s_k01.val, s_s1.val)
        Vx  = build_V_current()
    return normalize(psi), Vx

# ==========================
# State / buffers
# ==========================
psi  = np.zeros_like(x, dtype=complex)
Vx   = np.zeros_like(x, dtype=float)
ρ_prev = None

# ==========================
# Figure / Axes
# ==========================
plt.close('all')
fig = plt.figure(figsize=(12.8, 8.0))

ax_top = fig.add_axes([0.10, 0.62, 0.83, 0.30])  # |ψ|² + V
ax_mid = fig.add_axes([0.10, 0.37, 0.83, 0.18])  # phase
ax_bot = fig.add_axes([0.10, 0.12, 0.83, 0.18])  # current j(x)

# left radios
ax_mode = fig.add_axes([0.01, 0.62, 0.08, 0.30])
ax_pot  = fig.add_axes([0.01, 0.28, 0.08, 0.20])

# main sliders bottom
ax_dt   = fig.add_axes([0.10, 0.06, 0.18, 0.03])
ax_cap  = fig.add_axes([0.30, 0.06, 0.18, 0.03])
ax_phiG = fig.add_axes([0.50, 0.06, 0.18, 0.03])
ax_potv = fig.add_axes([0.70, 0.06, 0.23, 0.03])

# initial-state params (two rows)
ax_x01 = fig.add_axes([0.10, 0.02, 0.14, 0.025])
ax_k01 = fig.add_axes([0.25, 0.02, 0.14, 0.025])
ax_s1  = fig.add_axes([0.40, 0.02, 0.14, 0.025])

ax_x02 = fig.add_axes([0.55, 0.02, 0.14, 0.025])
ax_k02 = fig.add_axes([0.70, 0.02, 0.14, 0.025])
ax_s2  = fig.add_axes([0.85, 0.02, 0.08, 0.025])

# Two-packet amplitude/phase
ax_a2   = fig.add_axes([0.55, 0.095, 0.14, 0.025])
ax_dphi = fig.add_axes([0.70, 0.095, 0.14, 0.025])

# Barrier extras: width, center
ax_bw = fig.add_axes([0.10, 0.095, 0.14, 0.025])
ax_bc = fig.add_axes([0.25, 0.095, 0.14, 0.025])

# text
txt = fig.text(0.10, 0.92, "", fontsize=10, family="monospace")

# ==========================
# Widgets (keyword-only)
# ==========================
mode_radio = RadioButtons(ax_mode, labels=("Single","Two","HO n=0","HO n=1"), active=0)
pot_radio  = RadioButtons(ax_pot,  labels=("Free","Harmonic","Barrier"), active=0)

s_dt   = Slider(ax_dt,   label="dt",   valmin=0.005, valmax=0.15,  valinit=dt0)
s_cap  = Slider(ax_cap,  label="CAP",  valmin=0.0,   valmax=0.2,   valinit=cap0)
s_phiG = Slider(ax_phiG, label="Global phase φ_g (display)", valmin=-np.pi, valmax=np.pi, valinit=phi_g0)
s_pot  = Slider(ax_potv, label="Potential param (ω or height)", valmin=0.01, valmax=0.12, valinit=omega0)

# initial (Single/Two)
s_x01 = Slider(ax_x01, label="x01", valmin=-L/2+10, valmax=L/2-10, valinit=x0_1)
s_k01 = Slider(ax_k01, label="k01", valmin=-2.5,    valmax=2.5,    valinit=k0_1)
s_s1  = Slider(ax_s1,  label="σ1",  valmin=1.0,     valmax=20.0,   valinit=s1)

s_x02 = Slider(ax_x02, label="x02", valmin=-L/2+10, valmax=L/2-10, valinit=x0_2)
s_k02 = Slider(ax_k02, label="k02", valmin=-2.5,    valmax=2.5,    valinit=k0_2)
s_s2  = Slider(ax_s2,  label="σ2",  valmin=1.0,     valmax=20.0,   valinit=s2)

s_a2   = Slider(ax_a2,   label="α2 (amp)",  valmin=0.0,  valmax=1.5,  valinit=alpha2)
s_dphi = Slider(ax_dphi, label="Δφ (rad)",  valmin=-np.pi, valmax=np.pi, valinit=dphi)

# barrier extras
s_bw = Slider(ax_bw, label="Barrier width",  valmin=2.0,   valmax=40.0,  valinit=10.0)
s_bc = Slider(ax_bc, label="Barrier center", valmin=-L/4,  valmax=+L/4,  valinit=0.0)

# ==========================
# Rendering
# ==========================
def redraw_all():
    # global phase only for display
    psi_vis = psi * np.exp(1j * s_phiG.val)

    # |ψ|² + V
    ρ = np.abs(psi_vis)**2
    ρn = ρ / (ρ.max() if ρ.max()>0 else 1.0)

    Vn = Vx.copy()
    Vn -= Vn.min()
    vmax = Vn.max() if Vn.max()>0 else 1.0
    Vn = Vn / vmax

    ax_top.cla()
    ax_top.plot(x, ρn, lw=2, label="|ψ|² (norm)")
    ax_top.plot(x, Vn, lw=1, label="V(x) (scaled)")
    ax_top.set_xlim([-L/2, L/2]); ax_top.set_ylim([0, 1.05])
    ax_top.set_title("|ψ(x,t)|²  +  V(x)")
    ax_top.set_xlabel("x"); ax_top.set_ylabel("normed density"); ax_top.legend(loc="upper right")

    # phase
    ax_mid.cla()
    phase = np.angle(psi_vis)
    ax_mid.plot(x, phase, lw=1)
    ax_mid.set_xlim([-L/2, L/2]); ax_mid.set_ylim([-np.pi, np.pi])
    ax_mid.set_title("Phase arg(ψ(x,t))"); ax_mid.set_xlabel("x"); ax_mid.set_ylabel("phase [rad]")

    # current
    ax_bot.cla()
    j = prob_current(psi)  # phase-invariant
    ax_bot.plot(x, j, lw=1)
    ax_bot.set_xlim([-L/2, L/2])
    ax_bot.set_title("Probability current  j(x) = Im(ψ*∂xψ)")
    ax_bot.set_xlabel("x"); ax_bot.set_ylabel("j(x)")

    # text
    μx, σx = moments_x(psi)
    norm = np.sum(np.abs(psi)**2) * dx
    cont_rms = compute_continuity_RMS(ρ)
    
# 초기 생성 (빈 텍스트)
txt = fig.text(0.10, 0.92, "", fontsize=10, family="monospace")

# redraw_all() 함수 안에서만 업데이트
# ==========================
# Rendering
# ==========================
def redraw_all():
    psi_vis = psi * np.exp(1j * s_phiG.val)

    # 확률밀도
    ρ = np.abs(psi_vis)**2
    ρn = ρ / (ρ.max() if ρ.max()>0 else 1.0)

    # potential normalize
    Vn = Vx.copy()
    Vn -= Vn.min()
    vmax = Vn.max() if Vn.max()>0 else 1.0
    Vn = Vn / vmax

    # plots
    ax_top.cla()
    ax_top.plot(x, ρn, lw=2, label="|ψ|² (norm)")
    ax_top.plot(x, Vn, lw=1, label="V(x) (scaled)")
    ax_top.set_xlim([-L/2, L/2]); ax_top.set_ylim([0, 1.05])
    ax_top.set_title("|ψ(x,t)|²  +  V(x)")
    ax_top.set_xlabel("x"); ax_top.set_ylabel("normed density"); ax_top.legend(loc="upper right")

    ax_mid.cla()
    phase = np.angle(psi_vis)
    ax_mid.plot(x, phase, lw=1)
    ax_mid.set_xlim([-L/2, L/2]); ax_mid.set_ylim([-np.pi, np.pi])
    ax_mid.set_title("Phase arg(ψ(x,t))"); ax_mid.set_xlabel("x"); ax_mid.set_ylabel("phase [rad]")

    ax_bot.cla()
    j = prob_current(psi)
    ax_bot.plot(x, j, lw=1)
    ax_bot.set_xlim([-L/2, L/2])
    ax_bot.set_title("Probability current j(x)")
    ax_bot.set_xlabel("x"); ax_bot.set_ylabel("j(x)")

    # 텍스트
    norm = np.sum(np.abs(psi)**2) * dx
    μx, σx = moments_x(psi)
    cont_rms = compute_continuity_RMS(ρ)

    txt.set_text(
        f"Mode={mode_init} | Potential={pot_kind}\n"
        f"(param={s_pot.val:.3f})\n"
        f"dt={s_dt.val:.4f} | CAP={s_cap.val:.3f}\n"
        f"φ_g={s_phiG.val:+.3f}\n"
        f"Norm={norm:.6f}  <x>={μx:+.3f}\n"
        f"σ_x={σx:.3f}  | continuity RMS ≈ {cont_rms:.3e}"
    )

    fig.canvas.draw_idle()


    fig.canvas.draw_idle()

# continuity RMS
ρ_prev = None
def compute_continuity_RMS(ρ_now):
    global ρ_prev
    if ρ_prev is None:
        return 0.0
    j_now = prob_current(psi)
    dρ_dt = (ρ_now - ρ_prev) / s_dt.val
    dJ_dx = np.gradient(j_now, dx)
    resid = dρ_dt + dJ_dx
    # CAP adds source/sink near edges, so slight mismatch expected there
    rms = np.sqrt(np.mean(resid**2))
    return float(rms)

# ==========================
# Sim loop
# ==========================
running = True
def step_once():
    global psi, ρ_prev
    ρ_prev = np.abs(psi)**2
    psi = propagate(psi, Vx, s_dt.val, s_cap.val)

def on_timer():
    if not running: return
    step_once()
    redraw_all()

timer = fig.canvas.new_timer(interval=40)  # ~25 FPS
timer.add_callback(on_timer)
timer.start()

# ==========================
# Callbacks / init
# ==========================
def reset_state():
    global psi, Vx, ρ_prev, mode_init, pot_kind
    mode_init = mode_radio.value_selected
    pot_kind  = pot_radio.value_selected
    psi, Vx = make_initial_state()
    ρ_prev = None
    redraw_all()

def on_mode(label):
    reset_state()

def on_potential(label):
    # change potential kind; keep current ψ (no reset) unless HO modes are active
    global pot_kind, Vx
    pot_kind = label
    Vx = build_V_current()
    redraw_all()

def on_phase_only(val):
    # global phase only affects display
    redraw_all()

def on_evo_param(val):
    # dt, CAP changed -> just redraw (affects evolution on next steps)
    redraw_all()

def on_potential_param(val):
    # ω / height / width / center changed -> update V only
    global Vx
    Vx = build_V_current()
    redraw_all()

def on_initial_param(val):
    # changing initial-state params rebuilds ψ at t=0
    reset_state()

def on_key(event):
    global running
    if event.key == ' ':
        running = not running
    elif event.key in ('r','R'):
        reset_state()
        running = True

# events
mode_radio.on_clicked(on_mode)
pot_radio.on_clicked(on_potential)

# φ_g updates w/o reset
s_phiG.on_changed(on_phase_only)
# evolution params (no reset)
s_dt.on_changed(on_evo_param)
s_cap.on_changed(on_evo_param)
# potential params (update V only)
s_pot.on_changed(on_potential_param)
s_bw.on_changed(on_potential_param)
s_bc.on_changed(on_potential_param)
# initial-state params -> full reset
for s in (s_x01, s_k01, s_s1, s_x02, s_k02, s_s2, s_a2, s_dphi):
    s.on_changed(on_initial_param)

fig.canvas.mpl_connect('key_press_event', on_key)

# start
reset_state()
plt.tight_layout()

st.pyplot(fig)
