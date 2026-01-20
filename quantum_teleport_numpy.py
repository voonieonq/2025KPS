# teleport_roulette.py
# Quantum Teleportation visualized ONLY with spinning roulettes (no Streamlit, no Bloch sphere)
# - Left: input |ψ> controls (θ, φ sliders)
# - Center: Bell outcome roulette (00,01,10,11) spins and stops (random, true measurement)
# - Right: Bob correction roulette (I, X, Z, XZ) spins and stops; fidelity shown as text
# - Buttons: [Bell measure] -> [Send classical bits] -> [Reset]

import streamlit as st
import matplotlib
matplotlib.use("TkAgg")  # interactive window
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# ========= Linear algebra & gates =========
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
H  = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
P0 = np.array([[1,0],[0,0]], dtype=complex)
P1 = np.array([[0,0],[0,1]], dtype=complex)

def kron(*ops):
    out = np.array([[1.0+0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

def apply_single(state, U, n, target):
    ops = [U if q==target else I2 for q in range(n)]
    Ufull = ops[0]
    for q in range(1, n):
        Ufull = np.kron(Ufull, ops[q])
    return Ufull @ state

def apply_CNOT(state, n, control, target):
    # Sum_k (|0><0|_c ⊗ I_t + |1><1|_c ⊗ X_t) ⊗ I_rest
    proj0_ops, proj1_ops = [], []
    for q in range(n):
        if q==control:
            proj0_ops.append(P0); proj1_ops.append(P1)
        elif q==target:
            proj0_ops.append(I2); proj1_ops.append(X)
        else:
            proj0_ops.append(I2); proj1_ops.append(I2)
    P0_full = proj0_ops[0]; P1_full = proj1_ops[0]
    for q in range(1, n):
        P0_full = np.kron(P0_full, proj0_ops[q])
        P1_full = np.kron(P1_full, proj1_ops[q])
    return (P0_full + P1_full) @ state

def projector_on_qubit(n, target, outcome):
    ops = []
    for q in range(n):
        if q==target:
            ops.append(P0 if outcome==0 else P1)
        else:
            ops.append(I2)
    P_full = ops[0]
    for q in range(1,n):
        P_full = np.kron(P_full, ops[q])
    return P_full

def measure_qubit(state, n, target, rng):
    P0_full = projector_on_qubit(n, target, 0)
    P1_full = projector_on_qubit(n, target, 1)
    psi0 = P0_full @ state
    psi1 = P1_full @ state
    p0 = float(np.vdot(psi0, psi0).real)
    p1 = float(np.vdot(psi1, psi1).real)
    s = p0 + p1
    if s <= 0:  # should not happen for proper state
        raise RuntimeError("Degenerate measurement probabilities.")
    p0, p1 = p0/s, p1/s
    outcome = int(rng.choice([0,1], p=[p0,p1]))
    post = normalize(psi0 if outcome==0 else psi1)
    return outcome, post

def density_from_state(psi):
    v = psi.reshape(-1,1)
    return v @ v.conj().T

def partial_trace_3qubit(rho, keep):  # keep in {0,1,2}, order A,B,C
    rho6 = rho.reshape(2,2,2, 2,2,2)  # (a,b,c,a',b',c')
    if keep == 0:   # keep A
        tmp = np.trace(rho6, axis1=1, axis2=4)  # trace over B
        red = np.trace(tmp,  axis1=1, axis2=3)  # trace over C
    elif keep == 1: # keep B
        tmp = np.trace(rho6, axis1=0, axis2=3)  # trace over A
        red = np.trace(tmp,  axis1=1, axis2=3)  # trace over C
    else:           # keep C
        tmp = np.trace(rho6, axis1=0, axis2=3)  # trace over A
        red = np.trace(tmp,  axis1=0, axis2=2)  # trace over B
    return red

def fidelity_pure(psi_vec, rho1q):
    v = psi_vec.reshape(2,1)
    return float(np.real(v.conj().T @ rho1q @ v))

# ========= Teleportation core =========
rng = np.random.default_rng(123)

def prepare_state(theta, phi):
    # |ψ> = cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>
    return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)

def prepare_abc(psi):  # A=|ψ>, B=|0>, C=|0>
    return kron(psi.reshape(2,1), np.array([[1],[0]], dtype=complex), np.array([[1],[0]], dtype=complex)).reshape(-1)

def make_EPR_on_BC(state):
    # Create Bell pair between B(1) and C(2)
    s = apply_single(state, H, 3, target=1)      # H on B
    s = apply_CNOT(s,   3, control=1, target=2)  # CNOT B->C
    return s

def alice_bell_measure(state):
    # Bell basis on (A,B): CNOT A->B; H on A; then measure A, B in Z
    s = apply_CNOT(state, 3, control=0, target=1)
    s = apply_single(s, H,   3, target=0)
    mA, s = measure_qubit(s, 3, target=0, rng=rng)
    mB, s = measure_qubit(s, 3, target=1, rng=rng)
    return (int(mA), int(mB)), s  # s is post-measurement state

def bob_correct(state, mA, mB):
    s = state.copy()
    if mB == 1:  # X correction
        s = apply_single(s, X, 3, target=2)
    if mA == 1:  # Z correction
        s = apply_single(s, Z, 3, target=2)
    return s

# ========= Roulette drawing helpers =========
PALETTE = ["#4CAF50", "#F44336", "#FF9800", "#2196F3", "#9C27B0", "#009688"]

def build_wheel(ax, labels, radius=1.5):
    ax.set_aspect('equal')
    pad = 0.3  # 가장자리 여유
    ax.set_xlim(-(radius+pad),  (radius+pad))
    ax.set_ylim(-(radius+pad+0.2), (radius+pad))  # 위쪽 포인터 공간 조금 더

    ax.axis('off')
    n = len(labels)
    wedges = []
    base = []
    for k in range(n):
        t1 = (360.0/n)*k
        t2 = (360.0/n)*(k+1)
        w = Wedge(
            (0,0), radius, t1, t2,
            facecolor=PALETTE[k % len(PALETTE)],  # ✅ 팔레트 색 사용
            edgecolor="black", lw=1.4
        )
        ax.add_patch(w)
        wedges.append(w)
        base.append((t1,t2))

    hub = Circle((0,0), 0.08*radius, color="black"); ax.add_patch(hub)

    # 룰렛과 같이 회전하는 점(도트)
    dot, = ax.plot([], [], 'o', ms=8, mfc='white', mec='black')

    # 고정 포인터(위쪽)
    ax.plot([0,0],[radius*1.05, radius*1.28], lw=4, color='black')
    ax.text(0, radius*1.33, "▲", ha="center", va="center", fontsize=16)

    # 결과 중앙 텍스트
    center = ax.text(
        0, 0, "?", ha="center", va="center",
        fontsize=22, color="white", weight='bold',
        bbox=dict(boxstyle="circle,pad=0.35", fc="#00000088", ec="none")
    )

    # 섹션 라벨
    text_objs = []
    for k, lab in enumerate(labels):
        ang = np.radians((base[k][0] + base[k][1])/2)
        r = radius * 10
        txt = ax.text(r*np.cos(ang), r*np.sin(ang), lab,
                      ha="center", va="center", fontsize=12, weight='bold')
        text_objs.append(txt)

    # radius를 UI 딕셔너리에 넣어두면 dot 위치 계산에 사용 가능
    return {"wedges":wedges, "base":base, "dot":dot, "center":center,
            "labels":labels, "radius":radius}

    # center label (result)
    center = ax.text(0, 0, "?", ha="center", va="center",
                     fontsize=22, color="white", weight='bold',
                     bbox=dict(boxstyle="circle,pad=0.35", fc="#00000088", ec="none"))
    # labels around
    text_objs = []
    for k, lab in enumerate(labels):
        ang = np.radians((base[k][0] + base[k][1])/2)
        r = 1.4
        txt = ax.text(r*np.cos(ang), r*np.sin(ang), lab,
                      ha="center", va="center", fontsize=12, weight='bold')
        text_objs.append(txt)
    return {"wedges":wedges, "base":base, "dot":dot, "center":center, "labels":labels}

def set_wheel_angle(ui, angle_rad):
    a = (np.degrees(angle_rad)) % 360.0
    for w, (b1,b2) in zip(ui["wedges"], ui["base"]):
        w.theta1 = (b1 + a) % 360.0
        w.theta2 = (b2 + a) % 360.0
    r = ui.get("radius", 1.0) * 0.92   # ✅ 반지름 기반
    ui["dot"].set_data([r*np.cos(angle_rad)], [r*np.sin(angle_rad)])

def angle_to_index(angle_rad, n):
    # pointer at π/2 (top). Shift wheel by -π/2 to map top to angle 0.
    a = (angle_rad - np.pi/2) % (2*np.pi)
    frac = a/(2*np.pi)  # [0,1)
    idx = (int(np.floor(frac * n)) ) % n
    # Note: since we rotate the wheel, this mapping is consistent
    return idx

def index_to_target_angle(idx, n):
    # Target so that segment idx center hits pointer at π/2
    seg_size = 2*np.pi/n
    seg_center = idx*seg_size + seg_size/2
    target = (np.pi/2) - seg_center  # solve (wheel_angle + seg_center) = π/2
    return target % (2*np.pi)

# ========= Figure & UI =========
fig = plt.figure(figsize=(11.5, 6.4))
gs = fig.add_gridspec(3, 4, height_ratios=[3.1, 0.9, 0.9])

# Wheels
ax_bell = fig.add_subplot(gs[0, 1])  # center-left
ax_op   = fig.add_subplot(gs[0, 2])  # center-right
ui_bell = build_wheel(ax_bell, ["00","01","10","11"])
ui_op   = build_wheel(ax_op,   ["I","X","Z","XZ"])
ax_bell.set_title("Alice Bell outcome (mA mB)")
ax_op.set_title("Bob correction")

# Controls (θ, φ sliders on the left)
ax_theta = fig.add_subplot(gs[1, 0]); ax_phi = fig.add_subplot(gs[2, 0])
theta_slider = Slider(ax_theta, "θ (0..π)", 0.0, np.pi, valinit=np.pi/3)
phi_slider   = Slider(ax_phi,   "φ (0..2π)", 0.0, 2*np.pi, valinit=np.pi/4)

# Buttons on the right
btn_ax1 = fig.add_subplot(gs[1, 3]); btn_ax1.axis('off')
btn_ax2 = fig.add_subplot(gs[2, 3]); btn_ax2.axis('off')
btn_bell_ax = fig.add_axes([0.61, 0.15, 0.12, 0.06])
btn_send_ax = fig.add_axes([0.77, 0.15, 0.12, 0.06])
btn_reset_ax= fig.add_axes([0.69, 0.06, 0.12, 0.06])

btn_bell = Button(btn_bell_ax, "Bell measure")
btn_send = Button(btn_send_ax, "Send classical bits")
btn_reset= Button(btn_reset_ax,"Reset")

# Text panels
ax_info  = fig.add_subplot(gs[1,1]); ax_info.axis('off')
ax_info2 = fig.add_subplot(gs[1,2]); ax_info2.axis('off')
ax_info3 = fig.add_subplot(gs[2,1:3]); ax_info3.axis('off')
info_txt  = ax_info.text(0,1.0,"",va="top",fontsize=11)
info2_txt = ax_info2.text(0,1.0,"",va="top",fontsize=11)
status_txt= ax_info3.text(0,1.0,"",va="top",fontsize=11, family="monospace")

# ========= Animation state =========
spin_bell  = False
spin_op    = False
angle_bell = 0.0
angle_op   = 0.0
FPS = 60
dt  = 1.0/FPS
spin_speed_bell = 7.0  # rad/s
spin_speed_op   = 7.0
spin_t_total   = 2.0   # seconds of spin after click
spin_t         = 0.0
target_idx_bell = None
target_idx_op   = None
phase = "idle"  # "idle" -> "bell_spinning" -> "bell_done" -> "op_spinning" -> "done"

# Quantum state cache for current θ,φ
psi_in = None
state_prep = None
state_after_epr = None
post_meas_state = None
mA = None; mB = None

def update_info():
    theta = theta_slider.val; phi = phi_slider.val
    info_txt.set_text(
        "Input |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩\n"
        f"θ = {theta:.2f} rad,  φ = {phi:.2f} rad\n"
        "Steps:\n"
        " 1) Bell measure on A,B  → random bits (mA,mB)\n"
        " 2) Send classical bits  → Bob applies X^mB Z^mA\n"
        " 3) Bob's qubit becomes |ψ⟩"
    )
    if mA is None:
        info2_txt.set_text("Bell outcome (mA,mB):  (not measured yet)\nBob correction:  (waiting)")
    else:
        corr = ("I","X","Z","XZ")[(mA<<1)|mB] if False else None  # not used directly
        info2_txt.set_text(
            f"Bell outcome (mA,mB):  ({mA},{mB})\n"
            f"Bob correction rule :  X^mB Z^mA"
        )

def reset_all(_evt=None):
    global psi_in, state_prep, state_after_epr, post_meas_state
    global mA, mB, phase, spin_bell, spin_op, spin_t, target_idx_bell, target_idx_op
    psi_in = prepare_state(theta_slider.val, phi_slider.val)
    state_prep = prepare_abc(psi_in)
    state_after_epr = make_EPR_on_BC(state_prep)
    post_meas_state = None
    mA = None; mB = None
    phase = "idle"
    spin_bell = spin_op = False
    spin_t = 0.0
    target_idx_bell = target_idx_op = None
    ui_bell["center"].set_text("?")
    ui_op["center"].set_text("?")
    status_txt.set_text("Ready. Click 'Bell measure'.")
    update_info()
    fig.canvas.draw_idle()

def on_sliders(_):
    # When θ,φ change, reset (state changes)
    reset_all()

def start_bell(_evt=None):
    global phase, spin_bell, spin_t, target_idx_bell, mA, mB, post_meas_state
    if phase not in ("idle",):
        return
    # Do the ACTUAL Bell measurement to get (mA,mB)
    psi_in_local = prepare_state(theta_slider.val, phi_slider.val)
    s0 = prepare_abc(psi_in_local)
    s1 = make_EPR_on_BC(s0)
    (ma, mb), post = alice_bell_measure(s1)
    # cache
    mA, mB = ma, mb
    post_meas_state = post
    update_info()
    # roulette target index mapping (00,01,10,11) -> 0,1,2,3  using (mA<<1)|mB
    target_idx_bell = (mA<<1) | mB  # 0..3
    spin_bell = True
    spin_t = 0.0
    phase = "bell_spinning"
    status_txt.set_text(f"Bell measuring... target = {ui_bell['labels'][target_idx_bell]}")

def start_send(_evt=None):
    global phase, spin_op, spin_t, target_idx_op
    if phase != "bell_done":
        return
    # determine operation index mapping from (mA,mB):
    # Apply X if mB=1, Z if mA=1 → ops order [I, X, Z, XZ] indexed by (mA<<1)|mB
    op_idx = (mA<<1)|mB  # 00->0(I), 01->1(X), 10->2(Z), 11->3(XZ)
    target_idx_op = op_idx
    spin_op = True
    spin_t = 0.0
    phase = "op_spinning"
    status_txt.set_text(f"Sending bits... Bob will apply: {ui_op['labels'][target_idx_op]}")

def ease_out_cubic(t):
    return 1 - (1 - t)**3

def animate(_frame):
    global angle_bell, angle_op, spin_t, spin_bell, spin_op, phase
    # base slow drift
    angle_bell = (angle_bell + 0.6*dt) % (2*np.pi)
    angle_op   = (angle_op   + 0.6*dt) % (2*np.pi)

    if spin_bell:
        spin_t += dt
        t = min(spin_t / spin_t_total, 1.0)
        factor = 1.0 - ease_out_cubic(t)  # 1→0 (decelerate)
        angle_bell = (angle_bell + spin_speed_bell*factor*dt) % (2*np.pi)
        if t >= 1.0:
            # snap to target segment center
            tgt = index_to_target_angle(target_idx_bell, 4)
            angle_bell = tgt
            ui_bell["center"].set_text(ui_bell["labels"][target_idx_bell])
            spin_bell = False
            phase = "bell_done"
            status_txt.set_text(f"Bell done: (mA,mB)=({mA},{mB}). Click 'Send classical bits'.")

    if spin_op:
        spin_t += dt
        t = min(spin_t / spin_t_total, 1.0)
        factor = 1.0 - ease_out_cubic(t)
        angle_op = (angle_op + spin_speed_op*factor*dt) % (2*np.pi)
        if t >= 1.0:
            tgt = index_to_target_angle(target_idx_op, 4)
            angle_op = tgt
            ui_op["center"].set_text(ui_op["labels"][target_idx_op])
            spin_op = False
            phase = "done"
            # Apply actual correction & report fidelity
            corrected = bob_correct(post_meas_state, mA, mB)
            rho_out = partial_trace_3qubit(density_from_state(corrected), keep=2)
            psi_local = prepare_state(theta_slider.val, phi_slider.val)
            F = fidelity_pure(psi_local, rho_out)
            status_txt.set_text(
                f"Bob applied {ui_op['labels'][target_idx_op]}  →  Fidelity to |ψ⟩ = {F:.6f}  (ideal ≈ 1)"
            )

    set_wheel_angle(ui_bell, angle_bell)
    set_wheel_angle(ui_op,   angle_op)
    return []

# Wire up controls
theta_slider.on_changed(on_sliders)
phi_slider.on_changed(on_sliders)
btn_bell.on_clicked(start_bell)
btn_send.on_clicked(start_send)
btn_reset.on_clicked(reset_all)

# Init
reset_all()

ani = FuncAnimation(fig, animate, interval=1000/FPS, blit=False, repeat=True)
plt.tight_layout()

st.pyplot(fig)
