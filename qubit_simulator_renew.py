# qubit_simulator_gui_v2.py
# 1-Qubit simulator: Bloch sphere + probability bar, no clipping
# Requirements: numpy, matplotlib
# Run: python qubit_simulator_gui_v2.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from typing import List

# -----------------------------
# Linear algebra & gates
# -----------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

# basis
e0 = np.array([1.0, 0.0], dtype=complex)
e1 = np.array([0.0, 1.0], dtype=complex)

# gates
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)

def Rx(th):
    c, s = np.cos(th/2), -1j*np.sin(th/2)
    return np.array([[c, s], [s, c]], dtype=complex)

def Ry(th):
    c, s = np.cos(th/2), np.sin(th/2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rz(th):
    return np.array([[np.exp(-1j*th/2), 0],
                     [0, np.exp(1j*th/2)]], dtype=complex)

def bloch_coordinates(state: np.ndarray):
    a, b = state
    x = 2*np.real(np.conj(a)*b)
    y = 2*np.imag(np.conj(a)*b)
    z = (np.abs(a)**2 - np.abs(b)**2).real
    return float(x), float(y), float(z)

def probabilities(state: np.ndarray):
    return float(np.abs(state[0])**2), float(np.abs(state[1])**2)

# -----------------------------
# App state
# -----------------------------
state = e0.astype(complex)
history: List[np.ndarray] = []

def apply_unitary(U: np.ndarray):
    global state, history
    history.append(state.copy())
    state = normalize(U @ state)
    update_visuals()

def undo():
    global state, history
    if history:
        state = history.pop()
        update_visuals()

def reset():
    global state, history
    history.clear()
    state = e0.astype(complex)
    update_visuals()

# -----------------------------
# Figure & layout (no add_axes)
# -----------------------------
plt.close('all')
fig = plt.figure(figsize=(12.5, 7.4), constrained_layout=True)

# 좌: Bloch + 바차트 / 우: 컨트롤 패널
outer = fig.add_gridspec(
    nrows=2, ncols=2,
    width_ratios=[1.25, 1.0],
    height_ratios=[2.0, 1.25]
)

ax_bloch = fig.add_subplot(outer[0, 0], projection='3d')
ax_probs = fig.add_subplot(outer[1, 0])

# 우측 패널을 8x2로 쪼갬
right = outer[:, 1].subgridspec(8, 2, wspace=0.25, hspace=0.35)

# Gate buttons (3행) + Undo/Reset (1행)
ax_btn_H = fig.add_subplot(right[0, 0]); ax_btn_X = fig.add_subplot(right[0, 1])
ax_btn_Y = fig.add_subplot(right[1, 0]); ax_btn_Z = fig.add_subplot(right[1, 1])
ax_btn_S = fig.add_subplot(right[2, 0]); ax_btn_T = fig.add_subplot(right[2, 1])
ax_btn_U = fig.add_subplot(right[3, 0]); ax_btn_R = fig.add_subplot(right[3, 1])

# Sliders (3행, full width)
ax_sx = fig.add_subplot(right[4, :])
ax_sy = fig.add_subplot(right[5, :])
ax_sz = fig.add_subplot(right[6, :])

# Apply buttons (마지막 행은 3개로 다시 분할)
apply_row = right[7, :].subgridspec(1, 3, wspace=0.25)
ax_btn_Rx = fig.add_subplot(apply_row[0, 0])
ax_btn_Ry = fig.add_subplot(apply_row[0, 1])
ax_btn_Rz = fig.add_subplot(apply_row[0, 2])

# 버튼 축은 눈금 제거
for a in [ax_btn_H, ax_btn_X, ax_btn_Y, ax_btn_Z, ax_btn_S, ax_btn_T, ax_btn_U, ax_btn_R,
          ax_btn_Rx, ax_btn_Ry, ax_btn_Rz]:
    a.set_xticks([]); a.set_yticks([])

# -----------------------------
# Bloch sphere (static mesh)
# -----------------------------
def init_bloch():
    ax_bloch.cla()
    u = np.linspace(0, 2*np.pi, 64)
    v = np.linspace(0, np.pi, 32)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax_bloch.plot_surface(xs, ys, zs, alpha=0.08, edgecolor='w')
    t = np.linspace(0, 2*np.pi, 300)
    ax_bloch.plot(np.cos(t), np.sin(t), 0*t, linewidth=0.8)  # equator
    ax_bloch.plot([0,1],[0,0],[0,0], linewidth=0.8)
    ax_bloch.plot([0,0],[0,1],[0,0], linewidth=0.8)
    ax_bloch.plot([0,0],[0,0],[0,1], linewidth=0.8)
    ax_bloch.scatter([0],[0],[1], s=30); ax_bloch.text(0,0,1.05,"|0⟩", ha='center', va='bottom')
    ax_bloch.scatter([0],[0],[-1], s=30); ax_bloch.text(0,0,-1.05,"|1⟩", ha='center', va='top')
    ax_bloch.set_xlim([-1,1]); ax_bloch.set_ylim([-1,1]); ax_bloch.set_zlim([-1,1])
    ax_bloch.set_xlabel('X'); ax_bloch.set_ylabel('Y'); ax_bloch.set_zlabel('Z')
    ax_bloch.set_title('Bloch Sphere')
    line, = ax_bloch.plot([0,0],[0,0],[0,0], linewidth=3)
    return line

bloch_line = init_bloch()

# -----------------------------
# Probability bar (wide & safe margins)
# -----------------------------
bars = ax_probs.bar(['|0⟩', '|1⟩'], [1.0, 0.0])
ax_probs.set_ylim(0, 1)
ax_probs.set_ylabel('Probability')
ax_probs.margins(x=0.15)            # 좌우 여백
ax_probs.set_title('Measurement Probabilities (Z basis)')

# -----------------------------
# Buttons & sliders
# -----------------------------
btn_H = Button(ax_btn_H, 'H');   btn_X = Button(ax_btn_X, 'X')
btn_Y = Button(ax_btn_Y, 'Y');   btn_Z = Button(ax_btn_Z, 'Z')
btn_S = Button(ax_btn_S, 'S');   btn_T = Button(ax_btn_T, 'T')
btn_U = Button(ax_btn_U, 'Undo'); btn_R = Button(ax_btn_R, 'Reset')

btn_H.on_clicked(lambda e: apply_unitary(H))
btn_X.on_clicked(lambda e: apply_unitary(X))
btn_Y.on_clicked(lambda e: apply_unitary(Y))
btn_Z.on_clicked(lambda e: apply_unitary(Z))
btn_S.on_clicked(lambda e: apply_unitary(S))
btn_T.on_clicked(lambda e: apply_unitary(T))
btn_U.on_clicked(lambda e: undo())
btn_R.on_clicked(lambda e: reset())

sx = Slider(ax_sx, 'θx (deg)', -360, 360, valinit=0.0)
sy = Slider(ax_sy, 'θy (deg)', -360, 360, valinit=0.0)
sz = Slider(ax_sz, 'θz (deg)', -360, 360, valinit=0.0)

btn_Rx = Button(ax_btn_Rx, 'Apply Rx')
btn_Ry = Button(ax_btn_Ry, 'Apply Ry')
btn_Rz = Button(ax_btn_Rz, 'Apply Rz')

btn_Rx.on_clicked(lambda e: apply_unitary(Rx(np.deg2rad(sx.val))))
btn_Ry.on_clicked(lambda e: apply_unitary(Ry(np.deg2rad(sy.val))))
btn_Rz.on_clicked(lambda e: apply_unitary(Rz(np.deg2rad(sz.val))))

# -----------------------------
# Visual updater
# -----------------------------
def update_visuals():
    x, y, z = bloch_coordinates(state)
    bloch_line.set_data_3d([0, x], [0, y], [0, z])
    p0, p1 = probabilities(state)
    bars[0].set_height(p0); bars[1].set_height(p1)
    ax_probs.set_title(f'P(|0⟩)={p0:.3f}, P(|1⟩)={p1:.3f}')
    fig.canvas.draw_idle()

update_visuals()

# -----------------------------
# Keyboard shortcuts
# -----------------------------
def on_key(event):
    key = (event.key or '').lower()
    if   key == 'h': apply_unitary(H)
    elif key == 'x': apply_unitary(X)
    elif key == 'y': apply_unitary(Y)
    elif key == 'z': apply_unitary(Z)
    elif key == 's': apply_unitary(S)
    elif key == 't': apply_unitary(T)
    elif key == 'r': reset()
    elif key == 'u': undo()

fig.canvas.mpl_connect('key_press_event', on_key)

# -----------------------------
# Show
# -----------------------------
st.pyplot(fig)
