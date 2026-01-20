#quantum_state

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D backend 등록용)

# ---------------------------
# 양자 상태 및 게이트 정의
# ---------------------------
state = np.array([1+0j, 0+0j])  # 초기 |0>
STEP = np.pi / 8                 # 회전 단위각

# Pauli 행렬
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

# Hadamard
H = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1, -1]], dtype=complex)

def normalize(s):
    n = np.linalg.norm(s)
    return s if n == 0 else s / n

def apply_gate(U):
    global state
    state = normalize(U @ state)

def Rx(theta):
    # exp(-i theta/2 σx) = cos(theta/2) I - i sin(theta/2) σx
    return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*X

def Ry(theta):
    return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*Y

def Rz(theta):
    return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*Z

def probs(s):
    return (abs(s[0])**2, abs(s[1])**2)

def rel_phase(s):
    # 상대위상 φ = arg(b) - arg(a), [-π, π]로 반환
    a, b = s[0], s[1]
    return float(np.angle(b) - np.angle(a))

def bloch_coords(s):
    """
    |ψ> = a|0> + b|1>
    Bloch vector: (x, y, z) = (2 Re(a b*), 2 Im(a b*), |a|^2 - |b|^2)
    또한 ⟨σx⟩=x, ⟨σy⟩=y, ⟨σz⟩=z
    """
    a, b = s[0], s[1]
    x = 2 * np.real(a * np.conj(b))
    y = 2 * np.imag(a * np.conj(b))
    z = np.abs(a)**2 - np.abs(b)**2
    return float(x), float(y), float(z)

# ---------------------------
# 그림 설정 (3D Bloch sphere)
# ---------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.30)

# 블로흐 구(단위구) 와이어프레임
u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 40)
Xsp = np.outer(np.cos(u), np.sin(v))
Ysp = np.outer(np.sin(u), np.sin(v))
Zsp = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(Xsp, Ysp, Zsp, rstride=2, cstride=2, linewidth=0.4, alpha=0.35)

# 좌표축
ax.plot([-1, 1], [0, 0], [0, 0], lw=1)  # x
ax.plot([0, 0], [-1, 1], [0, 0], lw=1)  # y
ax.plot([0, 0], [0, 0], [-1, 1], lw=1)  # z
ax.text(1.08, 0, 0, 'X', fontsize=10)
ax.text(0, 1.08, 0, 'Y', fontsize=10)
ax.text(0, 0, 1.08, 'Z', fontsize=10)

# 상태점과 보조선
point, = ax.plot([], [], [], 'o', markersize=10)
vline, = ax.plot([], [], [], '-', lw=1, alpha=0.6)

# 텍스트(상태 정보)
info = ax.text2D(0.03, 0.95, "", transform=ax.transAxes, fontsize=10, family="monospace")

# 시야 설정
ax.set_box_aspect((1, 1, 1))
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_title("Quantum State on the Bloch Sphere (Unitary Evolution)")

# ---------------------------
# 애니메이션 업데이트
# ---------------------------
def init():
    point.set_data([], [])
    point.set_3d_properties([])
    vline.set_data([], [])
    vline.set_3d_properties([])
    info.set_text("")
    return point, vline, info

def update(_frame):
    x, y, z = bloch_coords(state)

    # 상태 점
    point.set_data([x], [y])
    point.set_3d_properties([z])

    # 원점에서 상태점까지 보조선
    vline.set_data([0, x], [0, y])
    vline.set_3d_properties([0, z])

    p0, p1 = probs(state)
    phi = rel_phase(state)

    a, b = state[0], state[1]
    # 정보 패널: 확률/상대위상/블로흐/진폭(크기, 위상)
    info.set_text(
        "State |ψ⟩ = a|0⟩ + b|1⟩\n"
        f"|a|^2={p0:.3f}, |b|^2={p1:.3f},  φ={phi:+.3f} rad\n"
        f"Bloch: x={x:+.3f}, y={y:+.3f}, z={z:+.3f}  (⟨σx⟩,⟨σy⟩,⟨σz⟩)\n"
        f"a={np.abs(a):.3f}·e^{1j*np.angle(a):+.3f}i, "
        f"b={np.abs(b):.3f}·e^{1j*np.angle(b):+.3f}i"
    )
    return point, vline, info

ani = animation.FuncAnimation(
    fig, update,
    frames=range(10**9),
    init_func=init,
    interval=120,
    blit=False,
    cache_frame_data=False
)

# ---------------------------
# 버튼 UI
# ---------------------------
# 1행: H, X, Y, Z
ax_h  = plt.axes([0.06, 0.19, 0.16, 0.07])
ax_x  = plt.axes([0.28, 0.19, 0.16, 0.07])
ax_y  = plt.axes([0.50, 0.19, 0.16, 0.07])
ax_z  = plt.axes([0.72, 0.19, 0.16, 0.07])

# 2행: Rx±, Ry±, Rz±
ax_rxp = plt.axes([0.06, 0.10, 0.16, 0.07])
ax_rxm = plt.axes([0.28, 0.10, 0.16, 0.07])
ax_ryp = plt.axes([0.50, 0.10, 0.16, 0.07])
ax_rym = plt.axes([0.72, 0.10, 0.16, 0.07])

ax_rzp = plt.axes([0.06, 0.02, 0.16, 0.07])
ax_rzm = plt.axes([0.28, 0.02, 0.16, 0.07])
ax_rst = plt.axes([0.50, 0.02, 0.16, 0.07])
ax_quit= plt.axes([0.72, 0.02, 0.16, 0.07])

btn_h   = Button(ax_h,  'Hadamard H')
btn_x   = Button(ax_x,  'Pauli-X')
btn_y   = Button(ax_y,  'Pauli-Y')
btn_z   = Button(ax_z,  'Pauli-Z')

btn_rxp = Button(ax_rxp, 'Rx  +π/8')
btn_rxm = Button(ax_rxm, 'Rx  -π/8')
btn_ryp = Button(ax_ryp, 'Ry  +π/8')
btn_rym = Button(ax_rym, 'Ry  -π/8')
btn_rzp = Button(ax_rzp, 'Rz  +π/8')
btn_rzm = Button(ax_rzm, 'Rz  -π/8')

btn_rst = Button(ax_rst, 'Reset |0⟩')
btn_quit= Button(ax_quit,'Quit')

# 콜백
btn_h.on_clicked(lambda e: apply_gate(H))
btn_x.on_clicked(lambda e: apply_gate(X))
btn_y.on_clicked(lambda e: apply_gate(Y))
btn_z.on_clicked(lambda e: apply_gate(Z))

btn_rxp.on_clicked(lambda e: apply_gate(Rx(+STEP)))
btn_rxm.on_clicked(lambda e: apply_gate(Rx(-STEP)))
btn_ryp.on_clicked(lambda e: apply_gate(Ry(+STEP)))
btn_rym.on_clicked(lambda e: apply_gate(Ry(-STEP)))
btn_rzp.on_clicked(lambda e: apply_gate(Rz(+STEP)))
btn_rzm.on_clicked(lambda e: apply_gate(Rz(-STEP)))

def reset():
    global state
    state = np.array([1+0j, 0+0j])
    print("Reset to |0⟩")

btn_rst.on_clicked(lambda e: reset())
btn_quit.on_clicked(lambda e: plt.close(fig))

# ---------------------------
# 키보드 단축키
# ---------------------------
def on_key(event):
    if not event.key:
        return
    k = event.key.lower()
    if k == 'h':   apply_gate(H);           print("H")
    elif k == 'x': apply_gate(X);           print("X")
    elif k == 'y': apply_gate(Y);           print("Y")
    elif k == 'z': apply_gate(Z);           print("Z")
    elif k == '1': apply_gate(Rx(+STEP));   print("Rx +")
    elif k == '2': apply_gate(Rx(-STEP));   print("Rx -")
    elif k == '3': apply_gate(Ry(+STEP));   print("Ry +")
    elif k == '4': apply_gate(Ry(-STEP));   print("Ry -")
    elif k == '5': apply_gate(Rz(+STEP));   print("Rz +")
    elif k == '6': apply_gate(Rz(-STEP));   print("Rz -")
    elif k == 'r': reset()
    elif k == 'q': print("Quit"); plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)

import streamlit as st
st.pyplot(fig)
