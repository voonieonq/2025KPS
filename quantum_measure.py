import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D)

# --- |0>, |1> 정의 ---
ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

# --- 초기 중첩 상태: (|0⟩ + i|1⟩) / √2 ---
psi_initial = (ket0 + 1j * ket1) / np.sqrt(2)

# 현재 상태 / 최근 측정 결과
psi_current = psi_initial.copy()
psi_measured = None

# --- 블로흐 좌표 ---
def bloch_coordinates(psi):
    a, b = psi[0, 0], psi[1, 0]
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = np.abs(a)**2 - np.abs(b)**2
    return float(x), float(y), float(z)

# --- Z-basis 측정 ---
def measure_z(psi):
    prob_0 = float(np.abs(np.vdot(ket0, psi))**2)
    outcome = np.random.choice([0, 1], p=[prob_0, 1 - prob_0])
    return ket0 if outcome == 0 else ket1

# --- 블로흐 구 그리기 ---
def draw_bloch_sphere(ax, state, color='blue', label=''):
    ax.cla()

    # 블로흐 구
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.1, edgecolor='gray', linewidth=0.2)

    # 좌표축
    ax.quiver(0, 0, 0, 1, 0, 0, color='gray', linewidth=1)
    ax.quiver(0, 0, 0, 0, 1, 0, color='gray', linewidth=1)
    ax.quiver(0, 0, 0, 0, 0, 1, color='gray', linewidth=1)
    ax.quiver(0, 0, 0, 0, 0, -1, color='gray', linewidth=1)
    ax.text(1.1, 0, 0, 'X')
    ax.text(0, 1.1, 0, 'Y')
    ax.text(0, 0, 1.1, '|0⟩')
    ax.text(0, 0, -1.2, '|1⟩')

    # 상태 벡터
    bx, by, bz = bloch_coordinates(state)
    ax.quiver(0, 0, 0, bx, by, bz, color=color, linewidth=2)
    ax.text(bx, by, bz, label, color=color)

    # 보기/축설정
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.set_zlim([-1.4, 1.4])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Visualization of Quantum State on Bloch Sphere")
    ax.view_init(elev=25, azim=45)

# --- 버튼 콜백 ---
def show_initial(event):
    global psi_current
    psi_current = psi_initial.copy()
    draw_bloch_sphere(ax, psi_current, color='blue', label='Initial State')
    fig.canvas.draw_idle()

def show_measurement(event):
<<<<<<< HEAD
    global psi_measured
    psi_measured = measure_z(psi_initial)
    draw_bloch_sphere(ax, psi_measured, color='red', label='Measured State')
    fig.canvas.draw()
=======
    global psi_current, psi_measured
    psi_measured = measure_z(psi_current)
    psi_current = psi_measured  # 측정 후 상태를 현재 상태로 업데이트
    draw_bloch_sphere(ax, psi_current, color='red', label='Measured State')
    fig.canvas.draw_idle()
>>>>>>> 4fd94af7f358cd260df6ced6e56d26c7177b687d

# --- 플롯/버튼 UI ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
draw_bloch_sphere(ax, psi_initial, color='blue', label='Initial State')

ax_button1 = plt.axes([0.25, 0.05, 0.2, 0.075])
btn1 = Button(ax_button1, 'show_initial')
btn1.on_clicked(show_initial)

ax_button2 = plt.axes([0.55, 0.05, 0.2, 0.075])
btn2 = Button(ax_button2, 'show_measurement')
btn2.on_clicked(show_measurement)

<<<<<<< HEAD
import streamlit as st
st.pyplot(fig)

=======
plt.show()
>>>>>>> 4fd94af7f358cd260df6ced6e56d26c7177b687d
