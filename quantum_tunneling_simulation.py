import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 사용자 입력
V0 = float(input("Enter barrier height V0 (e.g. 5): ")) #V0= 장벽 높이
a = float(input("Enter barrier width a (e.g. 5.2): ")) #a= 장벽 너비
kx0 = float(input("Enter wave speed (momentum) kx0 (e.g. 5): "))

# 상수
hbar = 1.05
m = 1.0

# 공간 격자
L = 10
N = 100
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

# 시간 설정
dt = 0.005
steps = 300
5
# 포텐셜: 장벽 + 중앙 터널
tunnel_radius = 0.4
V = np.zeros((N, N))
V[(np.abs(X) < a/2) & (np.abs(Y) < a/2)] = V0
V[(X**2 + Y**2) < tunnel_radius**2] = 0.0  # 중앙 터널

# 초기 파동함수
x0, y0 = -5.0, 0.05
ky0 = 0.0
sigma = 1.0
psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * np.exp(1j * (kx0 * X + ky0 * Y))
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dx)

# 라플라시안
def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z
    ) / dx**2

# 시간 진화
def evolve(psi, V):
    lap = laplacian(psi)
    return psi - 1j * dt * (-hbar**2 / (2 * m) * lap + V * psi) / hbar

# 시각화 준비
mid_index = N // 2
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133, projection='3d')

Z = np.abs(psi)**2
line1d, = ax1.plot(x, Z[mid_index], label='|ψ(x, y=0)|²')
ax1.axvspan(-a/2, a/2, color='gray', alpha=0.3, label='Barrier')
ax1.set_ylim(0, 0.12)
ax1.set_xlabel("x")
ax1.set_ylabel("Probability")
ax1.set_title("1D Slice (y=0)")
ax1.legend()

img2d = ax2.imshow(Z, extent=(-L, L, -L, L), origin='lower', cmap='viridis', vmin=0, vmax=0.1)
ax2.set_title("2D |ψ(x, y)|² Top View")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
fig.colorbar(img2d, ax=ax2)

surf = ax3.plot_surface(X, Y, Z, cmap='viridis')
ax3.set_zlim(0, 0.1)
ax3.set_title("3D Surface |ψ(x, y)|²")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("Density")

# 터널링 추적
tunnel_success = []
stop_reached = [False]

def update(frame):
    global psi
    if stop_reached[0]:
        plt.close(fig)
        return []

    psi = evolve(psi, V)
    Z = np.abs(psi)**2

    # 터널링 확률 측정
    success_prob = np.sum(Z[X > a/2]) * dx * dx
    tunnel_success.append(success_prob)

    line1d.set_ydata(Z[mid_index])
    ax1.set_title(f"1D Slice (step {frame})")

    img2d.set_data(Z)
    ax2.set_title(f"2D Top View (step {frame})")

    ax3.clear()
    ax3.plot_surface(X, Y, Z, cmap='viridis')
    ax3.set_zlim(0, 0.1)
    ax3.set_title(f"3D Surface (step {frame})")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("Density")

    # 파동이 x=10에 도달하면 종료
    right_edge_index = np.argmax(x >= 10.0)
    if np.sum(Z[:, right_edge_index]) > 0.001:
        stop_reached[0] = True
        print(f"\n⏹️ Wave reached x=10 at step {frame}")
        print(f"Final tunneling success probability: {success_prob:.3f}")
        if success_prob > 0.05:
            print("Tunneling likely succeeded!")
        else:
            print("Tunneling was unlikely.")
        plt.close(fig)

    return [line1d, img2d]

ani = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
plt.tight_layout()
st.pyplot(fig)

