# two_qubit_sim.py
# 2-Qubit Quantum Gate Simulator (no Streamlit, pure matplotlib)
# - 상태: |ψ> ∈ C^4, 순서 |00>, |01>, |10>, |11>
# - 단일 게이트(X,Y,Z,H,S,T, Rx/Ry/Rz) on Qubit A/B
# - 2큐비트 게이트: CNOT(A→B/B→A), CZ, SWAP
# - 측정: A, B 각각 Z-basis에 개별 측정
# - 시각화: 각 큐비트 부분추적 Bloch 벡터(두 개), 합성계 P(00),P(01),P(10),P(11) 막대
# - 엔탱글먼트 지표: 부분엔트로피 S(ρ_A)=S(ρ_B) (von Neumann), 순수상태면 둘이 같음
# - 회로 히스토리 기록 + Undo
#
# 단위: ħ = 1
# 의존성: numpy, matplotlib

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# ---------- 선형대수 유틸 ----------
def dagger(M): return np.conjugate(M.T)
def kron(A,B): return np.kron(A,B)
def normalize(psi):
    n = np.linalg.norm(psi)
    return psi if n == 0 else psi / n

# ---------- 상태/밀도행렬/부분추적 ----------
def state_to_rho(psi):
    psi = psi.reshape(-1,1)
    return psi @ dagger(psi)

def partial_trace_A(rho):
    # Tr_B(ρ) -> 2x2
    # ρ indices (ab, a'b'): a,b ∈ {0,1}
    rhoA = np.zeros((2,2), dtype=complex)
    for a in [0,1]:
        for ap in [0,1]:
            s = 0+0j
            for b in [0,1]:
                i = 2*a + b
                j = 2*ap + b
                s += rho[i,j]
            rhoA[a,ap] = s
    return rhoA

def partial_trace_B(rho):
    # Tr_A(ρ) -> 2x2
    rhoB = np.zeros((2,2), dtype=complex)
    for b in [0,1]:
        for bp in [0,1]:
            s = 0+0j
            for a in [0,1]:
                i = 2*a + b
                j = 2*a + bp
                s += rho[i,j]
            rhoB[b,bp] = s
    return rhoB

def bloch_from_rho(r):
    # r: 2x2 density → bloch (⟨σx⟩,⟨σy⟩,⟨σz⟩)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    sx = np.real(np.trace(r @ X))
    sy = np.real(np.trace(r @ Y))
    sz = np.real(np.trace(r @ Z))
    return sx, sy, sz

def von_neumann_entropy(r):
    # log base 2
    vals = np.linalg.eigvalsh(r)
    vals = np.clip(vals.real, 0, 1)
    nz = vals[vals>1e-12]
    return float(-np.sum(nz * (np.log(nz)/np.log(2))))

# ---------- 게이트 ----------
I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)
S = np.array([[1,0],[0,1j]], dtype=complex)
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=complex)

def Rx(theta):
    c = np.cos(theta/2)
    s = -1j*np.sin(theta/2)
    return np.array([[c, s],[s, c]], dtype=complex)

def Ry(theta):
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    return np.array([[c, -s],[s,  c]], dtype=complex)

def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp( 1j*theta/2)]], dtype=complex)

def U_single_on_A(U): return kron(U, I)
def U_single_on_B(U): return kron(I, U)

# 2큐비트
CNOT_AtoB = np.array([
    [1,0,0,0],  # |00>→|00>
    [0,1,0,0],  # |01>→|01>
    [0,0,0,1],  # |10>→|11>
    [0,0,1,0],  # |11>→|10>
], dtype=complex)

CNOT_BtoA = np.array([
    [1,0,0,0],  # |00>→|00>
    [0,0,1,0],  # |01>→|10>
    [0,1,0,0],  # |10>→|01>
    [0,0,0,1],  # |11>→|11>
], dtype=complex)

CZ = np.diag([1,1,1,-1]).astype(complex)

SWAP = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1],
], dtype=complex)

def GlobalPhase(phi): return np.exp(1j*phi) * np.eye(4, dtype=complex)

# ---------- 초기 상태 |00> ----------
psi = np.array([1,0,0,0], dtype=complex)

# 회로 히스토리(Undo 지원): (label, U) push
history = []

def apply_U(U, label=None):
    global psi, history
    psi = normalize(U @ psi)
    if label is not None:
        history.append((label, U))

def undo():
    global psi, history
    if not history:
        return
    # 마지막 U를 제거 → 처음부터 재적용
    history.pop()
    psi[:] = np.array([1,0,0,0], dtype=complex)
    for (lbl, U) in history:
        psi[:] = normalize(U @ psi)

# ---------- 측정 ----------
def measure_A():
    global psi, history
    # Z-basis on A: projectors P0 = |0><0| ⊗ I, P1 = |1><1| ⊗ I
    P0 = np.diag([1,1,0,0]).astype(complex)
    P1 = np.diag([0,0,1,1]).astype(complex)
    p0 = float(np.vdot(psi, P0 @ psi).real)
    p1 = float(np.vdot(psi, P1 @ psi).real)
    outcome = np.random.choice([0,1], p=[p0, p1])
    P = P0 if outcome==0 else P1
    post = P @ psi
    psi[:] = normalize(post)
    history.append((f"MeasA={outcome}", P))

def measure_B():
    global psi, history
    # Z-basis on B
    P0 = np.diag([1,0,1,0]).astype(complex)  # B=0 -> |00>,|10>
    P1 = np.diag([0,1,0,1]).astype(complex)  # B=1 -> |01>,|11>
    p0 = float(np.vdot(psi, P0 @ psi).real)
    p1 = float(np.vdot(psi, P1 @ psi).real)
    outcome = np.random.choice([0,1], p=[p0, p1])
    P = P0 if outcome==0 else P1
    post = P @ psi
    psi[:] = normalize(post)
    history.append((f"MeasB={outcome}", P))

def reset_state():
    global psi, history
    psi[:] = np.array([1,0,0,0], dtype=complex)
    history.clear()
    # 슬라이더도 리셋
    s_rxA.reset(); s_ryA.reset(); s_rzA.reset()
    s_rxB.reset(); s_ryB.reset(); s_rzB.reset()
    s_gph.reset()

# ---------- 플롯 레이아웃 ----------
plt.close('all')
fig = plt.figure(figsize=(13, 7))

# 좌: Bloch A, Bloch B
axA = fig.add_axes([0.04, 0.43, 0.28, 0.52], projection='3d')
axB = fig.add_axes([0.36, 0.43, 0.28, 0.52], projection='3d')

# 우상: 합성계 확률 막대
axBar = fig.add_axes([0.70, 0.60, 0.27, 0.32])

# 우중: 엔탱글먼트/상태 표시
txt = fig.text(0.70, 0.48, "", fontsize=11, va='top')

# 우하: 회로 히스토리
axHist = fig.add_axes([0.70, 0.08, 0.27, 0.36])
axHist.axis('off')
hist_text = fig.text(0.70, 0.40, "", fontsize=10, va='top', family='monospace')

# 슬라이더(하단)
ax_rxA = fig.add_axes([0.05, 0.30, 0.26, 0.03])
ax_ryA = fig.add_axes([0.05, 0.26, 0.26, 0.03])
ax_rzA = fig.add_axes([0.05, 0.22, 0.26, 0.03])

ax_rxB = fig.add_axes([0.37, 0.30, 0.26, 0.03])
ax_ryB = fig.add_axes([0.37, 0.26, 0.26, 0.03])
ax_rzB = fig.add_axes([0.37, 0.22, 0.26, 0.03])

ax_gph = fig.add_axes([0.05, 0.16, 0.58, 0.03])

# 버튼(아래)
# 단일 게이트 A
axXA = fig.add_axes([0.05, 0.10, 0.06, 0.05])
axYA = fig.add_axes([0.12, 0.10, 0.06, 0.05])
axZA = fig.add_axes([0.19, 0.10, 0.06, 0.05])
axHA = fig.add_axes([0.26, 0.10, 0.06, 0.05])
axSA = fig.add_axes([0.33, 0.10, 0.06, 0.05])
axTA = fig.add_axes([0.40, 0.10, 0.06, 0.05])

# 단일 게이트 B
axXB = fig.add_axes([0.47, 0.10, 0.06, 0.05])
axYB = fig.add_axes([0.54, 0.10, 0.06, 0.05])
axZB = fig.add_axes([0.61, 0.10, 0.06, 0.05])
axHB = fig.add_axes([0.68, 0.10, 0.06, 0.05])
axSB = fig.add_axes([0.75, 0.10, 0.06, 0.05])
axTB = fig.add_axes([0.82, 0.10, 0.06, 0.05])

# 2큐비트/측정/유틸
axC01 = fig.add_axes([0.05, 0.05, 0.10, 0.04])  # CNOT A→B
axC10 = fig.add_axes([0.16, 0.05, 0.10, 0.04])  # CNOT B→A
axCZ  = fig.add_axes([0.27, 0.05, 0.10, 0.04])
axSW  = fig.add_axes([0.38, 0.05, 0.10, 0.04])

axMA  = fig.add_axes([0.50, 0.05, 0.08, 0.04])
axMB  = fig.add_axes([0.59, 0.05, 0.08, 0.04])
axUD  = fig.add_axes([0.68, 0.05, 0.08, 0.04])
axRS  = fig.add_axes([0.77, 0.05, 0.10, 0.04])

# 위젯 생성
s_rxA = Slider(ax_rxA, label="Rx_A θ", valmin=-2*np.pi, valmax= 2*np.pi, valinit=0.0)
s_ryA = Slider(ax_ryA, label="Ry_A θ", valmin=-2*np.pi, valmax= 2*np.pi, valinit=0.0)
s_rzA = Slider(ax_rzA, label="Rz_A θ", valmin=-2*np.pi, valmax= 2*np.pi, valinit=0.0)

s_rxB = Slider(ax_rxB, label="Rx_B θ", valmin=-2*np.pi, valmax= 2*np.pi, valinit=0.0)
s_ryB = Slider(ax_ryB, label="Ry_B θ", valmin=-2*np.pi, valmax= 2*np.pi, valinit=0.0)
s_rzB = Slider(ax_rzB, label="Rz_B θ", valmin=-2*np.pi, valmax= 2*np.pi, valinit=0.0)

s_gph = Slider(ax_gph, label="Global Phase φ (entanglement엔 영향 X)", valmin=-2*np.pi, valmax=2*np.pi, valinit=0.0)

bXA = Button(axXA, "X_A"); bYA = Button(axYA, "Y_A"); bZA = Button(axZA, "Z_A")
bHA = Button(axHA, "H_A"); bSA = Button(axSA, "S_A"); bTA = Button(axTA, "T_A")

bXB = Button(axXB, "X_B"); bYB = Button(axYB, "Y_B"); bZB = Button(axZB, "Z_B")
bHB = Button(axHB, "H_B"); bSB = Button(axSB, "S_B"); bTB = Button(axTB, "T_B")

bC01 = Button(axC01, "CNOT A→B")
bC10 = Button(axC10, "CNOT B→A")
bCZ  = Button(axCZ,  "CZ")
bSW  = Button(axSW,  "SWAP")

bMA  = Button(axMA,  "Measure A")
bMB  = Button(axMB,  "Measure B")
bUD  = Button(axUD,  "Undo")
bRS  = Button(axRS,  "Reset")

# ---------- 그리기 ----------
def draw_bloch(ax, bloch, title):
    ax.cla()
    # wireframe sphere
    u = np.linspace(0, 2*np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.5, alpha=0.3)
    # axes
    ax.quiver(0,0,0, 1,0,0, length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,1,0, length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,0,1, length=1.0, arrow_length_ratio=0.08)
    ax.text(1.05,0,0,'X', fontsize=9); ax.text(0,1.05,0,'Y', fontsize=9); ax.text(0,0,1.05,'Z', fontsize=9)
    # vector
    sx, sy, sz = bloch
    ax.quiver(0,0,0, sx, sy, sz, length=1.0, arrow_length_ratio=0.1, linewidth=2.0)
    ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1]); ax.set_zlim([-1.1,1.1])
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.set_zticks([-1,0,1])
    ax.set_title(title)

def probs_4(psi):
    # |00>,|01>,|10>,|11|
    p = np.abs(psi)**2
    return p

def update_all():
    rho = state_to_rho(psi)
    rhoA = partial_trace_A(rho)
    rhoB = partial_trace_B(rho)
    blochA = bloch_from_rho(rhoA)
    blochB = bloch_from_rho(rhoB)

    draw_bloch(axA, blochA, "Bloch of Qubit A (Tr_B)")
    draw_bloch(axB, blochB, "Bloch of Qubit B (Tr_A)")

    # joint probabilities
    axBar.cla()
    p = probs_4(psi)
    axBar.bar([0,1,2,3], p, tick_label=["00","01","10","11"])
    axBar.set_ylim(0,1.0)
    axBar.set_title("Joint Probabilities P(ij) in Z-basis")

    # entropy
    SA = von_neumann_entropy(rhoA)
    SB = von_neumann_entropy(rhoB)
    txt.set_text(
        f"⟨History length⟩: {len(history)}"
        f"\nS(ρ_A) = {SA:.4f} bits,  S(ρ_B) = {SB:.4f} bits"
        f"\n|ψ> = {np.array2string(psi, precision=3, suppress_small=True)}"
        "\n※ 순수상태에서 S(ρ_A)=S(ρ_B)이며, 벨 상태에서 ≈1 bit"
    )

    # history print (마지막 12줄)
    shown = history[-12:]
    lines = [f"{i+1}. {lbl}" for i, (lbl, _) in enumerate(shown, start=max(0,len(history)-len(shown))+1)]
    hist_text.set_text("Circuit History (tail):\n" + ("\n".join(lines) if lines else "(empty)"))

    fig.canvas.draw_idle()

# ---------- 콜백 ----------
def do_single_A(U, name): apply_U(U_single_on_A(U), name)
def do_single_B(U, name): apply_U(U_single_on_B(U), name)

def on_XA(e): do_single_A(X, "X_A")
def on_YA(e): do_single_A(Y, "Y_A")
def on_ZA(e): do_single_A(Z, "Z_A")
def on_HA(e): do_single_A(H, "H_A")
def on_SA(e): do_single_A(S, "S_A")
def on_TA(e): do_single_A(T, "T_A")

def on_XB(e): do_single_B(X, "X_B")
def on_YB(e): do_single_B(Y, "Y_B")
def on_ZB(e): do_single_B(Z, "Z_B")
def on_HB(e): do_single_B(H, "H_B")
def on_SB(e): do_single_B(S, "S_B")
def on_TB(e): do_single_B(T, "T_B")

# 회전은 슬라이더의 변화량만큼 누적 적용
last = {"rxA":0.0,"ryA":0.0,"rzA":0.0, "rxB":0.0,"ryB":0.0,"rzB":0.0, "gph":0.0}

def on_rxA(val):
    d = val - last["rxA"]; last["rxA"] = val
    apply_U(U_single_on_A(Rx(d)), f"Rx_A({d:.3f})"); update_all()
def on_ryA(val):
    d = val - last["ryA"]; last["ryA"] = val
    apply_U(U_single_on_A(Ry(d)), f"Ry_A({d:.3f})"); update_all()
def on_rzA(val):
    d = val - last["rzA"]; last["rzA"] = val
    apply_U(U_single_on_A(Rz(d)), f"Rz_A({d:.3f})"); update_all()

def on_rxB(val):
    d = val - last["rxB"]; last["rxB"] = val
    apply_U(U_single_on_B(Rx(d)), f"Rx_B({d:.3f})"); update_all()
def on_ryB(val):
    d = val - last["ryB"]; last["ryB"] = val
    apply_U(U_single_on_B(Ry(d)), f"Ry_B({d:.3f})"); update_all()
def on_rzB(val):
    d = val - last["rzB"]; last["rzB"] = val
    apply_U(U_single_on_B(Rz(d)), f"Rz_B({d:.3f})"); update_all()

def on_gph(val):
    d = val - last["gph"]; last["gph"] = val
    apply_U(GlobalPhase(d), f"GlobalPhase({d:.3f})"); update_all()

def on_C01(e): apply_U(CNOT_AtoB, "CNOT A→B"); update_all()
def on_C10(e): apply_U(CNOT_BtoA, "CNOT B→A"); update_all()
def on_CZ (e): apply_U(CZ, "CZ"); update_all()
def on_SWAP(e): apply_U(SWAP, "SWAP"); update_all()

def on_MA(e): measure_A(); update_all()
def on_MB(e): measure_B(); update_all()
def on_UNDO(e): undo(); update_all()
def on_RESET(e): reset_state(); update_all()

# 이벤트 연결
bXA.on_clicked(on_XA); bYA.on_clicked(on_YA); bZA.on_clicked(on_ZA)
bHA.on_clicked(on_HA); bSA.on_clicked(on_SA); bTA.on_clicked(on_TA)

bXB.on_clicked(on_XB); bYB.on_clicked(on_YB); bZB.on_clicked(on_ZB)
bHB.on_clicked(on_HB); bSB.on_clicked(on_SB); bTB.on_clicked(on_TB)

bC01.on_clicked(on_C01); bC10.on_clicked(on_C10); bCZ.on_clicked(on_CZ); bSW.on_clicked(on_SWAP)

bMA.on_clicked(on_MA); bMB.on_clicked(on_MB); bUD.on_clicked(on_UNDO); bRS.on_clicked(on_RESET)

s_rxA.on_changed(on_rxA); s_ryA.on_changed(on_ryA); s_rzA.on_changed(on_rzA)
s_rxB.on_changed(on_rxB); s_ryB.on_changed(on_ryB); s_rzB.on_changed(on_rzB)
s_gph.on_changed(on_gph)


update_all()                         
st.pyplot(fig)