# quantum_circuit_gui.py — Quantum Circuit GUI with Visual Diagram (no Streamlit)
# Requirements: numpy, matplotlib
# Run: python quantum_circuit_gui.py
#
# Features
# - 1~2 qubits (toggle). Gates: H, X, Y, Z, S, T, Rx/Ry/Rz + CNOT.
# - Matplotlib GUI (Buttons/Radio/Sliders). No terminal input.
# - **Proper circuit diagram** (wires, boxes, CNOT dot+circle) in its own window.
# - Live probability bar chart + state vector text.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple

# ----------------------------
# Linear algebra: bases & gates
# ----------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)


def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)

# ----------------------------
# Circuit machinery
# ----------------------------
GateToken = Tuple[str, tuple]  # (name, params)


def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    M = ops[0]
    for A in ops[1:]:
        M = np.kron(M, A)
    return M


def embed_1q(U: np.ndarray, target: int, nq: int) -> np.ndarray:
    ops = []
    for q in range(nq):
        ops.append(U if q == target else I2)
    return kron_n(ops)


def CNOT_matrix(ctrl: int, tgt: int, nq: int) -> np.ndarray:
    assert ctrl != tgt
    if nq == 1:
        raise ValueError("CNOT needs at least 2 qubits")
    ops0 = [I2]*nq
    ops1 = [I2]*nq
    ops0[ctrl] = P0
    ops1[ctrl] = P1
    return kron_n(ops0) + kron_n(ops1) @ embed_1q(X, tgt, nq)


def apply_circuit(history: List[GateToken], nq: int) -> np.ndarray:
    psi = np.zeros(2**nq, dtype=complex)
    psi[0] = 1.0
    for name, params in history:
        if name in {"H","X","Y","Z","S","T"}:
            (q,) = params
            U = {"H":H, "X":X, "Y":Y, "Z":Z, "S":S, "T":T}[name]
            psi = embed_1q(U, q, nq) @ psi
        elif name in {"Rx","Ry","Rz"}:
            q, theta = params
            U = {"Rx":Rx, "Ry":Ry, "Rz":Rz}[name](theta)
            psi = embed_1q(U, q, nq) @ psi
        elif name == "CNOT":
            ctrl, tgt = params
            psi = CNOT_matrix(ctrl, tgt, nq) @ psi
        else:
            raise ValueError(f"Unknown gate {name}")
    nrm = np.linalg.norm(psi)
    if nrm > 0:
        psi = psi / nrm
    return psi

# ----------------------------
# Readouts & helpers
# ----------------------------
BOX = {"H":"H","X":"X","Y":"Y","Z":"Z","S":"S","T":"T","Rx":"Rx","Ry":"Ry","Rz":"Rz"}

def basis_labels(nq: int) -> List[str]:
    return [format(i, f"0{nq}b") for i in range(2**nq)]


def probs_from_state(psi: np.ndarray) -> np.ndarray:
    return np.real(psi*np.conj(psi))


def pretty_state(psi: np.ndarray) -> str:
    labels = basis_labels(int(np.log2(len(psi))))
    return "\n".join(f"|{l}>: {psi[i]: .4f}" for i, l in enumerate(labels))

# ----------------------------
# Diagram rendering (matplotlib)
# ----------------------------

def render_circuit(ax, history: List[GateToken], nq: int):
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0, max(10, 2 + len(history)))
    ax.set_ylim(-1, nq)

    # Wires
    for q in range(nq):
        ax.plot([0, 2 + len(history)], [q, q], lw=1.5)
        ax.text(-0.3, q, f"q{q}", va='center', ha='right')

    # Column spacing
    x0 = 1.0
    dx = 1.0
    box_w, box_h = 0.7, 0.45

    for i, (name, params) in enumerate(history):
        x = x0 + i*dx
        if name == 'CNOT':
            ctrl, tgt = params
            y1, y2 = ctrl, tgt
            # vertical line
            ax.plot([x, x], [min(y1,y2), max(y1,y2)], lw=1.2)
            # control dot
            ax.add_patch(Circle((x, y1), 0.08, color='k'))
            # target: plus in circle
            circ = Circle((x, y2), 0.18, fill=False, lw=1.2)
            ax.add_patch(circ)
            ax.plot([x-0.12, x+0.12], [y2, y2], lw=1.2)
            ax.plot([x, x], [y2-0.12, y2+0.12], lw=1.2)
        else:
            # 1-qubit gate box
            q = params[0]
            rect = Rectangle((x - box_w/2, q - box_h/2), box_w, box_h, fill=False, lw=1.5)
            ax.add_patch(rect)
            ax.text(x, q, BOX[name], ha='center', va='center', fontsize=10)

    ax.set_title('Quantum Circuit', pad=8)

# ----------------------------
# GUI App (matplotlib)
# ----------------------------
class App:
    def __init__(self):
        self.nq = 2
        self.history: List[GateToken] = []
        self.single_qubit_target = 0
        self.ctrl = 0
        self.tgt = 1

        # Figure layout
        self.fig = plt.figure(figsize=(12, 7))
        try:
            self.fig.canvas.manager.set_window_title('Quantum Circuit GUI')
        except Exception:
            pass

        # Axes: circuit diagram (top), state & probs (bottom)
        self.ax_circuit = self.fig.add_axes([0.05, 0.60, 0.62, 0.35])
        self.ax_state   = self.fig.add_axes([0.05, 0.28, 0.30, 0.25])
        self.ax_probs   = self.fig.add_axes([0.38, 0.28, 0.29, 0.25])

        # Controls area (right)
        self.ax_qubits = self.fig.add_axes([0.72, 0.82, 0.10, 0.12])
        self.ax_target = self.fig.add_axes([0.84, 0.82, 0.12, 0.12])
        self.ax_ctrl   = self.fig.add_axes([0.72, 0.68, 0.10, 0.12])
        self.ax_tgt    = self.fig.add_axes([0.84, 0.68, 0.12, 0.12])

        self.ax_h  = self.fig.add_axes([0.72, 0.55, 0.07, 0.06])
        self.ax_x  = self.fig.add_axes([0.80, 0.55, 0.07, 0.06])
        self.ax_y  = self.fig.add_axes([0.88, 0.55, 0.07, 0.06])
        self.ax_z  = self.fig.add_axes([0.96, 0.55, 0.03, 0.06])
        self.ax_s  = self.fig.add_axes([0.72, 0.48, 0.07, 0.06])
        self.ax_t  = self.fig.add_axes([0.80, 0.48, 0.07, 0.06])

        self.ax_rx = self.fig.add_axes([0.72, 0.39, 0.23, 0.03])
        self.ax_ry = self.fig.add_axes([0.72, 0.34, 0.23, 0.03])
        self.ax_rz = self.fig.add_axes([0.72, 0.29, 0.23, 0.03])
        self.ax_apply_rx = self.fig.add_axes([0.96, 0.39, 0.03, 0.03])
        self.ax_apply_ry = self.fig.add_axes([0.96, 0.34, 0.03, 0.03])
        self.ax_apply_rz = self.fig.add_axes([0.96, 0.29, 0.03, 0.03])

        self.ax_cnot = self.fig.add_axes([0.72, 0.21, 0.27, 0.05])
        self.ax_undo = self.fig.add_axes([0.72, 0.13, 0.13, 0.06])
        self.ax_reset= self.fig.add_axes([0.86, 0.13, 0.13, 0.06])

        # Widgets
        self.rb_qubits = RadioButtons(self.ax_qubits, ("2 qubits", "1 qubit"))
        self.rb_qubits.on_clicked(self.on_toggle_qubits)
        self.rb_target = RadioButtons(self.ax_target, ("target q0", "target q1"))
        self.rb_target.on_clicked(self.on_target_change)
        self.rb_ctrl   = RadioButtons(self.ax_ctrl,   ("ctrl q0", "ctrl q1"))
        self.rb_ctrl.on_clicked(self.on_ctrl_change)
        self.rb_tgt    = RadioButtons(self.ax_tgt,    ("tgt q0", "tgt q1"))
        self.rb_tgt.on_clicked(self.on_tgt_change)

        self.btn_h = Button(self.ax_h, 'H'); self.btn_h.on_clicked(lambda evt: self.add_1q('H'))
        self.btn_x = Button(self.ax_x, 'X'); self.btn_x.on_clicked(lambda evt: self.add_1q('X'))
        self.btn_y = Button(self.ax_y, 'Y'); self.btn_y.on_clicked(lambda evt: self.add_1q('Y'))
        self.btn_z = Button(self.ax_z, 'Z'); self.btn_z.on_clicked(lambda evt: self.add_1q('Z'))
        self.btn_s = Button(self.ax_s, 'S'); self.btn_s.on_clicked(lambda evt: self.add_1q('S'))
        self.btn_t = Button(self.ax_t, 'T'); self.btn_t.on_clicked(lambda evt: self.add_1q('T'))

        self.sl_rx = Slider(self.ax_rx, 'Rx (deg)', -360, 360, valinit=0, valstep=15)
        self.sl_ry = Slider(self.ax_ry, 'Ry (deg)', -360, 360, valinit=0, valstep=15)
        self.sl_rz = Slider(self.ax_rz, 'Rz (deg)', -360, 360, valinit=0, valstep=15)
        self.btn_apply_rx = Button(self.ax_apply_rx, '→'); self.btn_apply_rx.on_clicked(lambda evt: self.add_rot('Rx', np.deg2rad(self.sl_rx.val)))
        self.btn_apply_ry = Button(self.ax_apply_ry, '→'); self.btn_apply_ry.on_clicked(lambda evt: self.add_rot('Ry', np.deg2rad(self.sl_ry.val)))
        self.btn_apply_rz = Button(self.ax_apply_rz, '→'); self.btn_apply_rz.on_clicked(lambda evt: self.add_rot('Rz', np.deg2rad(self.sl_rz.val)))

        self.btn_cnot = Button(self.ax_cnot, 'Apply CNOT (ctrl→tgt)'); self.btn_cnot.on_clicked(self.add_cnot)
        self.btn_undo = Button(self.ax_undo, 'Undo'); self.btn_undo.on_clicked(self.undo)
        self.btn_reset= Button(self.ax_reset,'Reset'); self.btn_reset.on_clicked(self.reset)

        self.update_views()

    # ---- callbacks ----
    def on_toggle_qubits(self, label):
        self.nq = 2 if label.startswith('2') else 1
        self.history.clear()
        self.single_qubit_target = min(self.single_qubit_target, self.nq-1)
        self.ctrl = 0
        self.tgt  = 1 if self.nq == 2 else 0
        self.update_views()

    def on_target_change(self, label):
        self.single_qubit_target = 0 if 'q0' in label else 1

    def on_ctrl_change(self, label):
        self.ctrl = 0 if 'q0' in label else 1

    def on_tgt_change(self, label):
        self.tgt = 0 if 'q0' in label else 1

    def add_1q(self, name: str):
        q = self.single_qubit_target
        if q >= self.nq:
            return
        self.history.append((name, (q,)))
        self.update_views()

    def add_rot(self, name: str, theta: float):
        q = self.single_qubit_target
        if q >= self.nq:
            return
        self.history.append((name, (q, theta)))
        self.update_views()

    def add_cnot(self, evt):
        if self.nq < 2 or self.ctrl == self.tgt:
            return
        self.history.append(("CNOT", (self.ctrl, self.tgt)))
        self.update_views()

    def undo(self, evt):
        if self.history:
            self.history.pop()
        self.update_views()

    def reset(self, evt):
        self.history.clear()
        self.update_views()

    # ---- rendering ----
    def update_views(self):
        # Circuit diagram
        render_circuit(self.ax_circuit, self.history, self.nq)

        # State + probabilities
        psi = apply_circuit(self.history, self.nq)
        labels = basis_labels(self.nq)
        probs = probs_from_state(psi)

        # ---- state text ----
        self.ax_state.clear(); self.ax_state.set_axis_off()
        self.ax_state.text(
            0.01, 0.98,
            'State vector',
            fontsize=11, fontweight='bold', va='top'
        )
        self.ax_state.text(
            0.01, 0.90,
            pretty_state(psi),
            family='monospace',
            va='top',
            multialignment='left',
            linespacing=1.4,   # 줄 간격 늘리기
            fontsize=9
        )

        # ---- prob bar chart ----
        self.ax_probs.clear()
        self.ax_probs.set_title('Measurement probabilities (Z-basis)')
        self.ax_probs.bar(range(len(probs)), probs)
        self.ax_probs.set_xticks(range(len(probs)))
        self.ax_probs.set_xticklabels([f"|{l}>" for l in labels], rotation=30, ha='right')
        self.ax_probs.set_ylim(0, 1)
        self.ax_probs.grid(True, axis='y', alpha=0.3)

        # 자동 레이아웃 조정
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    print("Tip: Create a Bell state with H on q0 then CNOT (ctrl q0 → tgt q1).")
    app = App()

def render(**params):
    # 1) 새 그림을 만들거나
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    # ax.plot(...), ax.imshow(...), 등등
    return fig
