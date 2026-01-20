#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bloch Sphere Visualizer (Matplotlib + Widgets + Auto Rotation)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

PI = np.pi

def state_from_angles(theta: float, phi: float):
    a = np.cos(theta / 2.0)
    b = np.exp(1j * phi) * np.sin(theta / 2.0)
    return np.array([a, b], dtype=complex)

def bloch_vector_from_state(psi: np.ndarray):
    a, b = psi
    sx = 2 * (a.conjugate() * b).real
    sy = 2 * (a.conjugate() * b).imag
    sz = (np.abs(a) ** 2 - np.abs(b) ** 2).real
    return float(sx), float(sy), float(sz)

def draw_bloch_sphere(ax):
    u = np.linspace(0, 2 * PI, 60)
    v = np.linspace(0, PI, 30)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0.3,
                    edgecolor='lightgray', alpha=0.08, shade=False)
    ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, linewidth=0.3,
                      color='silver', alpha=0.6)
    ax.plot([-1.1, 1.1], [0, 0], [0, 0], lw=0.8, color='black')
    ax.plot([0, 0], [-1.1, 1.1], [0, 0], lw=0.8, color='black')
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], lw=0.8, color='black')
    ax.text(1.15, 0, 0, 'x', fontsize=11)
    ax.text(0, 1.15, 0, 'y', fontsize=11)
    ax.text(0, 0, 1.15, 'z', fontsize=11)
    ax.set_box_aspect([1, 1, 1])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    markers = {
        r'|0⟩': (0, 0, 1),
        r'|1⟩': (0, 0, -1),
        r'|+x⟩': (1, 0, 0),
        r'|-x⟩': (-1, 0, 0),
        r'|+y⟩': (0, 1, 0),
        r'|-y⟩': (0, -1, 0),
    }
    for label, (mx, my, mz) in markers.items():
        ax.scatter([mx], [my], [mz], s=30, depthshade=False, color='black')
        ax.text(mx * 1.06, my * 1.06, mz * 1.06, label, fontsize=10,
                ha='center', va='center')

class BlochApp:
    def __init__(self):
        self.fig = plt.figure('Bloch Sphere — Quantum Viz', figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.35)

        draw_bloch_sphere(self.ax)

        self.state_line, = self.ax.plot([0, 0], [0, 0], [0, 1], lw=3.0, color='C3')
        self.state_dot = self.ax.scatter([0], [0], [1], s=45, color='C3', depthshade=False)

        self.info_ax = self.fig.add_axes([0.05, 0.15, 0.42, 0.18])
        self.info_ax.axis('off')
        self.info_text = self.info_ax.text(0.01, 0.98, '', va='top', ha='left', fontsize=10, family='monospace')

        self.ax_preset = self.fig.add_axes([0.05, 0.02, 0.42, 0.11])
        self.presets = RadioButtons(
            self.ax_preset,
            (r'|0⟩', r'|1⟩', r'|+x⟩', r'|-x⟩', r'|+y⟩', r'|-y⟩', 'Custom'),
            active=0
        )

        self.ax_theta = self.fig.add_axes([0.55, 0.12, 0.38, 0.03])
        self.ax_phi   = self.fig.add_axes([0.55, 0.07, 0.38, 0.03])
        self.s_theta = Slider(self.ax_theta, 'θ (deg)', 0.0, 180.0, valinit=0.0)
        self.s_phi   = Slider(self.ax_phi, 'φ (deg)', 0.0, 360.0, valinit=0.0)

        self.ax_btn_anim = self.fig.add_axes([0.74, 0.02, 0.09, 0.035])
        self.btn_anim = Button(self.ax_btn_anim, 'Anim')
        self.ax_btn_reset = self.fig.add_axes([0.85, 0.02, 0.09, 0.035])
        self.btn_reset = Button(self.ax_btn_reset, 'Reset')

        self.theta = 0.0
        self.phi = 0.0
        self.anim_on = False
        self.ani = None
        self.omega = 1.2
        self.view_angle = 0  # ✅ 자동 회전 각도

        self.s_theta.on_changed(self._on_slider)
        self.s_phi.on_changed(self._on_slider)
        self.presets.on_clicked(self._on_preset)
        self.btn_anim.on_clicked(self._toggle_anim)
        self.btn_reset.on_clicked(self._reset)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._update_plot()

    def _on_slider(self, _):
        if self.presets.value_selected != 'Custom':
            self.presets.set_active(6)
        self.theta = np.deg2rad(self.s_theta.val)
        self.phi = np.deg2rad(self.s_phi.val)
        self._update_plot()

    def _on_preset(self, label):
        preset_map = {
            r'|0⟩': (0.0, 0.0),
            r'|1⟩': (PI, 0.0),
            r'|+x⟩': (PI/2, 0.0),
            r'|-x⟩': (PI/2, PI),
            r'|+y⟩': (PI/2, PI/2),
            r'|-y⟩': (PI/2, 3*PI/2),
        }
        if label in preset_map:
            th, ph = preset_map[label]
            self.theta, self.phi = th, ph
            self.s_theta.set_val(np.rad2deg(th))
            self.s_phi.set_val(np.rad2deg(ph))
        self._update_plot()

    def _toggle_anim(self, _):
        if not self.anim_on:
            self.anim_on = True
            self.btn_anim.label.set_text('Pause')
            self.ani = FuncAnimation(self.fig, self._anim_step, interval=30)        
        else:
            self.anim_on = False
            self.btn_anim.label.set_text('Anim')
            if self.ani is not None:
                self.ani.event_source.stop()
                self.ani = None

    def _reset(self, _):
        self.anim_on = False
        if self.ani is not None:
            self.ani.event_source.stop()
            self.ani = None
        self.btn_anim.label.set_text('Anim')
        self.theta, self.phi = 0.0, 0.0
        self.s_theta.set_val(0.0)
        self.s_phi.set_val(0.0)
        self.presets.set_active(0)
        self.view_angle = 0
        self._update_plot()

    def _on_key(self, event):
        if event.key == 'a':
            self._toggle_anim(None)
        elif event.key == 'left':
            self.s_phi.set_val((self.s_phi.val - 5) % 360)
        elif event.key == 'right':
            self.s_phi.set_val((self.s_phi.val + 5) % 360)
        elif event.key == 'up':
            self.s_theta.set_val(min(180, self.s_theta.val + 5))
        elif event.key == 'down':
            self.s_theta.set_val(max(0, self.s_theta.val - 5))

    def _anim_step(self, _frame):
        if not self.anim_on:
            return
        dphi = self.omega * (30 / 1000.0)
        self.phi = (self.phi + dphi) % (2 * PI)
        self.s_phi.set_val(np.rad2deg(self.phi))

        # ✅ 자동 회전 추가
        self.view_angle = (self.view_angle + 0.5) % 360
        self.ax.view_init(elev=20, azim=self.view_angle)

        self._update_plot()

    def _update_plot(self):
        psi = state_from_angles(self.theta, self.phi)
        rx, ry, rz = bloch_vector_from_state(psi)
        self.state_line.set_data_3d([0, rx], [0, ry], [0, rz])
        self.state_dot._offsets3d = ([rx], [ry], [rz])
        a, b = psi
        text = (
            r"|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩" + "\n" +
            f"θ = {np.degrees(self.theta):6.2f}°   φ = {np.degrees(self.phi):6.2f}°\n" +
            f"a = {a.real:+.4f}{a.imag:+.4f}i\n" +
            f"b = {b.real:+.4f}{b.imag:+.4f}i\n" +
            f"r = (⟨σx⟩,⟨σy⟩,⟨σz⟩) = ({rx:+.4f}, {ry:+.4f}, {rz:+.4f})"
        )
        self.info_text.set_text(text)
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    app = BlochApp()
    plt.show()
