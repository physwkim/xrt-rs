#!/usr/bin/env python3
"""XRT-RS Beamline Simulator GUI.

qtpy + silx (pygfx/wgpu backend) GUI for interactive beamline simulation.

Features:
  - Source / OE / Screen configuration
  - 2D footprint, X/Z profiles, energy-colored ray scatter
  - 4-panel XRT-style plot (2D + X/Z projections + energy histogram)
  - 3D beamline viewer (xrtGlow-style, pygfx/wgpu)
  - Beamline layout side view with real-time preview
  - Export Python code
  - Save / Load layout (JSON)
"""

import json
import os
import sys
from datetime import date

import numpy as np

import PySide6  # must be imported before rendercanvas.qt

from qtpy.QtCore import Qt, Signal, Slot, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QMessageBox,
    QFileDialog,
    QPlainTextEdit,
    QDialog,
)
from qtpy.QtGui import QFont

from silx.gui.plot import Plot2D, Plot1D
from silx.gui.plot.ImageView import ImageView

import pygfx as gfx
from rendercanvas.qt import QRenderWidget

import xrt_rs as xr

# ─── Constants ──────────────────────────────────────────────────────────────

BACKEND = "pygfx"

OE_TYPES = {
    "FlatMirror": {
        "class": xr.FlatMirror,
        "params": {},
    },
    "ToroidMirror": {
        "class": xr.ToroidMirror,
        "params": {"r_major": 5e6, "r_minor": 50.0},
    },
    "SphericalMirror": {
        "class": xr.SphericalMirror,
        "params": {"radius": 1e6},
    },
    "CylindricalMirror": {
        "class": xr.CylindricalMirror,
        "params": {"radius": 1e6},
    },
    "EllipticalMirror": {
        "class": xr.EllipticalMirror,
        "params": {"p": 10000.0, "q": 5000.0, "theta": 0.003},
    },
    "ParabolicalMirror": {
        "class": xr.ParabolicalMirror,
        "params": {"p": 10000.0, "q": 5000.0, "theta": 0.003},
    },
    "BlazedGrating": {
        "class": xr.BlazedGrating,
        "params": {
            "rho": 600.0,
            "blaze_angle": 0.03,
            "anti_blaze_angle": 0.06,
            "order": 1,
        },
    },
}


# ─── Helpers ────────────────────────────────────────────────────────────────


def sci_spin(value=0.0, minimum=-1e12, maximum=1e12, decimals=6):
    sb = QDoubleSpinBox()
    sb.setRange(minimum, maximum)
    sb.setDecimals(decimals)
    sb.setValue(value)
    sb.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
    sb.setMinimumWidth(120)
    return sb


def energy_to_rgb(e_arr):
    """Map energy values to RGB colors using a viridis-like colormap."""
    if len(e_arr) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    e_min, e_max = e_arr.min(), e_arr.max()
    if e_max - e_min < 1e-10:
        t = np.full_like(e_arr, 0.5)
    else:
        t = (e_arr - e_min) / (e_max - e_min)
    # Simple viridis-like: purple(0) -> teal(0.5) -> yellow(1)
    r = np.clip(1.5 * t - 0.25, 0, 1)
    g = np.clip(np.where(t < 0.5, 2 * t, 1.0 - 0.5 * (t - 0.5)), 0, 1)
    b = np.clip(1.0 - 1.5 * t, 0, 1)
    a = np.ones_like(t)
    return np.column_stack([r, g, b, a]).astype(np.float32)


# ─── Source Config Widget ───────────────────────────────────────────────────


class SourceWidget(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Source", parent)
        form = QFormLayout(self)
        self.nrays = QSpinBox()
        self.nrays.setRange(100, 1_000_000)
        self.nrays.setValue(50000)
        self.nrays.setSingleStep(10000)
        form.addRow("nrays:", self.nrays)
        self.energy = sci_spin(10000.0, 100, 1e6, 1)
        form.addRow("Energy [eV]:", self.energy)
        self.dx = sci_spin(0.1, 0, 100)
        form.addRow("dx [mm]:", self.dx)
        self.dz = sci_spin(0.05, 0, 100)
        form.addRow("dz [mm]:", self.dz)
        self.dxprime = sci_spin(1e-4, 0, 1)
        form.addRow("dx' [rad]:", self.dxprime)
        self.dzprime = sci_spin(5e-5, 0, 1)
        form.addRow("dz' [rad]:", self.dzprime)
        for w in (self.nrays, self.energy, self.dx, self.dz, self.dxprime, self.dzprime):
            w.valueChanged.connect(self.changed)

    def create_source(self, bl):
        return xr.GeometricSource(
            bl, "source", nrays=self.nrays.value(),
            dx=self.dx.value(), dz=self.dz.value(),
            dxprime=self.dxprime.value(), dzprime=self.dzprime.value(),
            energies=[self.energy.value()],
        )

    def to_dict(self):
        return dict(nrays=self.nrays.value(), energy=self.energy.value(),
                    dx=self.dx.value(), dz=self.dz.value(),
                    dxprime=self.dxprime.value(), dzprime=self.dzprime.value())

    def from_dict(self, d):
        self.nrays.setValue(d.get("nrays", 50000))
        self.energy.setValue(d.get("energy", 10000.0))
        self.dx.setValue(d.get("dx", 0.1))
        self.dz.setValue(d.get("dz", 0.05))
        self.dxprime.setValue(d.get("dxprime", 1e-4))
        self.dzprime.setValue(d.get("dzprime", 5e-5))


# ─── Material Config Widget ────────────────────────────────────────────────


class MaterialWidget(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Material", parent)
        form = QFormLayout(self)
        self.elements = QLineEdit("Si")
        form.addRow("Elements:", self.elements)
        self.rho = sci_spin(2.33, 0.001, 30.0, 3)
        form.addRow("Density [g/cm³]:", self.rho)
        self.enabled = QPushButton("Material ON")
        self.enabled.setCheckable(True)
        self.enabled.setChecked(True)
        self.enabled.toggled.connect(lambda c: self.enabled.setText("Material ON" if c else "Material OFF"))
        form.addRow(self.enabled)
        self.elements.textChanged.connect(self.changed)
        self.rho.valueChanged.connect(self.changed)
        self.enabled.toggled.connect(self.changed)

    def create_material(self):
        if not self.enabled.isChecked():
            return None
        elems = [e.strip() for e in self.elements.text().split(",") if e.strip()]
        if not elems:
            return None
        try:
            return xr.Material(elems, rho=self.rho.value())
        except ValueError:
            return None

    def to_dict(self):
        return dict(elements=self.elements.text(), rho=self.rho.value(), enabled=self.enabled.isChecked())

    def from_dict(self, d):
        self.elements.setText(d.get("elements", "Si"))
        self.rho.setValue(d.get("rho", 2.33))
        self.enabled.setChecked(d.get("enabled", True))


# ─── OE Config Widget ──────────────────────────────────────────────────────


class OEWidget(QGroupBox):
    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, name="OE1", oe_type="FlatMirror", parent=None):
        super().__init__(name, parent)
        self._name = name
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(OE_TYPES.keys()))
        self.type_combo.setCurrentText(oe_type)
        top.addWidget(QLabel("Type:"))
        top.addWidget(self.type_combo)
        remove_btn = QPushButton("Remove")
        remove_btn.setFixedWidth(70)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        top.addWidget(remove_btn)
        layout.addLayout(top)
        form = QFormLayout()
        self.center_y = sci_spin(5000.0, -1e6, 1e6, 1)
        form.addRow("Center Y [mm]:", self.center_y)
        self.pitch = sci_spin(0.005, -1, 1)
        form.addRow("Pitch [rad]:", self.pitch)
        layout.addLayout(form)
        self.specific_form = QFormLayout()
        self.specific_widgets = {}
        layout.addLayout(self.specific_form)
        self.material = MaterialWidget()
        layout.addWidget(self.material)
        self.type_combo.currentTextChanged.connect(self._rebuild_specific)
        self._rebuild_specific(oe_type)
        self.type_combo.currentTextChanged.connect(lambda: self.changed.emit())
        self.center_y.valueChanged.connect(self.changed)
        self.pitch.valueChanged.connect(self.changed)
        self.material.changed.connect(self.changed)

    def _rebuild_specific(self, oe_type):
        while self.specific_form.rowCount() > 0:
            self.specific_form.removeRow(0)
        self.specific_widgets.clear()
        for pname, pdefault in OE_TYPES.get(oe_type, {}).get("params", {}).items():
            if isinstance(pdefault, int):
                w = QSpinBox(); w.setRange(-100, 100); w.setValue(pdefault)
            else:
                w = sci_spin(pdefault)
            self.specific_widgets[pname] = w
            self.specific_form.addRow(f"{pname}:", w)
            w.valueChanged.connect(self.changed)

    def create_oe(self, bl):
        oe_type = self.type_combo.currentText()
        cls = OE_TYPES[oe_type]["class"]
        kwargs = {"center": [0, self.center_y.value(), 0], "pitch": self.pitch.value()}
        for pname, widget in self.specific_widgets.items():
            kwargs[pname] = widget.value()
        if oe_type not in ("EllipticalMirror", "ParabolicalMirror"):
            mat = self.material.create_material()
            if mat is not None:
                kwargs["material"] = mat
        return cls(bl, self._name, **kwargs)

    def to_dict(self):
        d = dict(name=self._name, type=self.type_combo.currentText(),
                 center_y=self.center_y.value(), pitch=self.pitch.value(),
                 material=self.material.to_dict(), specific={})
        for pname, widget in self.specific_widgets.items():
            d["specific"][pname] = widget.value()
        return d

    def from_dict(self, d):
        self._name = d.get("name", self._name)
        self.setTitle(self._name)
        self.type_combo.setCurrentText(d.get("type", "FlatMirror"))
        self.center_y.setValue(d.get("center_y", 5000.0))
        self.pitch.setValue(d.get("pitch", 0.005))
        self.material.from_dict(d.get("material", {}))
        for pname, val in d.get("specific", {}).items():
            if pname in self.specific_widgets:
                self.specific_widgets[pname].setValue(val)


# ─── Screen Config Widget ──────────────────────────────────────────────────


class ScreenWidget(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Screen", parent)
        form = QFormLayout(self)
        self.center_y = sci_spin(10000.0, 0, 1e8, 1)
        form.addRow("Center Y [mm]:", self.center_y)
        self.dx = sci_spin(20.0, 0.1, 1000, 1)
        form.addRow("Half-width X [mm]:", self.dx)
        self.dz = sci_spin(20.0, 0.1, 1000, 1)
        form.addRow("Half-width Z [mm]:", self.dz)
        self.nx = QSpinBox(); self.nx.setRange(10, 2000); self.nx.setValue(200)
        form.addRow("Bins X:", self.nx)
        self.nz = QSpinBox(); self.nz.setRange(10, 2000); self.nz.setValue(200)
        form.addRow("Bins Z:", self.nz)
        for w in (self.center_y, self.dx, self.dz, self.nx, self.nz):
            w.valueChanged.connect(self.changed)

    def create_screen(self, bl):
        return xr.Screen(bl, "screen", center=[0, self.center_y.value(), 0],
                         dx=self.dx.value(), dz=self.dz.value(),
                         nx=self.nx.value(), nz=self.nz.value())

    def to_dict(self):
        return dict(center_y=self.center_y.value(), dx=self.dx.value(),
                    dz=self.dz.value(), nx=self.nx.value(), nz=self.nz.value())

    def from_dict(self, d):
        self.center_y.setValue(d.get("center_y", 10000.0))
        self.dx.setValue(d.get("dx", 20.0))
        self.dz.setValue(d.get("dz", 20.0))
        self.nx.setValue(d.get("nx", 200))
        self.nz.setValue(d.get("nz", 200))


# ─── Stats Panel ───────────────────────────────────────────────────────────


class StatsWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Statistics", parent)
        form = QFormLayout(self)
        self.labels = {}
        for key in ("Total rays", "Good rays", "Captured", "FWHM X [mm]",
                     "FWHM Z [mm]", "Centroid [mm]", "RMS [mm]", "Peak intensity"):
            lbl = QLabel("—")
            form.addRow(f"{key}:", lbl)
            self.labels[key] = lbl

    def update_stats(self, beam, capture):
        self.labels["Total rays"].setText(str(beam.nrays()))
        self.labels["Good rays"].setText(str(beam.good_count()))
        self.labels["Captured"].setText(str(capture.n_captured))
        self.labels["FWHM X [mm]"].setText(f"{capture.fwhm_x:.4f}")
        self.labels["FWHM Z [mm]"].setText(f"{capture.fwhm_z:.4f}")
        self.labels["Centroid [mm]"].setText(f"({capture.centroid_x:.4f}, {capture.centroid_z:.4f})")
        self.labels["RMS [mm]"].setText(f"({capture.rms_x:.4f}, {capture.rms_z:.4f})")
        self.labels["Peak intensity"].setText(f"{capture.peak_intensity:.2f}")


# ─── 4-Panel XRT-style Plot (silx ImageView) ──────────────────────────────


class FourPanelWidget(QWidget):
    """XRT-style 4-panel plot using silx ImageView.

    ImageView provides: 2D image + X histogram (top) + Z histogram (right)
    with integrated profile tools and colormap controls.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_view = ImageView(backend=BACKEND)
        self.image_view.setGraphXLabel("X [mm]")
        self.image_view.setGraphYLabel("Z [mm]")
        self.image_view.setKeepDataAspectRatio(True)
        layout.addWidget(self.image_view)

    def update(self, x_good, z_good, e_good, dx, dz, nx, nz):
        hist, xedges, zedges = np.histogram2d(
            x_good, z_good, bins=[nx, nz], range=[[-dx, dx], [-dz, dz]],
        )
        # ImageView.setImage expects (rows, cols) with origin info
        origin = (xedges[0], zedges[0])
        scale = ((xedges[-1] - xedges[0]) / nx, (zedges[-1] - zedges[0]) / nz)
        self.image_view.setImage(hist.T, origin=origin, scale=scale, reset=True)


# ─── 3D Beamline Viewer (xrtGlow-style) ────────────────────────────────────


class GlowWidget(QWidget):
    """3D beamline viewer using pygfx/wgpu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = QRenderWidget()
        layout.addWidget(self.canvas)

        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()

        # Lighting
        self.scene.add(gfx.AmbientLight(intensity=0.4))
        dl = gfx.DirectionalLight(intensity=0.8)
        dl.local.position = (100, 200, 300)
        self.scene.add(dl)

        # Camera
        self.camera = gfx.PerspectiveCamera(50)
        self.camera.local.position = (0, -5000, 2000)
        self.camera.look_at((0, 5000, 0))
        self.controller = gfx.OrbitController(self.camera, register_events=self.renderer)

        # Groups for dynamic content
        self.beamline_group = gfx.Group()
        self.rays_group = gfx.Group()
        self.scene.add(self.beamline_group)
        self.scene.add(self.rays_group)

        # Grid floor
        grid = gfx.Grid(
            gfx.Geometry(),
            gfx.GridMaterial(
                major_step=1000, minor_step=200,
                thickness_space="screen",
                major_thickness=1, minor_thickness=0.5,
                infinite=True,
            ),
            orientation="xy",
        )
        self.scene.add(grid)

        self.canvas.request_draw(self._animate)

    def _animate(self):
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw(self._animate)

    def update_beamline(self, elements):
        """Update 3D beamline elements.

        elements: list of dicts with keys: name, y, kind, pitch, type
        """
        # Clear old
        for child in list(self.beamline_group.children):
            self.beamline_group.remove(child)

        if not elements:
            return

        # Scale: keep mm but compress Y by a factor for visibility
        y_all = [e["y"] for e in elements]
        y_span = max(y_all) - min(y_all) or 1000
        # Use real coordinates, no compression

        colors = {"source": (0.13, 0.59, 0.95, 1), "oe": (1.0, 0.6, 0.0, 1), "screen": (0.3, 0.69, 0.31, 1)}

        for el in elements:
            y = el["y"]
            kind = el["kind"]
            color = colors.get(kind, (0.5, 0.5, 0.5, 1))

            if kind == "source":
                # Sphere at source
                geo = gfx.sphere_geometry(radius=y_span * 0.01)
                mat = gfx.MeshPhongMaterial(color=color)
                mesh = gfx.Mesh(geo, mat)
                mesh.local.position = (0, y, 0)
                self.beamline_group.add(mesh)

            elif kind == "oe":
                # Flat rectangle tilted by pitch
                oe_size = y_span * 0.03
                geo = gfx.plane_geometry(width=oe_size, height=oe_size)
                mat = gfx.MeshPhongMaterial(color=color, side="both")
                mesh = gfx.Mesh(geo, mat)
                mesh.local.position = (0, y, 0)
                pitch = el.get("pitch", 0.005)
                # Rotate around X axis by pitch (plane is in XY, we want it in XZ tilted)
                mesh.local.euler_x = -np.pi / 2 + pitch
                self.beamline_group.add(mesh)

            elif kind == "screen":
                # Tall green rectangle
                oe_size = y_span * 0.03
                geo = gfx.plane_geometry(width=oe_size, height=oe_size)
                mat = gfx.MeshPhongMaterial(color=color, side="both")
                mesh = gfx.Mesh(geo, mat)
                mesh.local.position = (0, y, 0)
                mesh.local.euler_x = -np.pi / 2
                self.beamline_group.add(mesh)

        # Beam path line
        positions = np.array(
            [[0, e["y"], 0] for e in sorted(elements, key=lambda e: e["y"])],
            dtype=np.float32,
        )
        if len(positions) >= 2:
            line = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineMaterial(color=(0.26, 0.65, 0.96, 1), thickness=3),
            )
            self.beamline_group.add(line)

    def update_rays(self, x, z, e, screen_y):
        """Show ray positions on screen as 3D points colored by energy."""
        for child in list(self.rays_group.children):
            self.rays_group.remove(child)

        if len(x) == 0:
            return

        # Subsample
        max_pts = 5000
        if len(x) > max_pts:
            idx = np.random.choice(len(x), max_pts, replace=False)
            x, z, e = x[idx], z[idx], e[idx]

        positions = np.column_stack([x, np.full_like(x, screen_y), z]).astype(np.float32)
        colors = energy_to_rgb(e)

        pts = gfx.Points(
            gfx.Geometry(positions=positions, colors=colors),
            gfx.PointsMaterial(size=3, color_mode="vertex"),
        )
        self.rays_group.add(pts)

    def fit_view(self):
        try:
            self.camera.show_object(self.scene, view_dir=(0.5, -1, 0.3), up=(0, 0, 1))
        except ValueError:
            pass  # empty scene


# ─── Code Preview Dialog ───────────────────────────────────────────────────


class CodeDialog(QDialog):
    def __init__(self, code, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated Python Code")
        self.resize(700, 600)
        self._code = code
        layout = QVBoxLayout(self)
        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setFont(QFont("Menlo", 12))
        self.editor.setPlainText(code)
        layout.addWidget(self.editor)
        btn_row = QHBoxLayout()
        for text, fn in [("Copy", self._copy), ("Save .py", self._save), ("Close", self.accept)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

    def _copy(self):
        QApplication.clipboard().setText(self._code)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "beamline.py", "Python (*.py)")
        if path:
            with open(path, "w") as f:
                f.write(self._code)


# ─── Main Window ────────────────────────────────────────────────────────────


class BeamlineWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRT-RS Beamline Simulator")
        self.resize(1400, 900)
        self.oe_widgets = []
        self._layout_file = ""
        self._build_ui()
        self._connect_signals()
        QTimer.singleShot(100, self._update_layout)

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ── Left: config ──
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setMinimumWidth(340)
        config_scroll.setMaximumWidth(450)
        config_widget = QWidget()
        cl = QVBoxLayout(config_widget)

        self.source_widget = SourceWidget()
        cl.addWidget(self.source_widget)

        oe_header = QHBoxLayout()
        oe_header.addWidget(QLabel("Optical Elements"))
        add_oe_btn = QPushButton("+ Add OE")
        add_oe_btn.clicked.connect(self._add_oe)
        oe_header.addWidget(add_oe_btn)
        cl.addLayout(oe_header)
        self.oe_container = QVBoxLayout()
        cl.addLayout(self.oe_container)

        self.screen_widget = ScreenWidget()
        cl.addWidget(self.screen_widget)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setStyleSheet(
            "QPushButton{background:#2d7d46;color:white;font-weight:bold;padding:8px;font-size:14px}"
            "QPushButton:hover{background:#3a9957}")
        btn_row.addWidget(self.run_btn)
        export_btn = QPushButton("Export .py")
        export_btn.clicked.connect(self._export_code)
        btn_row.addWidget(export_btn)
        cl.addLayout(btn_row)

        io_row = QHBoxLayout()
        save_btn = QPushButton("Save Layout")
        save_btn.clicked.connect(self._save_layout)
        io_row.addWidget(save_btn)
        load_btn = QPushButton("Load Layout")
        load_btn.clicked.connect(self._load_layout)
        io_row.addWidget(load_btn)
        cl.addLayout(io_row)

        self.stats_widget = StatsWidget()
        cl.addWidget(self.stats_widget)
        cl.addStretch()
        config_scroll.setWidget(config_widget)
        splitter.addWidget(config_scroll)

        # ── Right: plots ──
        plot_tabs = QTabWidget()

        # 4-panel XRT-style
        self.four_panel = FourPanelWidget()
        plot_tabs.addTab(self.four_panel, "Beam Profile")

        # 2D footprint
        self.plot_2d = Plot2D(backend=BACKEND)
        self.plot_2d.setGraphTitle("Screen Footprint")
        self.plot_2d.setGraphXLabel("X [mm]")
        self.plot_2d.setGraphYLabel("Z [mm]")
        self.plot_2d.setKeepDataAspectRatio(True)
        plot_tabs.addTab(self.plot_2d, "2D Footprint")

        # X projection
        self.plot_x = Plot1D(backend=BACKEND)
        self.plot_x.setGraphTitle("X Projection")
        self.plot_x.setGraphXLabel("X [mm]")
        self.plot_x.setGraphYLabel("Intensity")
        plot_tabs.addTab(self.plot_x, "X Profile")

        # Z projection
        self.plot_z = Plot1D(backend=BACKEND)
        self.plot_z.setGraphTitle("Z Projection")
        self.plot_z.setGraphXLabel("Z [mm]")
        self.plot_z.setGraphYLabel("Intensity")
        plot_tabs.addTab(self.plot_z, "Z Profile")

        # Energy scatter
        self.plot_scatter = Plot2D(backend=BACKEND)
        self.plot_scatter.setGraphTitle("Ray Scatter (color = energy)")
        self.plot_scatter.setGraphXLabel("X [mm]")
        self.plot_scatter.setGraphYLabel("Z [mm]")
        self.plot_scatter.setKeepDataAspectRatio(True)
        plot_tabs.addTab(self.plot_scatter, "Energy Scatter")

        # Layout side view
        self.plot_layout = Plot1D(backend=BACKEND)
        self.plot_layout.setGraphTitle("Beamline Layout (side view)")
        self.plot_layout.setGraphXLabel("Y [mm]  (beam direction)")
        self.plot_layout.setGraphYLabel("Z [mm]  (vertical)")
        plot_tabs.addTab(self.plot_layout, "Layout")

        # 3D Glow viewer
        self.glow = GlowWidget()
        plot_tabs.addTab(self.glow, "3D View")

        splitter.addWidget(plot_tabs)
        splitter.setSizes([380, 1020])

        self.statusBar().showMessage("Ready")
        self._add_oe()

    def _connect_signals(self):
        self.run_btn.clicked.connect(self._run_simulation)
        self.source_widget.changed.connect(self._update_layout)
        self.screen_widget.changed.connect(self._update_layout)

    def _add_oe(self, oe_type="ToroidMirror"):
        idx = len(self.oe_widgets) + 1
        oe = OEWidget(name=f"OE{idx}", oe_type=oe_type)
        oe.remove_requested.connect(self._remove_oe)
        oe.changed.connect(self._update_layout)
        self.oe_container.addWidget(oe)
        self.oe_widgets.append(oe)
        self._update_layout()
        return oe

    def _remove_oe(self, oe_widget):
        if oe_widget in self.oe_widgets:
            self.oe_widgets.remove(oe_widget)
            self.oe_container.removeWidget(oe_widget)
            oe_widget.deleteLater()
            self._update_layout()

    # ── Simulation ──

    @Slot()
    def _run_simulation(self):
        self.statusBar().showMessage("Running simulation...")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            bl = xr.BeamLine()
            source = self.source_widget.create_source(bl)
            beam = source.shine()
            for oe_w in self.oe_widgets:
                oe = oe_w.create_oe(bl)
                oe.reflect(beam)
            screen = self.screen_widget.create_screen(bl)
            capture = screen.expose(beam)
            self.stats_widget.update_stats(beam, capture)
            self._update_plots(beam)
            self._update_layout()
            self._update_3d_rays(beam)
            self.statusBar().showMessage(
                f"Done: {beam.good_count()} good, {capture.n_captured} captured, "
                f"FWHM=({capture.fwhm_x:.4f}, {capture.fwhm_z:.4f}) mm")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.run_btn.setEnabled(True)

    def _update_plots(self, beam):
        dx = self.screen_widget.dx.value()
        dz = self.screen_widget.dz.value()
        nx = self.screen_widget.nx.value()
        nz = self.screen_widget.nz.value()

        good = beam.state == 1
        x_good, z_good, e_good = beam.x[good], beam.z[good], beam.e[good]

        # 4-panel
        self.four_panel.update(x_good, z_good, e_good, dx, dz, nx, nz)

        # 2D histogram for individual tabs
        hist, xedges, zedges = np.histogram2d(
            x_good, z_good, bins=[nx, nz], range=[[-dx, dx], [-dz, dz]])

        self.plot_2d.addImage(
            hist.T, legend="footprint",
            origin=(xedges[0], zedges[0]),
            scale=((xedges[-1] - xedges[0]) / nx, (zedges[-1] - zedges[0]) / nz),
            replace=True)
        self.plot_2d.resetZoom()

        xc = 0.5 * (xedges[:-1] + xedges[1:])
        self.plot_x.addCurve(xc, hist.sum(axis=1), legend="X", replace=True)
        self.plot_x.resetZoom()

        zc = 0.5 * (zedges[:-1] + zedges[1:])
        self.plot_z.addCurve(zc, hist.sum(axis=0), legend="Z", replace=True)
        self.plot_z.resetZoom()

        # Energy scatter (subsampled)
        max_s = 10000
        if len(x_good) > max_s:
            idx = np.random.choice(len(x_good), max_s, replace=False)
            xs, zs, es = x_good[idx], z_good[idx], e_good[idx]
        else:
            xs, zs, es = x_good, z_good, e_good

        self.plot_scatter.remove("rays")
        self.plot_scatter.addScatter(xs, zs, es, legend="rays", symbol=".")
        self.plot_scatter.resetZoom()

    def _update_3d_rays(self, beam):
        good = beam.state == 1
        x, z, e = beam.x[good], beam.z[good], beam.e[good]
        screen_y = self.screen_widget.center_y.value()
        self.glow.update_rays(x, z, e, screen_y)

    # ── Layout ──

    def _collect_elements(self):
        elements = [{"name": "Source", "y": 0.0, "kind": "source"}]
        for oe_w in self.oe_widgets:
            elements.append({
                "name": oe_w._name, "y": oe_w.center_y.value(),
                "pitch": oe_w.pitch.value(), "kind": "oe",
                "type": oe_w.type_combo.currentText(),
            })
        elements.append({"name": "Screen", "y": self.screen_widget.center_y.value(), "kind": "screen"})
        elements.sort(key=lambda e: e["y"])
        return elements

    def _update_layout(self):
        elements = self._collect_elements()

        # 2D side view
        p = self.plot_layout
        p.clear()
        y_all = [e["y"] for e in elements]
        y_min, y_max = min(y_all), max(y_all)
        margin = (y_max - y_min) * 0.05 or 500
        p.addCurve([y_min - margin, y_max + margin], [0, 0],
                   legend="_axis", color="#cccccc", linestyle="--", linewidth=1)
        cmap = {"source": "#2196F3", "oe": "#FF9800", "screen": "#4CAF50"}
        for el in elements:
            y, kind, color = el["y"], el["kind"], cmap[el["kind"]]
            if kind == "source":
                p.addMarker(y, 0, legend=f"_m_{el['name']}", text=el["name"], color=color, symbol="d")
            elif kind == "screen":
                p.addCurve([y, y], [-3, 3], legend=f"_e_{el['name']}", color=color, linewidth=3)
                p.addMarker(y, 3.5, legend=f"_m_{el['name']}", text=el["name"], color=color, symbol="s")
            elif kind == "oe":
                pitch = el.get("pitch", 0.005)
                va = max(abs(pitch) * 200, 0.3)
                h = 2.0
                p.addCurve([y - h * np.sin(va), y + h * np.sin(va)],
                           [-h * np.cos(va), h * np.cos(va)],
                           legend=f"_e_{el['name']}", color=color, linewidth=3)
                p.addMarker(y, h * np.cos(va) + 1.0, legend=f"_m_{el['name']}",
                            text=f"{el['name']}\n({el.get('type', '')})", color=color, symbol="o")
        for i in range(len(elements) - 1):
            p.addCurve([elements[i]["y"], elements[i + 1]["y"]], [0, 0],
                       legend=f"_b_{i}", color="#42A5F5", linewidth=2)
        p.setGraphYLimits(-8, 8)
        p.resetZoom()

        # 3D view
        self.glow.update_beamline(elements)
        self.glow.fit_view()

    # ── Save / Load ──

    @Slot()
    def _save_layout(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Layout",
                                              self._layout_file or "beamline.json", "JSON (*.json)")
        if not path:
            return
        self._layout_file = path
        data = dict(version=1, source=self.source_widget.to_dict(),
                    oes=[oe.to_dict() for oe in self.oe_widgets],
                    screen=self.screen_widget.to_dict())
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.setWindowTitle(f"XRT-RS — {os.path.basename(path)}")
        self.statusBar().showMessage(f"Saved to {path}", 3000)

    @Slot()
    def _load_layout(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Layout", "", "JSON (*.json);;All (*)")
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self._layout_file = path
        self.source_widget.from_dict(data.get("source", {}))
        for oe in list(self.oe_widgets):
            self.oe_widgets.remove(oe)
            self.oe_container.removeWidget(oe)
            oe.deleteLater()
        for oe_data in data.get("oes", []):
            oe = self._add_oe(oe_data.get("type", "FlatMirror"))
            oe.from_dict(oe_data)
        self.screen_widget.from_dict(data.get("screen", {}))
        self._update_layout()
        self.setWindowTitle(f"XRT-RS — {os.path.basename(path)}")
        self.statusBar().showMessage(f"Loaded from {path}", 3000)

    # ── Export Code ──

    @Slot()
    def _export_code(self):
        CodeDialog(self._generate_code(), self).exec()

    def _generate_code(self):
        src = self.source_widget.to_dict()
        scr = self.screen_widget.to_dict()
        lines = [
            f'"""Beamline simulation — generated {date.today()} by XRT-RS GUI."""',
            "", "import xrt_rs as xr", "", "",
            "def main():",
            "    bl = xr.BeamLine()", "",
            "    source = xr.GeometricSource(",
            f"        bl, 'source', nrays={src['nrays']},",
            f"        dx={src['dx']}, dz={src['dz']},",
            f"        dxprime={src['dxprime']}, dzprime={src['dzprime']},",
            f"        energies=[{src['energy']}],",
            "    )", "",
        ]
        mat_vars = {}
        for oe_w in self.oe_widgets:
            od = oe_w.to_dict()
            md = od["material"]
            if md["enabled"] and od["type"] not in ("EllipticalMirror", "ParabolicalMirror"):
                mk = f"{md['elements']}_{md['rho']}"
                if mk not in mat_vars:
                    v = f"mat{len(mat_vars) + 1}"
                    mat_vars[mk] = v
                    elems = [e.strip() for e in md["elements"].split(",")]
                    lines += [f"    {v} = xr.Material({elems}, rho={md['rho']})", ""]

        for i, oe_w in enumerate(self.oe_widgets):
            od = oe_w.to_dict()
            var = f"oe{i + 1}"
            args = [f"bl, '{od['name']}'", f"center=[0, {od['center_y']}, 0]", f"pitch={od['pitch']}"]
            for pn, pv in od["specific"].items():
                args.append(f"{pn}={pv}")
            md = od["material"]
            if md["enabled"] and od["type"] not in ("EllipticalMirror", "ParabolicalMirror"):
                mk = f"{md['elements']}_{md['rho']}"
                args.append(f"material={mat_vars[mk]}")
            lines.append(f"    {var} = xr.{od['type']}(")
            for a in args:
                lines.append(f"        {a},")
            lines += [f"    )", ""]

        lines += [
            "    screen = xr.Screen(",
            f"        bl, 'screen', center=[0, {scr['center_y']}, 0],",
            f"        dx={scr['dx']}, dz={scr['dz']}, nx={scr['nx']}, nz={scr['nz']},",
            "    )", "",
            "    beam = source.shine()",
        ]
        for i in range(len(self.oe_widgets)):
            lines.append(f"    oe{i + 1}.reflect(beam)")
        lines += [
            "    result = screen.expose(beam)", "",
            "    print(f'Good rays: {beam.good_count()}')",
            "    print(f'FWHM: {result.fwhm_x:.4f} x {result.fwhm_z:.4f} mm')",
            "    print(f'Centroid: ({result.centroid_x:.4f}, {result.centroid_z:.4f})')",
            "", "", "if __name__ == '__main__':", "    main()", "",
        ]
        return "\n".join(lines)


# ─── Entry Point ────────────────────────────────────────────────────────────


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BeamlineWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
