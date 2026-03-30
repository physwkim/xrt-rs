#!/usr/bin/env python3
"""XRT-RS Beamline Simulator GUI.

qtpy + silx (pygfx/wgpu backend) GUI for interactive beamline simulation.

Features:
  - Source / OE / Screen configuration
  - 2D footprint, X/Z profiles, energy-colored ray scatter
  - Beamline layout side view
  - Export Python code
  - Save / Load layout (JSON)
"""

import json
import os
import sys
import textwrap
from datetime import date

import numpy as np

from qtpy.QtCore import Qt, Signal, Slot, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
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
    QStatusBar,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QFileDialog,
    QPlainTextEdit,
)
from qtpy.QtGui import QAction, QIcon, QFont

from silx.gui.plot import Plot2D, Plot1D

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
    """Create a QDoubleSpinBox with scientific-friendly defaults."""
    sb = QDoubleSpinBox()
    sb.setRange(minimum, maximum)
    sb.setDecimals(decimals)
    sb.setValue(value)
    sb.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
    sb.setMinimumWidth(120)
    return sb


# ─── Source Config Widget ───────────────────────────────────────────────────


class SourceWidget(QGroupBox):
    """Configuration widget for the geometric source."""

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
            bl,
            "source",
            nrays=self.nrays.value(),
            dx=self.dx.value(),
            dz=self.dz.value(),
            dxprime=self.dxprime.value(),
            dzprime=self.dzprime.value(),
            energies=[self.energy.value()],
        )

    def to_dict(self):
        return {
            "nrays": self.nrays.value(),
            "energy": self.energy.value(),
            "dx": self.dx.value(),
            "dz": self.dz.value(),
            "dxprime": self.dxprime.value(),
            "dzprime": self.dzprime.value(),
        }

    def from_dict(self, d):
        self.nrays.setValue(d.get("nrays", 50000))
        self.energy.setValue(d.get("energy", 10000.0))
        self.dx.setValue(d.get("dx", 0.1))
        self.dz.setValue(d.get("dz", 0.05))
        self.dxprime.setValue(d.get("dxprime", 1e-4))
        self.dzprime.setValue(d.get("dzprime", 5e-5))


# ─── Material Config Widget ────────────────────────────────────────────────


class MaterialWidget(QGroupBox):
    """Configuration widget for a material."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Material", parent)
        form = QFormLayout(self)

        self.elements = QLineEdit("Si")
        self.elements.setToolTip("Comma-separated element symbols, e.g. Si or Si,O")
        form.addRow("Elements:", self.elements)

        self.rho = sci_spin(2.33, 0.001, 30.0, 3)
        form.addRow("Density [g/cm³]:", self.rho)

        self.enabled = QPushButton("Material ON")
        self.enabled.setCheckable(True)
        self.enabled.setChecked(True)
        self.enabled.toggled.connect(self._on_toggle)
        form.addRow(self.enabled)

        self.elements.textChanged.connect(self.changed)
        self.rho.valueChanged.connect(self.changed)
        self.enabled.toggled.connect(self.changed)

    def _on_toggle(self, checked):
        self.enabled.setText("Material ON" if checked else "Material OFF")

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
        return {
            "elements": self.elements.text(),
            "rho": self.rho.value(),
            "enabled": self.enabled.isChecked(),
        }

    def from_dict(self, d):
        self.elements.setText(d.get("elements", "Si"))
        self.rho.setValue(d.get("rho", 2.33))
        self.enabled.setChecked(d.get("enabled", True))


# ─── OE Config Widget ──────────────────────────────────────────────────────


class OEWidget(QGroupBox):
    """Configuration widget for a single optical element."""

    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, name="OE1", oe_type="FlatMirror", parent=None):
        super().__init__(name, parent)
        self._name = name
        layout = QVBoxLayout(self)

        # Type selector + remove button
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

        # Common params
        form = QFormLayout()
        self.center_y = sci_spin(5000.0, -1e6, 1e6, 1)
        form.addRow("Center Y [mm]:", self.center_y)

        self.pitch = sci_spin(0.005, -1, 1)
        form.addRow("Pitch [rad]:", self.pitch)
        layout.addLayout(form)

        # Type-specific params container
        self.specific_form = QFormLayout()
        self.specific_widgets = {}
        layout.addLayout(self.specific_form)

        # Material
        self.material = MaterialWidget()
        layout.addWidget(self.material)

        # Build type-specific params
        self.type_combo.currentTextChanged.connect(self._rebuild_specific)
        self._rebuild_specific(oe_type)

        # Connect signals
        self.type_combo.currentTextChanged.connect(lambda: self.changed.emit())
        self.center_y.valueChanged.connect(self.changed)
        self.pitch.valueChanged.connect(self.changed)
        self.material.changed.connect(self.changed)

    def _rebuild_specific(self, oe_type):
        while self.specific_form.rowCount() > 0:
            self.specific_form.removeRow(0)
        self.specific_widgets.clear()

        info = OE_TYPES.get(oe_type, {})
        for pname, pdefault in info.get("params", {}).items():
            if isinstance(pdefault, int):
                w = QSpinBox()
                w.setRange(-100, 100)
                w.setValue(pdefault)
            else:
                w = sci_spin(pdefault)
            self.specific_widgets[pname] = w
            self.specific_form.addRow(f"{pname}:", w)
            w.valueChanged.connect(self.changed)

    def create_oe(self, bl):
        oe_type = self.type_combo.currentText()
        info = OE_TYPES[oe_type]
        cls = info["class"]

        kwargs = {
            "center": [0, self.center_y.value(), 0],
            "pitch": self.pitch.value(),
        }

        for pname, widget in self.specific_widgets.items():
            kwargs[pname] = widget.value()

        if oe_type not in ("EllipticalMirror", "ParabolicalMirror"):
            mat = self.material.create_material()
            if mat is not None:
                kwargs["material"] = mat

        return cls(bl, self._name, **kwargs)

    def to_dict(self):
        d = {
            "name": self._name,
            "type": self.type_combo.currentText(),
            "center_y": self.center_y.value(),
            "pitch": self.pitch.value(),
            "material": self.material.to_dict(),
            "specific": {},
        }
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
    """Configuration widget for the screen/detector."""

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

        self.nx = QSpinBox()
        self.nx.setRange(10, 2000)
        self.nx.setValue(200)
        form.addRow("Bins X:", self.nx)

        self.nz = QSpinBox()
        self.nz.setRange(10, 2000)
        self.nz.setValue(200)
        form.addRow("Bins Z:", self.nz)

        for w in (self.center_y, self.dx, self.dz, self.nx, self.nz):
            w.valueChanged.connect(self.changed)

    def create_screen(self, bl):
        return xr.Screen(
            bl,
            "screen",
            center=[0, self.center_y.value(), 0],
            dx=self.dx.value(),
            dz=self.dz.value(),
            nx=self.nx.value(),
            nz=self.nz.value(),
        )

    def to_dict(self):
        return {
            "center_y": self.center_y.value(),
            "dx": self.dx.value(),
            "dz": self.dz.value(),
            "nx": self.nx.value(),
            "nz": self.nz.value(),
        }

    def from_dict(self, d):
        self.center_y.setValue(d.get("center_y", 10000.0))
        self.dx.setValue(d.get("dx", 20.0))
        self.dz.setValue(d.get("dz", 20.0))
        self.nx.setValue(d.get("nx", 200))
        self.nz.setValue(d.get("nz", 200))


# ─── Stats Panel ───────────────────────────────────────────────────────────


class StatsWidget(QGroupBox):
    """Displays beam and screen capture statistics."""

    def __init__(self, parent=None):
        super().__init__("Statistics", parent)
        form = QFormLayout(self)

        self.lbl_nrays = QLabel("—")
        self.lbl_good = QLabel("—")
        self.lbl_captured = QLabel("—")
        self.lbl_fwhm_x = QLabel("—")
        self.lbl_fwhm_z = QLabel("—")
        self.lbl_centroid = QLabel("—")
        self.lbl_rms = QLabel("—")
        self.lbl_peak = QLabel("—")

        form.addRow("Total rays:", self.lbl_nrays)
        form.addRow("Good rays:", self.lbl_good)
        form.addRow("Captured:", self.lbl_captured)
        form.addRow("FWHM X [mm]:", self.lbl_fwhm_x)
        form.addRow("FWHM Z [mm]:", self.lbl_fwhm_z)
        form.addRow("Centroid [mm]:", self.lbl_centroid)
        form.addRow("RMS [mm]:", self.lbl_rms)
        form.addRow("Peak intensity:", self.lbl_peak)

    def update_stats(self, beam, capture):
        self.lbl_nrays.setText(str(beam.nrays()))
        self.lbl_good.setText(str(beam.good_count()))
        self.lbl_captured.setText(str(capture.n_captured))
        self.lbl_fwhm_x.setText(f"{capture.fwhm_x:.4f}")
        self.lbl_fwhm_z.setText(f"{capture.fwhm_z:.4f}")
        self.lbl_centroid.setText(f"({capture.centroid_x:.4f}, {capture.centroid_z:.4f})")
        self.lbl_rms.setText(f"({capture.rms_x:.4f}, {capture.rms_z:.4f})")
        self.lbl_peak.setText(f"{capture.peak_intensity:.2f}")

    def clear(self):
        for lbl in (
            self.lbl_nrays, self.lbl_good, self.lbl_captured,
            self.lbl_fwhm_x, self.lbl_fwhm_z, self.lbl_centroid,
            self.lbl_rms, self.lbl_peak,
        ):
            lbl.setText("—")


# ─── Code Preview Dialog ───────────────────────────────────────────────────


class CodeDialog(QDialog):
    """Dialog showing generated Python code with copy/save buttons."""

    def __init__(self, code: str, parent=None):
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
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self._copy)
        btn_row.addWidget(copy_btn)

        save_btn = QPushButton("Save as .py")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    def _copy(self):
        QApplication.clipboard().setText(self._code)
        self.parent().statusBar().showMessage("Code copied to clipboard", 3000)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Python Script", "beamline.py", "Python files (*.py)"
        )
        if path:
            with open(path, "w") as f:
                f.write(self._code)
            self.parent().statusBar().showMessage(f"Saved to {path}", 3000)


# ─── Main Window ────────────────────────────────────────────────────────────


class BeamlineWindow(QMainWindow):
    """Main beamline simulator window."""

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
        # Central splitter: left config | right plots
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ── Left: config panel ──
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setMinimumWidth(340)
        config_scroll.setMaximumWidth(450)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        # Source
        self.source_widget = SourceWidget()
        config_layout.addWidget(self.source_widget)

        # OE list
        oe_header = QHBoxLayout()
        oe_header.addWidget(QLabel("Optical Elements"))
        add_oe_btn = QPushButton("+ Add OE")
        add_oe_btn.clicked.connect(self._add_oe)
        oe_header.addWidget(add_oe_btn)
        config_layout.addLayout(oe_header)

        self.oe_container = QVBoxLayout()
        config_layout.addLayout(self.oe_container)

        # Screen
        self.screen_widget = ScreenWidget()
        config_layout.addWidget(self.screen_widget)

        # Buttons row
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("Run")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: white; "
            "font-weight: bold; padding: 8px; font-size: 14px; }"
            "QPushButton:hover { background-color: #3a9957; }"
        )
        btn_row.addWidget(self.run_btn)

        export_btn = QPushButton("Export .py")
        export_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-size: 13px; }"
        )
        export_btn.clicked.connect(self._export_code)
        btn_row.addWidget(export_btn)
        config_layout.addLayout(btn_row)

        # Save/Load row
        io_row = QHBoxLayout()
        save_btn = QPushButton("Save Layout")
        save_btn.clicked.connect(self._save_layout)
        io_row.addWidget(save_btn)

        load_btn = QPushButton("Load Layout")
        load_btn.clicked.connect(self._load_layout)
        io_row.addWidget(load_btn)
        config_layout.addLayout(io_row)

        # Stats
        self.stats_widget = StatsWidget()
        config_layout.addWidget(self.stats_widget)

        config_layout.addStretch()
        config_scroll.setWidget(config_widget)
        splitter.addWidget(config_scroll)

        # ── Right: plots ──
        plot_tabs = QTabWidget()

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

        # Scatter (raw rays) - energy colored
        self.plot_scatter = Plot2D(backend=BACKEND)
        self.plot_scatter.setGraphTitle("Ray Scatter (color = energy)")
        self.plot_scatter.setGraphXLabel("X [mm]")
        self.plot_scatter.setGraphYLabel("Z [mm]")
        self.plot_scatter.setKeepDataAspectRatio(True)
        plot_tabs.addTab(self.plot_scatter, "Energy Scatter")

        # Beamline layout (side view)
        self.plot_layout = Plot1D(backend=BACKEND)
        self.plot_layout.setGraphTitle("Beamline Layout (side view)")
        self.plot_layout.setGraphXLabel("Y [mm]  (beam direction)")
        self.plot_layout.setGraphYLabel("Z [mm]  (vertical)")
        plot_tabs.addTab(self.plot_layout, "Layout")

        splitter.addWidget(plot_tabs)
        splitter.setSizes([380, 1020])

        # Status bar
        self.statusBar().showMessage("Ready")

        # Add a default OE
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

    # ── Simulation ──────────────────────────────────────────────────────

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
                try:
                    oe = oe_w.create_oe(bl)
                    oe.reflect(beam)
                except Exception as e:
                    QMessageBox.warning(self, "OE Error", str(e))
                    return

            screen = self.screen_widget.create_screen(bl)
            capture = screen.expose(beam)

            self.stats_widget.update_stats(beam, capture)
            self._update_plots(beam, capture, screen)
            self._update_layout()

            self.statusBar().showMessage(
                f"Done: {beam.good_count()} good rays, "
                f"{capture.n_captured} captured, "
                f"FWHM=({capture.fwhm_x:.4f}, {capture.fwhm_z:.4f}) mm"
            )

        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            QMessageBox.critical(self, "Simulation Error", str(e))
        finally:
            self.run_btn.setEnabled(True)

    def _update_plots(self, beam, capture, screen):
        dx = self.screen_widget.dx.value()
        dz = self.screen_widget.dz.value()
        nx = self.screen_widget.nx.value()
        nz = self.screen_widget.nz.value()

        x = beam.x
        z = beam.z
        e = beam.e
        state = beam.state
        good = state == 1

        x_good = x[good]
        z_good = z[good]
        e_good = e[good]

        # 2D histogram
        hist, xedges, zedges = np.histogram2d(
            x_good, z_good, bins=[nx, nz],
            range=[[-dx, dx], [-dz, dz]],
        )

        # 2D footprint
        self.plot_2d.addImage(
            hist.T,
            legend="footprint",
            origin=(xedges[0], zedges[0]),
            scale=(
                (xedges[-1] - xedges[0]) / nx,
                (zedges[-1] - zedges[0]) / nz,
            ),
            replace=True,
        )
        self.plot_2d.resetZoom()

        # X projection
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        x_proj = hist.sum(axis=1)
        self.plot_x.addCurve(x_centers, x_proj, legend="X projection", replace=True)
        self.plot_x.resetZoom()

        # Z projection
        z_centers = 0.5 * (zedges[:-1] + zedges[1:])
        z_proj = hist.sum(axis=0)
        self.plot_z.addCurve(z_centers, z_proj, legend="Z projection", replace=True)
        self.plot_z.resetZoom()

        # Energy-colored scatter plot
        max_scatter = 10000
        if len(x_good) > max_scatter:
            idx = np.random.choice(len(x_good), max_scatter, replace=False)
            xs, zs, es = x_good[idx], z_good[idx], e_good[idx]
        else:
            xs, zs, es = x_good, z_good, e_good

        self.plot_scatter.remove("rays")
        self.plot_scatter.addScatter(
            xs, zs, es,
            legend="rays",
            symbol=".",
        )
        self.plot_scatter.resetZoom()

    # ── Layout View ─────────────────────────────────────────────────────

    def _update_layout(self):
        """Draw beamline layout: side view (Y along beam, Z vertical)."""
        p = self.plot_layout
        p.clear()

        elements = []
        elements.append({"name": "Source", "y": 0.0, "kind": "source"})

        for oe_w in self.oe_widgets:
            elements.append({
                "name": oe_w._name,
                "y": oe_w.center_y.value(),
                "pitch": oe_w.pitch.value(),
                "kind": "oe",
                "type": oe_w.type_combo.currentText(),
            })

        elements.append({
            "name": "Screen",
            "y": self.screen_widget.center_y.value(),
            "kind": "screen",
        })
        elements.sort(key=lambda e: e["y"])

        y_all = [e["y"] for e in elements]
        y_min, y_max = min(y_all), max(y_all)
        margin = (y_max - y_min) * 0.05 or 500

        p.addCurve(
            [y_min - margin, y_max + margin], [0, 0],
            legend="_beam_axis", color="#cccccc", linestyle="--", linewidth=1,
        )

        colors_map = {"source": "#2196F3", "oe": "#FF9800", "screen": "#4CAF50"}
        for el in elements:
            y = el["y"]
            color = colors_map[el["kind"]]

            if el["kind"] == "source":
                p.addMarker(y, 0, legend=f"_m_{el['name']}",
                            text=el["name"], color=color, symbol="d")
            elif el["kind"] == "screen":
                p.addCurve([y, y], [-3, 3], legend=f"_el_{el['name']}",
                           color=color, linewidth=3)
                p.addMarker(y, 3.5, legend=f"_m_{el['name']}",
                            text=el["name"], color=color, symbol="s")
            elif el["kind"] == "oe":
                pitch = el.get("pitch", 0.005)
                vis_angle = max(abs(pitch) * 200, 0.3)
                half = 2.0
                dy = half * np.sin(vis_angle)
                dz = half * np.cos(vis_angle)
                p.addCurve(
                    [y - dy, y + dy], [-dz, dz],
                    legend=f"_el_{el['name']}", color=color, linewidth=3,
                )
                label = f"{el['name']}\n({el.get('type', '')})"
                p.addMarker(y, dz + 1.0, legend=f"_m_{el['name']}",
                            text=label, color=color, symbol="o")

        for i in range(len(elements) - 1):
            p.addCurve(
                [elements[i]["y"], elements[i + 1]["y"]], [0, 0],
                legend=f"_beam_{i}", color="#42A5F5", linewidth=2,
            )

        p.setGraphYLimits(-8, 8)
        p.resetZoom()

    # ── Save / Load Layout (JSON) ───────────────────────────────────────

    def _get_layout_dict(self):
        return {
            "version": 1,
            "source": self.source_widget.to_dict(),
            "oes": [oe.to_dict() for oe in self.oe_widgets],
            "screen": self.screen_widget.to_dict(),
        }

    def _apply_layout_dict(self, data):
        # Source
        self.source_widget.from_dict(data.get("source", {}))

        # Remove existing OEs
        for oe in list(self.oe_widgets):
            self.oe_widgets.remove(oe)
            self.oe_container.removeWidget(oe)
            oe.deleteLater()

        # Add OEs from data
        for oe_data in data.get("oes", []):
            oe = self._add_oe(oe_data.get("type", "FlatMirror"))
            oe.from_dict(oe_data)

        # Screen
        self.screen_widget.from_dict(data.get("screen", {}))
        self._update_layout()

    @Slot()
    def _save_layout(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Beamline Layout", self._layout_file or "beamline.json",
            "JSON files (*.json)",
        )
        if not path:
            return
        self._layout_file = path
        data = self._get_layout_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.setWindowTitle(f"XRT-RS — {os.path.basename(path)}")
        self.statusBar().showMessage(f"Layout saved to {path}", 3000)

    @Slot()
    def _load_layout(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Beamline Layout", "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self._layout_file = path
        self._apply_layout_dict(data)
        self.setWindowTitle(f"XRT-RS — {os.path.basename(path)}")
        self.statusBar().showMessage(f"Layout loaded from {path}", 3000)

    # ── Export Python Code ──────────────────────────────────────────────

    @Slot()
    def _export_code(self):
        code = self._generate_code()
        dlg = CodeDialog(code, self)
        dlg.exec()

    def _generate_code(self):
        src = self.source_widget.to_dict()
        scr = self.screen_widget.to_dict()

        lines = [
            f'"""Beamline simulation generated by XRT-RS GUI on {date.today()}."""',
            "",
            "import xrt_rs as xr",
            "",
            "",
            "def main():",
            "    bl = xr.BeamLine()",
            "",
            "    # Source",
            f"    source = xr.GeometricSource(",
            f"        bl, 'source',",
            f"        nrays={src['nrays']},",
            f"        dx={src['dx']}, dz={src['dz']},",
            f"        dxprime={src['dxprime']}, dzprime={src['dzprime']},",
            f"        energies=[{src['energy']}],",
            f"    )",
            "",
        ]

        # Materials
        mat_vars = {}
        for i, oe_w in enumerate(self.oe_widgets):
            od = oe_w.to_dict()
            md = od["material"]
            if md["enabled"]:
                mat_key = f"{md['elements']}_{md['rho']}"
                if mat_key not in mat_vars:
                    var = f"mat{len(mat_vars) + 1}"
                    mat_vars[mat_key] = var
                    elems = [e.strip() for e in md["elements"].split(",")]
                    lines.append(f"    # Material: {md['elements']}")
                    lines.append(
                        f"    {var} = xr.Material({elems}, rho={md['rho']})"
                    )
                    lines.append("")

        # OEs
        for i, oe_w in enumerate(self.oe_widgets):
            od = oe_w.to_dict()
            cls_name = od["type"]
            name = od["name"]
            var = f"oe{i + 1}"

            args = [
                f"bl, '{name}'",
                f"center=[0, {od['center_y']}, 0]",
                f"pitch={od['pitch']}",
            ]
            for pname, pval in od["specific"].items():
                args.append(f"{pname}={pval}")

            md = od["material"]
            if md["enabled"] and cls_name not in ("EllipticalMirror", "ParabolicalMirror"):
                mat_key = f"{md['elements']}_{md['rho']}"
                args.append(f"material={mat_vars[mat_key]}")

            lines.append(f"    # {name} ({cls_name})")
            lines.append(f"    {var} = xr.{cls_name}(")
            for j, a in enumerate(args):
                comma = "," if j < len(args) - 1 else ","
                lines.append(f"        {a}{comma}")
            lines.append(f"    )")
            lines.append("")

        # Screen
        lines.extend([
            "    # Screen",
            f"    screen = xr.Screen(",
            f"        bl, 'screen',",
            f"        center=[0, {scr['center_y']}, 0],",
            f"        dx={scr['dx']}, dz={scr['dz']},",
            f"        nx={scr['nx']}, nz={scr['nz']},",
            f"    )",
            "",
            "    # Run",
            "    beam = source.shine()",
        ])

        for i in range(len(self.oe_widgets)):
            lines.append(f"    oe{i + 1}.reflect(beam)")

        lines.extend([
            "    result = screen.expose(beam)",
            "",
            "    # Results",
            "    print(f'Good rays: {beam.good_count()}')",
            "    print(f'Captured:  {result.n_captured}')",
            "    print(f'FWHM:      {result.fwhm_x:.4f} x {result.fwhm_z:.4f} mm')",
            "    print(f'Centroid:  ({result.centroid_x:.4f}, {result.centroid_z:.4f})')",
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ])

        return "\n".join(lines)


# ─── Entry Point ────────────────────────────────────────────────────────────


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = BeamlineWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
