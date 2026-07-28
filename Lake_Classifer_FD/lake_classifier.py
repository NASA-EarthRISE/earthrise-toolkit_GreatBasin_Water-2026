"""Main QGIS plugin controller for the Saline Lake Classifier.

This file owns the plugin toolbar action, plugin-menu entry, dock-widget
lifecycle, and monthly time-series workflow.

The dock itself is implemented in dock.py. Earth Engine processing is handled
by ee_backend.py and imported only when a time-series request is made.
"""

import os
import re
import tempfile

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAction, QApplication

from .dock import LakeClassifierDock


class LakeClassifierPlugin:
    """Create, display, and coordinate the Saline Lake Classifier dock.

    One instance of this controller is created when QGIS loads the plugin.
    The toolbar action is created immediately, while the dock is built lazily
    the first time the user opens it.
    """

    MENU_NAME = "&Saline Lake Classifier"
    ACTION_NAME = "Saline Lake Classifier"

    def __init__(self, iface):
        """Store the QGIS interface and initialize plugin state.

        `iface` is the QGIS application interface supplied by QGIS when the
        plugin is loaded. The action and dock begin as None because they are
        created later by initGui and toggle, respectively.
        """
        self.iface = iface
        self.action = None
        self.dock = None

    def initGui(self):
        """Add the plugin action to the QGIS toolbar and plugin menu."""
        self.action = QAction(
            self.ACTION_NAME,
            self.iface.mainWindow(),
        )

        # The action remains checked while the dock is visible and unchecked
        # while it is hidden.
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle)

        self.iface.addToolBarIcon(self.action)

        self.iface.addPluginToMenu(
            self.MENU_NAME,
            self.action,
        )

    def unload(self):
        """Remove the dock and plugin actions when the plugin is unloaded.

        QGIS calls this method when the plugin is disabled, reloaded, or the
        application closes. Removing and deleting both Qt objects prevents
        stale toolbar entries and signal connections from remaining in QGIS.
        """
        if self.dock is not None:
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None

        if self.action is not None:
            self.iface.removeToolBarIcon(self.action)

            self.iface.removePluginMenu(
                self.MENU_NAME,
                self.action,
            )

            self.action.deleteLater()
            self.action = None

    def toggle(self, checked):
        """Create the dock on first use, then show or hide it.

        The same dock instance is reused for the remainder of the QGIS session,
        preserving the user's current AOI, thresholds, and fetched-layer state
        when the panel is temporarily hidden.
        """
        # Build the dock only the first time the plugin is opened. Reusing one
        # instance avoids duplicating widgets, state, and signal connections.
        if self.dock is None:
            self.dock = LakeClassifierDock(self.iface)

            # Section 4's time-series button is exposed by dock.py but its
            # workflow belongs to this controller, so the connection is made
            # here rather than inside the dock.
            self.dock.btn_timeseries.clicked.connect(
                self.run_timeseries
            )

            self.iface.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea,
                self.dock,
            )

            # Keep the toolbar action synchronized when the user closes or
            # reopens the dock using QGIS's own dock-widget controls.
            self.dock.visibilityChanged.connect(
                self.action.setChecked
            )

        self.dock.setVisible(checked)

        if checked:
            # Bring the panel in front of other dock widgets if it is already
            # open but obscured or grouped in a tabbed dock area.
            self.dock.raise_()

    def run_timeseries(self):
        """Generate, plot, and export the selected year's monthly indices.

        Validates the AOI, Earth Engine project ID, and year; requests monthly
        NDWI, NDVI, and NDMI statistics from ee_backend; sends the resulting
        table to the dock for plotting; and writes a CSV file to the operating
        system's temporary directory.
        """
        # The handler should only run while the dock exists, but this guard
        # also makes direct or delayed signal calls safe during plugin shutdown.
        if self.dock is None:
            return

        project = self.dock.project_edit.text().strip()
        extent = self.dock.aoi_extent

        # An AOI is required before Earth Engine can filter and summarize the
        # Landsat imagery.
        if extent is None:
            self._show_warning(
                "Choose an area of interest before downloading a time series."
            )
            return

        # Earth Engine initialization requires the user's cloud project ID.
        if not project:
            self._show_warning(
                "Enter your Earth Engine project ID before downloading "
                "a time series."
            )
            return

        try:
            year = int(
                self.dock.year_combo.currentText()
            )

        except (TypeError, ValueError):
            self._show_warning(
                "Select a valid year."
            )
            return

        # Update and repaint the status label before the synchronous Earth
        # Engine request blocks the QGIS interface.
        self.dock.csv_label.setText(
            "Computing monthly time series…"
        )
        QApplication.processEvents()

        try:
            # Imported lazily so QGIS can still load the plugin and display a
            # useful setup message if earthengine-api is unavailable.
            from . import ee_backend

            ee = ee_backend.initialize(project)

            df = ee_backend.compute_monthly_timeseries(
                ee,
                extent,
                year,
            )

            # An empty table cannot produce a useful plot or CSV and usually
            # means no valid Landsat observations were available for the AOI.
            if df is None or df.empty:
                raise ValueError(
                    "Earth Engine returned no monthly observations "
                    "for the selected AOI and year."
                )

            self.dock.plot_timeseries(
                df,
                year,
            )

            # Project IDs may contain spaces or punctuation that are unsuitable
            # for filenames. Replace those characters while retaining letters,
            # numbers, periods, underscores, and hyphens.
            safe_project = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                project,
            ).strip("_") or "earth_engine"

            # Save the generated table in the operating system's temporary
            # directory. The full path is shown in the dock so the user can
            # locate and move the file if a permanent copy is needed.
            csv_path = os.path.join(
                tempfile.gettempdir(),
                f"{safe_project}_timeseries_{year}.csv",
            )

            df.to_csv(
                csv_path,
                index=False,
            )

            self.dock.csv_label.setText(
                "CSV saved to: %s" % csv_path
            )

            self.iface.messageBar().pushSuccess(
                self.ACTION_NAME,
                "Monthly time series created and saved successfully.",
            )

        except ImportError:
            # Treat a missing Earth Engine package as a setup problem and give
            # the user a specific installation path rather than a generic error.
            message = (
                "The Earth Engine Python API is not available in QGIS. "
                "Install or enable the Google Earth Engine plugin and "
                "try again."
            )

            self.dock.csv_label.setText(
                "Time-series error: %s" % message
            )

            self._show_warning(message)

        except Exception as exc:
            # Earth Engine, plotting, and file-writing errors all reach this
            # shared path so the dock label and QGIS message bar remain
            # consistent.
            self.dock.csv_label.setText(
                "Time-series error: %s" % exc
            )

            self._show_warning(
                str(exc)
            )

    def _show_warning(self, message):
        """Show a consistent time-series warning in the QGIS message bar."""
        self.iface.messageBar().pushWarning(
            "Time Series Error",
            message,
        )