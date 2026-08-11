# Section 508 / WCAG 2.2 Compliance Report

**Project:** NASA EarthRISE Developers Academy — Great Basin Saline Lake Classifier  
**Branch:** `compliance-testing`  
**Test Date:** 2026-08-11  
**Standard:** Section 508 (Revised, 2018) / WCAG 2.1 + 2.2 Level A and AA  
**Testers:** Pa11y 9.1.1 (htmlcs + axe runners) + manual Qt UI audit

---

## 1. Scope

This project has two testable surfaces:

| Surface | Technology | Test Method |
|---|---|---|
| Jupyter Notebooks (HTML export) | HTML rendered by JupyterLab nbconvert | Pa11y 9.1.1 — htmlcs + axe, WCAG2AA |
| QGIS Plug-in UI (`Lake_Classifer_FD/dock.py`) | Qt 5 / PyQt (QDockWidget) | Manual audit against WCAG 2.1 AA + Section 508 §502 |

**Out of scope:** Python analysis scripts (CLI, no UI), Google Earth Engine JavaScript snippets (external platform), matplotlib time-series plots (rendered in a native window, not a web view).

---

## 2. Methodology

### 2.1 Automated Web Testing

Three notebooks were exported to HTML with `jupyter nbconvert --to html`, then each file was tested with:

```
npx pa11y --runner htmlcs --runner axe --standard WCAG2AA <file>
```

Both runners were combined to maximise coverage: htmlcs evaluates WCAG technical failure criteria; axe (Deque, v4.11) provides additional rule coverage including several WCAG 2.2 criteria.

### 2.2 Desktop Application Manual Audit

The QGIS plug-in cannot be tested by pa11y (no web rendering). It was manually audited against:

- **WCAG 2.1 / 2.2 Success Criteria** applicable to desktop software (1.1.1, 1.3.1, 1.3.3, 1.4.3, 2.1.1, 4.1.2, and applicable 2.2 additions 2.4.11, 2.5.7, 2.5.8)
- **Section 508 §502** (Interoperability with Assistive Technology)

Source file reviewed: [`Lake_Classifer_FD/dock.py`](Lake_Classifer_FD/dock.py)

---

## 3. Executive Summary

| Surface | Total Errors | Severity | Status |
|---|---|---|---|
| `threshold_grid_search_demo.html` | 1006 | ● Medium–High | **Does Not Conform** |
| `csda_Planet_fetcher_final.html` | 285 | ● Medium–High | **Does Not Conform** |
| `notebook.html` | 3 | ● Medium | **Does Not Conform** |
| QGIS Plug-in (`dock.py`) | 6 distinct issues | ● High | **Does Not Conform** |

**Overall conformance level:** Not conformant with WCAG 2.1 / 2.2 Level AA or Section 508.

The dominant failures across all surfaces are:

1. **Color contrast** — syntax-highlighting token colors in the Pygments/CodeMirror Jupyter theme fall below the 4.5:1 AA minimum for normal text. This accounts for ≥98% of automated errors.
2. **Missing accessible names** — interactive controls in the QGIS plugin have no programmatic accessible names, making them unusable with screen readers or other AT.
3. **Inadequate image alt text** — matplotlib plot outputs carry the Jupyter default placeholder text rather than a description of the chart.
4. **Missing `<iframe>` title** — one output frame in `threshold_grid_search_demo.html` has no `title` attribute.
5. **Broken table header associations** — two `<td>` elements in `threshold_grid_search_demo.html` lack proper `<th>` scope or `headers` associations.

---

## 4. Automated Test Results — Jupyter Notebook HTML Exports

### 4.1 `threshold_grid_search_demo.html`

**Pa11y summary: 1006 errors**

| Rule / Code | Runner | Count | WCAG SC | Description |
|---|---|---|---|---|
| `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Fail` | htmlcs | 752 | 1.4.3 Contrast (Minimum) | Syntax-highlighting tokens below 4.5:1 contrast ratio |
| `color-contrast` | axe | 249 | 1.4.3 Contrast (Minimum) | Same elements also flagged by axe (overlap with above) |
| `WCAG2AA.Principle1.Guideline1_3.1_3_1.H43,H63` | htmlcs | 2 | 1.3.1 Info and Relationships | `<td>` elements without defined `<th>` associations |
| `WCAG2AA.Principle2.Guideline2_4.2_4_1.H64.1` | htmlcs | 1 | 2.4.1 Bypass Blocks | `<iframe>` missing `title` attribute |
| `frame-title` | axe | 1 | 4.1.2 Name, Role, Value | Same `<iframe>` flagged by axe |
| `frame-tested` | axe | 1 | — | Axe requires iframe content be tested separately |

**Contrast detail — token types failing:**

| Pygments CSS class | Token type | Contrast ratio | Required | Status |
|---|---|---|---|---|
| `.c1` | Inline comment | 4.17:1 | 4.5:1 | ✗ Fail |
| `.mf`, `.mi` | Float / integer literal | 4.26:1 | 4.5:1 | ✗ Fail |
| `.o`, `.ow` | Operator / operator-word | 4.40:1 | 4.5:1 | ✗ Fail |

All three ratios are marginally below the threshold. This is a systemic issue with the default Pygments `default` or `friendly` style, not individual code errors.

**Image alt text:**

The three embedded matplotlib plots carry Jupyter's auto-generated alt text: `"No description has been provided for this image"`. This placeholder does not satisfy WCAG 1.1.1 (Non-text Content) for informational images.

---

### 4.2 `csda_Planet_fetcher_final.html`

**Pa11y summary: 285 errors**

| Rule / Code | Runner | Count | WCAG SC | Description |
|---|---|---|---|---|
| `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Fail` | htmlcs | 204 | 1.4.3 Contrast (Minimum) | Syntax-highlighting tokens below 4.5:1 |
| `color-contrast` | axe | 81 | 1.4.3 Contrast (Minimum) | Same elements flagged by axe |

All 285 errors are color contrast failures in code cells. No structural or semantic issues were detected beyond those of the Jupyter template itself.

---

### 4.3 `notebook.html`

**Pa11y summary: 3 errors**

| Rule / Code | Runner | Count | WCAG SC | Description |
|---|---|---|---|---|
| `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Fail` | htmlcs | 1 | 1.4.3 Contrast (Minimum) | Single code cell comment token |
| `color-contrast` | axe | 2 | 1.4.3 Contrast (Minimum) | Input prompt `In [ ]:` and comment token |

The `In [ ]:` input prompt label (color `#aaaaaa` on white) also fails the 4.5:1 threshold. This is the standard JupyterLab input-prompt style.

---

### 4.4 Passed Criteria (Automated)

The following WCAG criteria were verified as passing across all three notebook exports:

| Criterion | Verified |
|---|---|
| 3.1.1 Language of Page — `<html lang="en">` present | ✓ Pass |
| 2.4.2 Page Titled — `<title>` present | ✓ Pass |
| 1.3.1 Heading structure — sequential `h1`/`h2` present | ✓ Pass |
| 1.1.1 Images with empty alt for decorative images | ✓ Pass |
| No `tabindex > 0` overrides detected | ✓ Pass |
| No ARIA role misuse detected | ✓ Pass |

---

## 5. Desktop Application Audit — QGIS Plug-in (`dock.py`)

The QGIS plug-in renders a Qt `QDockWidget` panel. Automated HTML testing is not applicable; findings are based on static code review of [`Lake_Classifer_FD/dock.py`](Lake_Classifer_FD/dock.py).

### Issue 1 — No Accessible Names on Interactive Controls

**WCAG 4.1.2 (Name, Role, Value) / Section 508 §502.3.3**  
**Severity: High**

No `setAccessibleName()` calls exist anywhere in `dock.py`. Every interactive widget — `QComboBox` (lake selector, year selector), `QLineEdit` (Earth Engine project ID), `QPushButton` (five action buttons), `QSlider` (three threshold sliders), and `QDoubleSpinBox` (three threshold spinboxes) — lacks a programmatic accessible name.

AT such as NVDA, JAWS, and VoiceOver rely on `QAccessible::name()` to announce what a control does. Without explicit names, screen readers will fall back to the control's class name (e.g., "ComboBox") or remain silent, making the panel effectively unusable for blind or low-vision users.

**Relevant code:** [`dock.py:876–1150`](Lake_Classifer_FD/dock.py#L876) (`_build_ui` method) — no `setAccessibleName()` calls present.

**Fix:** Add `setAccessibleName()` to each interactive widget, for example:

```python
self.lake_combo.setAccessibleName("Select preset lake")
self.year_combo.setAccessibleName("Select year for summer composite")
self.project_edit.setAccessibleName("Google Earth Engine project ID")
```

For the `ThresholdControl` sliders, add inside `__init__` ([`dock.py:644`](Lake_Classifer_FD/dock.py#L644)):

```python
self._slider.setAccessibleName(title + " slider")
self._spin.setAccessibleName(title + " value")
```

---

### Issue 2 — "Reset" Buttons Have No Context

**WCAG 4.1.2 (Name, Role, Value) / Section 508 §502.3.3**  
**Severity: High**

The `ThresholdControl` widget creates a `QPushButton` labelled only `"reset"` ([`dock.py:675`](Lake_Classifer_FD/dock.py#L675)). Three such buttons exist on the panel (one per threshold). Without context, AT announces all three as "reset" with no indication of which threshold each resets.

**Fix:**

```python
reset_btn.setAccessibleName("Reset %s to default" % title)
```

---

### Issue 3 — Visible Labels Not Programmatically Linked to Controls

**WCAG 1.3.1 (Info and Relationships) / Section 508 §502.3.1**  
**Severity: High**

Caption `QLabel` widgets are placed visually above each control (e.g., "Earth Engine project ID:" above `self.project_edit`) but are not programmatically associated with the control. Qt's `QLabel::setBuddy()` establishes this relationship for AT. Without it, a screen reader encounters the input field without hearing its label.

**Relevant code:** [`dock.py:925–945`](Lake_Classifer_FD/dock.py#L925) (Earth Engine project ID section):

```python
v.addWidget(_caption("Earth Engine project ID:"))
self.project_edit = QLineEdit()
```

**Fix:** Assign buddies for each label/control pair:

```python
lbl = _caption("Earth Engine project ID:")
lbl.setBuddy(self.project_edit)
v.addWidget(lbl)
```

---

### Issue 4 — Hardcoded Text Colors Fail Contrast Against System Light Themes

**WCAG 1.4.3 (Contrast, Minimum) / Section 508 §502**  
**Severity: Medium**

Several `setStyleSheet` calls use light-blue/gray text colors that were chosen for a dark panel background. If a user runs QGIS with the default light system theme, these colors render light text on a light background.

| Widget | Color | Contrast vs. white | Required | Status |
|---|---|---|---|---|
| Caption labels (`_caption()`) | `#9fb3c1` | ~2.1:1 | 4.5:1 | ✗ Fail |
| Status labels (`vis_status`, `aoi_status`, `openet_status`) | `#8ea6b6` | ~2.4:1 | 4.5:1 | ✗ Fail |
| Area readout (`area_readout`) | `#cdd9e1` | ~1.5:1 | 4.5:1 | ✗ Fail |
| Reset link button | `#48c9d6` | ~3.5:1 | 4.5:1 | ✗ Fail (normal text) |

**Relevant code:** [`dock.py:223–229`](Lake_Classifer_FD/dock.py#L223), [`dock.py:676–678`](Lake_Classifer_FD/dock.py#L676), [`dock.py:1088–1093`](Lake_Classifer_FD/dock.py#L1088).

**Fix:** Either (a) explicitly set a dark background on the dock widget so the contrast pair is controlled, or (b) derive text colors from the Qt palette (`QPalette.ColorRole`) so they adapt to the system theme.

---

### Issue 5 — Sliders Lack Keyboard Value Readout

**WCAG 4.1.2 (Name, Role, Value) / Section 508 §502.3.14**  
**Severity: Medium**

`QSlider` does expose its value through `QAccessible`, but without a text description of the current value in context (e.g., "NDWI threshold: −0.055"), a non-visual user cannot confirm what the current threshold is without tabbing to the adjacent spin box. The spin box provides the numeric value, but the two are not grouped or labeled together.

**Fix:** Set `setAccessibleDescription()` on `committed` signal to reflect the current value:

```python
self._slider.setAccessibleDescription("Current value: %.3f" % self.value())
```

---

### Issue 6 — Matplotlib Time-Series Plot Is Image-Only Output

**WCAG 1.1.1 (Non-text Content) / Section 508 §502**  
**Severity: Medium**

`plot_timeseries()` ([`dock.py:2049`](Lake_Classifer_FD/dock.py#L2049)) opens a blocking `matplotlib` window. This chart has no text alternative — numeric values are annotated on the chart as floating text, but are not exposed to AT as structured data.

**Fix:** After `plt.show()`, optionally print or log a text summary of the series data, or offer CSV export (already present in Stage 4) as the accessible alternative. Document this in the UI.

---

## 6. WCAG 2.2-Specific Criteria Assessment

WCAG 2.2 introduces several new AA criteria. Their applicability to this project:

| WCAG 2.2 Criterion | Level | Applicable? | Status |
|---|---|---|---|
| 2.4.11 Focus Appearance | AA | Notebooks: N/A (static). QGIS plugin: partially applicable | Not evaluated (QGIS theme-dependent) |
| 2.4.12 Focus Appearance (Enhanced) | AAA | Out of scope (AAA) | — |
| 2.5.7 Dragging Movements | AA | QGIS sliders can be operated via keyboard (arrow keys) | ✓ Pass |
| 2.5.8 Target Size (Minimum) | AA | QGIS buttons use default Qt sizing; standard buttons exceed 24×24 CSS equivalent | ✓ Pass (native) |
| 3.2.6 Consistent Help | AA | No help mechanism present | Not applicable |
| 3.3.7 Redundant Entry | A | No multi-step form flows | Not applicable |
| 3.3.8 Accessible Authentication | AA | No authentication flow | Not applicable |

---

## 7. Remediation Priorities

### Immediate (blocks basic AT usability)

1. **Add `setAccessibleName()` to all interactive Qt widgets** — resolves Issues 1 and 2.
2. **Link caption labels to controls via `QLabel.setBuddy()`** — resolves Issue 3.

### Short-term (contrast compliance)

3. **Replace hardcoded hex colors with QPalette-derived values** in `dock.py` stylesheet calls — resolves Issue 4.
4. **Apply a WCAG-compliant Pygments syntax theme** to notebook exports (e.g., `--Highlight.style=sas` or `--Highlight.style=bw`) — resolves the 1,294 color-contrast errors across notebooks.

### Medium-term (completeness)

5. **Write descriptive alt text for matplotlib output images** in the three notebooks.
6. **Add `<iframe title>` to the Plotly/widget output iframe** in `threshold_grid_search_demo.ipynb`.
7. **Fix table header associations** (`scope` or `headers` attributes) in the two `<td>` elements flagged in `threshold_grid_search_demo.html`.
8. **Add a text alternative or data table** companion to the `plot_timeseries` matplotlib window.

---

## 8. Reproduction

```bash
# Re-run automated tests
pip install nbconvert
jupyter nbconvert --to html threshold_grid_search_demo.ipynb notebook.ipynb csda_Planet_fetcher_final.ipynb
npx pa11y --runner htmlcs --runner axe --standard WCAG2AA \
    "file://$(pwd)/threshold_grid_search_demo.html"
npx pa11y --runner htmlcs --runner axe --standard WCAG2AA \
    "file://$(pwd)/csda_Planet_fetcher_final.html"
npx pa11y --runner htmlcs --runner axe --standard WCAG2AA \
    "file://$(pwd)/notebook.html"
```

---

## 9. Conformance Statement

This application **does not conform** to WCAG 2.1 Level AA, WCAG 2.2 Level AA, or the Revised Section 508 Standards as of 2026-08-11.

The critical blocking issues are the absence of programmatic accessible names on all interactive QGIS plug-in controls (Issues 1–3), which prevents screen reader users from operating the primary deliverable. The color contrast failures in exported notebooks (Issues 4 in web surfaces) affect readability for low-vision users but do not prevent all access.

A conformant release requires, at minimum, resolving Issues 1, 2, and 3 in the QGIS plug-in and applying an accessible syntax highlighting theme to notebook exports.
