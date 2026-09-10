"""Driver Demand Table Validator - GUI for validating Ford Coyote driver
demand torque tables.

Features:
  * Paste-in or load driver demand tables (Normal, Sport, Fault/FMEM, OSS).
  * Paste-in or load the indicated torque table (for max-torque comparison).
  * Run validation checks: max torque, monotonicity, smoothness, idle, coverage.
  * Color-coded issue highlighting on the table grid.
  * Cross-table comparison (Normal vs Sport vs Fault).
  * Export validated tables to CSV.

Run:
    python driver_demand_tool.py
"""
from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

# ensure local imports work when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

from table_model import DriverDemandTable, IndicatedTorqueTable, _fmt
from validator import (
    ValidationConfig, ValidationResult, Severity,
    validate_demand_table, compare_tables, format_report,
    check_oss_modifier, validate_with_oss, validate_vs_logged_max,
)
from log_loader import load_csv, load_hpl, extract_wot_max, WotMaxCurve

# auto-save file: stored next to the script
SAVE_FILE = Path(__file__).resolve().parent / "driver_demand_session.json"

# Default paste-in tables (user's 2014 Mustang GT data)
DEFAULT_DEMAND_NORMAL = """N m\t600\t1150\t1500\t1950\t2500\t3200\t3900\t4600\t5300\t6850\trpm
15\t30\t-1\t-6\t-11\t-16\t-22\t-30\t-35\t-43\t-55
25\t55\t38\t25\t20\t12\t0\t-10\t-17\t-27\t-40
56\t100\t100\t81\t66\t55\t39\t24\t11\t2\t-10
83\t140\t140\t131\t107\t86\t65\t50\t38\t27\t12
112\t190\t190\t190\t158\t127\t101\t86\t74\t62\t44
143\t242\t242\t242\t218\t184\t153\t133\t120\t110\t92
176\t293\t293\t293\t277\t240\t204\t179\t165\t157\t140
242\t330\t330\t330\t330\t330\t305\t252\t232\t220\t205
311\t350\t350\t350\t370\t415\t420\t420\t400\t350\t330
542\t350\t350\t351\t370\t418\t447\t485\t485\t480\t380"""

DEFAULT_INDICATED = """N m\t550\t1000\t1500\t2000\t2500\t3000\t4000\t5000\t6000\t7000\trpm
0.10000000149011612\t40\t40.5\t41.5\t34\t38\t40\t45\t53\t62.03020095825195\t49.13800048828125
0.20000000298023224\t96.5\t101\t102.5\t104\t109\t111\t114\t118.77749633789062\t129.4340057373047\t111.95649719238281
0.30000001192092896\t150.71189880371094\t156.6678009033203\t161.73440551757812\t167.24349975585938\t172.97230529785156\t176.13470458984375\t179.3946075439453\t182.06619262695312\t190.97509765625\t180.85079956054688
0.4000000059604645\t206.52659606933594\t214.8092041015625\t221.11099243164062\t226.9976043701172\t232.8690948486328\t235.7480926513672\t239.61090087890625\t240.74839782714844\t248.25430297851562\t247.65480041503906
0.5\t260.16448974609375\t270.61749267578125\t279.2748107910156\t286.3434143066406\t292.5411071777344\t294.7618103027344\t299.42620849609375\t296.62939453125\t298.2742919921875\t306.5237121582031
0.6000000238418579\t319.2395935058594\t328.1034851074219\t337.1288146972656\t345.04779052734375\t351.934814453125\t353.80828857421875\t359.6001892089844\t354.4302062988281\t351.7149963378906\t361.2604064941406
0.699999988079071\t380.2016906738281\t388.3760070800781\t397.3411865234375\t404.4299011230469\t411.3396911621094\t412.9538879394531\t419.2421875\t415.2908935546875\t411.4541931152344\t417.5459899902344
0.800000011920929\t441.0647888183594\t448.50408935546875\t456.5262145996094\t463.46728515625\t468.9988098144531\t472.322509765625\t477.3140869140625\t475.3841857910156\t472.62139892578125\t475.9097900390625
0.8999999761581421\t501.3370056152344\t508.00518798828125\t515.181396484375\t521.5422973632812\t526.68408203125\t530.498779296875\t535.0869750976562\t534.7656860351562\t533.393798828125\t535.28271484375
1.2000000476837158\t679.8170776367188\t684.8408813476562\t690.2097778320312\t695.1533813476562\t699.5059814453125\t703.1658935546875\t708.3110961914062\t710.9860229492188\t712.4697265625\t713.9534912109375"""

# Default OSS modifier table (multiplier on driver demand torque)
# Rows = pedal ADC, Cols = OSS RPM
DEFAULT_OSS_MODIFIER = """mult\t78\t196\t392\t588\t784\t1176\t1568\t1960\t2549\t2941\tOSS RPM
15\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1
25\t1\t1\t1\t1\t1.14999997615814\t1.20000004768372\t1.20000004768372\t1.20000004768372\t1.20000004768372\t1.20000004768372
56\t0.899999976158142\t1\t1.14999997615814\t1.20000004768372\t1.25\t1.35000002384186\t1.35000002384186\t1.35000002384186\t1.39999997615814\t1.39999997615814
83\t0.899999976158142\t1\t1.14999997615814\t1.20000004768372\t1.25\t1.35000002384186\t1.35000002384186\t1.35000002384186\t1.39999997615814\t1.39999997615814
112\t0.899999976158142\t1.02499997615814\t1.10000002384186\t1.14999997615814\t1.25\t1.35000002384186\t1.35000002384186\t1.35000002384186\t1.29999995231628\t1.29999995231628
143\t0.899999976158142\t1.04999995231628\t1.14999997615814\t1.14999997615814\t1.25\t1.29999995231628\t1.25\t1.25\t1.14999997615814\t1.14999997615814
176\t0.899999976158142\t1.04999995231628\t1.14999997615814\t1.14999997615814\t1.20000004768372\t1.20000004768372\t1.04999995231628\t1.04999995231628\t0.899999976158142\t0.899999976158142
242\t0.899999976158142\t1.04999995231628\t1.14999997615814\t1.10000002384186\t1.10000002384186\t0.899999976158142\t0.649999976158142\t0.649999976158142\t0.649999976158142\t0.649999976158142
311\t1\t1\t1\t0.699999988079071\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5
542\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1"""


def _axis_fmt_func(label: str):
    """Return a formatter callable for an axis based on its label."""
    lbl = label.lower()
    if "rpm" in lbl or "torque" in lbl or "n m" in lbl or "pedal" in lbl:
        return _fmt_int
    return _fmt


def _fmt_int(v: float) -> str:
    return f"{int(round(v))}"


class TableGrid(ttk.Frame):
    """An editable 2D table grid with individual cells (tk.Entry widgets),
    per-cell color highlighting, and click-drag multi-cell selection."""

    # Dark theme colors — match the main app's dark palette.
    _BG_HEADER = "#333333"
    _BG_GRID = "#2a2a2a"
    _BG_CELL = "#2d2d2d"
    _FG = "#e0e0e0"
    _SELECT = "#0a4a6e"

    def __init__(self, parent, table, title: str,
                 highlight: Optional[np.ndarray] = None,
                 highlight_label: str = "",
                 editable: bool = True,
                 on_edit=None):
        super().__init__(parent)
        self.table = table
        self.title = title
        self.highlight = highlight
        self.highlight_label = highlight_label
        self.editable = editable
        self.on_edit = on_edit
        self._cells: dict[tuple[int, int], tk.Entry] = {}
        self._row_axis_entries: list[tk.Entry] = []
        self._col_axis_entries: list[tk.Entry] = []
        self._cell_bgs: dict[tuple[int, int], str] = {}
        self._sel_start: Optional[tuple[int, int]] = None
        self._sel_end: Optional[tuple[int, int]] = None
        self._selected_cells: set[tuple[int, int]] = set()
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text=self.title, font=("Segoe UI", 10, "bold")).pack(side="left")
        if self.highlight_label:
            ttk.Label(top, text=self.highlight_label,
                      foreground="#888888").pack(side="left", padx=8)
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0, bg=self._BG_GRID)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        hsb = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        grid_frame = tk.Frame(canvas, bg=self._BG_GRID)
        canvas.create_window(0, 0, anchor="nw", window=grid_frame)
        def _config(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        grid_frame.bind("<Configure>", _config)

        nr, nc = self.table.n_rows, self.table.n_cols
        row_fmt = _axis_fmt_func(self.table.row_label)
        col_fmt = _axis_fmt_func(self.table.col_label)
        scale = (float(np.nanmax(np.abs(self.highlight))) or 1.0
                 if self.highlight is not None else 1.0)
        tk.Label(grid_frame, text=self.table.row_label,
                 relief="ridge", padx=4, pady=2, bg=self._BG_HEADER, fg=self._FG,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="nsew")
        self._col_axis_entries = []
        for c in range(nc):
            if self.editable:
                e = tk.Entry(grid_frame, width=8, justify="right",
                             relief="solid", borderwidth=1, font=("Consolas", 9),
                             bg=self._BG_CELL, readonlybackground=self._BG_CELL,
                             fg=self._FG, insertbackground=self._FG,
                             disabledbackground=self._BG_CELL,
                             disabledforeground=self._FG)
                e.insert(0, col_fmt(self.table.col_axis[c]))
                e.grid(row=0, column=c + 1, sticky="nsew", padx=1, pady=1)
                e.bind("<FocusOut>",
                       lambda _ev, cc=c: self._on_col_axis_edit(cc))
                self._col_axis_entries.append(e)
            else:
                tk.Label(grid_frame, text=col_fmt(self.table.col_axis[c]),
                         relief="ridge", padx=4, pady=2, bg=self._BG_HEADER, fg=self._FG,
                         font=("Consolas", 9)).grid(row=0, column=c + 1, sticky="nsew")
        tk.Label(grid_frame, text=self.table.col_label,
                 relief="ridge", padx=4, pady=2, bg=self._BG_HEADER, fg=self._FG,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=nc + 1, sticky="nsew")
        self._row_axis_entries = []
        for r in range(nr):
            if self.editable:
                e = tk.Entry(grid_frame, width=8, justify="right",
                             relief="solid", borderwidth=1, font=("Consolas", 9),
                             bg=self._BG_CELL, readonlybackground=self._BG_CELL,
                             fg=self._FG, insertbackground=self._FG,
                             disabledbackground=self._BG_CELL,
                             disabledforeground=self._FG)
                e.insert(0, row_fmt(self.table.row_axis[r]))
                e.grid(row=r + 1, column=0, sticky="nsew", padx=1, pady=1)
                e.bind("<FocusOut>",
                       lambda _ev, rr=r: self._on_row_axis_edit(rr))
                self._row_axis_entries.append(e)
            else:
                tk.Label(grid_frame, text=row_fmt(self.table.row_axis[r]),
                         relief="ridge", padx=4, pady=2, bg=self._BG_HEADER, fg=self._FG,
                         font=("Consolas", 9)).grid(row=r + 1, column=0, sticky="nsew")
            for c in range(nc):
                val = self.table.values[r, c]
                state = "normal" if self.editable else "readonly"
                e = tk.Entry(grid_frame, width=10, justify="right",
                             relief="solid", borderwidth=1, font=("Consolas", 9),
                             bg=self._BG_CELL, readonlybackground=self._BG_CELL,
                             fg=self._FG, insertbackground=self._FG,
                             disabledbackground=self._BG_CELL,
                             disabledforeground=self._FG)
                e.insert(0, f"{val:.4f}")
                e.configure(state=state)
                e.grid(row=r + 1, column=c + 1, sticky="nsew", padx=1, pady=1)
                if self.editable and self.on_edit:
                    e.bind("<FocusOut>",
                           lambda _ev, rr=r, cc=c: self._on_edit(rr, cc))
                if self.highlight is not None:
                    color = self._heat_color(self.highlight[r, c], scale)
                    e.configure(background=color, readonlybackground=color)
                    self._cell_bgs[(r, c)] = color
                else:
                    e.configure(readonlybackground=self._BG_CELL)
                    self._cell_bgs[(r, c)] = self._BG_CELL
                e.bind("<Button-1>", lambda ev, rr=r, cc=c: self._on_cell_click(ev, rr, cc))
                e.bind("<B1-Motion>", self._on_cell_drag)
                e.bind("<Shift-Button-1>", lambda ev, rr=r, cc=c: self._on_shift_click(ev, rr, cc))
                if self.editable:
                    e.bind("<Control-a>", lambda _ev: self._select_all())
                    e.bind("<Control-c>", lambda _ev: self._copy_selection())
                    e.bind("<Control-v>", lambda _ev: self._paste_to_selection())
                self._cells[(r, c)] = e
        tk.Label(grid_frame, text=self.table.row_label,
                 relief="ridge", padx=4, pady=2, bg=self._BG_HEADER, fg=self._FG,
                 font=("Segoe UI", 9, "bold")).grid(row=nr + 1, column=0, sticky="nsew")

    # ---- multi-cell selection ------------------------------------------

    def _cell_at_pos(self, event) -> Optional[tuple[int, int]]:
        w = event.widget.winfo_containing(event.x_root, event.y_root)
        for (r, c), e in self._cells.items():
            if e is w:
                return (r, c)
        return None

    def _on_cell_click(self, _event, r, c):
        self._clear_selection()
        self._sel_start = (r, c)
        self._sel_end = (r, c)
        self._update_selection()

    def _on_shift_click(self, _event, r, c):
        if self._sel_start is None:
            self._sel_start = (r, c)
        self._sel_end = (r, c)
        self._update_selection()

    def _on_cell_drag(self, event):
        pos = self._cell_at_pos(event)
        if pos is not None:
            self._sel_end = pos
            self._update_selection()

    def _update_selection(self):
        if self._sel_start is None or self._sel_end is None:
            return
        r0, c0 = self._sel_start
        r1, c1 = self._sel_end
        rmin, rmax = min(r0, r1), max(r0, r1)
        cmin, cmax = min(c0, c1), max(c0, c1)
        self._clear_selection_highlight()
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                e = self._cells.get((r, c))
                if e:
                    e.configure(background=self._SELECT,
                                readonlybackground=self._SELECT)
                    self._selected_cells.add((r, c))

    def _clear_selection_highlight(self):
        for (r, c) in self._selected_cells:
            e = self._cells.get((r, c))
            if e:
                bg = self._cell_bgs.get((r, c), self._BG_CELL)
                e.configure(background=bg, readonlybackground=bg)
        self._selected_cells.clear()

    def _clear_selection(self):
        self._clear_selection_highlight()
        self._sel_start = None
        self._sel_end = None

    def _select_all(self):
        self._clear_selection()
        self._sel_start = (0, 0)
        self._sel_end = (self.table.n_rows - 1, self.table.n_cols - 1)
        self._update_selection()
        return "break"

    def _copy_selection(self):
        if not self._selected_cells:
            self.copy_to_clipboard()
            return "break"
        if self._sel_start is None or self._sel_end is None:
            return "break"
        r0, c0 = self._sel_start
        r1, c1 = self._sel_end
        rmin, rmax = min(r0, r1), max(r0, r1)
        cmin, cmax = min(c0, c1), max(c0, c1)
        lines = []
        for r in range(rmin, rmax + 1):
            vals = []
            for c in range(cmin, cmax + 1):
                e = self._cells.get((r, c))
                vals.append(e.get() if e else "")
            lines.append("\t".join(vals))
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        return "break"

    def _paste_to_selection(self):
        try:
            raw = self.clipboard_get()
        except tk.TclError:
            return "break"
        if not raw or not raw.strip():
            return "break"
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        delim = "\t" if "\t" in lines[0] else ","
        grid = [[c.strip() for c in ln.split(delim)] for ln in lines]
        if self._sel_start is not None:
            r_start, c_start = self._sel_start
        else:
            r_start, c_start = 0, 0
        nr, nc = self.table.n_rows, self.table.n_cols
        for ri, row_vals in enumerate(grid):
            r = r_start + ri
            if r >= nr:
                break
            for ci, val in enumerate(row_vals):
                c = c_start + ci
                if c >= nc:
                    break
                try:
                    v = float(val)
                except ValueError:
                    continue
                self.table.values[r, c] = v
                e = self._cells.get((r, c))
                if e:
                    state = e.cget("state")
                    e.configure(state="normal")
                    e.delete(0, "end")
                    e.insert(0, f"{v:.4f}")
                    e.configure(state=state)
        if self.on_edit:
            self.on_edit()
        return "break"

    # ---- cell editing ---------------------------------------------------

    def _on_edit(self, r, c):
        e = self._cells[(r, c)]
        try:
            v = float(e.get())
        except ValueError:
            e.delete(0, "end")
            e.insert(0, f"{self.table.values[r, c]:.4f}")
            return
        self.table.values[r, c] = v
        if self.on_edit:
            self.on_edit()

    def _on_row_axis_edit(self, r):
        e = self._row_axis_entries[r]
        try:
            v = float(e.get())
        except ValueError:
            e.delete(0, "end")
            e.insert(0, _axis_fmt_func(self.table.row_label)(self.table.row_axis[r]))
            return
        self.table.row_axis[r] = v
        if self.on_edit:
            self.on_edit()

    def _on_col_axis_edit(self, c):
        e = self._col_axis_entries[c]
        try:
            v = float(e.get())
        except ValueError:
            e.delete(0, "end")
            e.insert(0, _axis_fmt_func(self.table.col_label)(self.table.col_axis[c]))
            return
        self.table.col_axis[c] = v
        if self.on_edit:
            self.on_edit()

    # ---- clipboard buttons ---------------------------------------------

    def paste_from_clipboard(self) -> bool:
        try:
            raw = self.clipboard_get()
        except tk.TclError:
            return False
        if not raw or not raw.strip():
            return False
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return False
        delim = "\t" if "\t" in lines[0] else ","
        grid = [[c.strip() for c in ln.split(delim)] for ln in lines]
        n_rows_in = len(grid)
        n_cols_in = max(len(row) for row in grid)
        nr, nc = self.table.n_rows, self.table.n_cols
        has_header = n_rows_in == nr + 1 and n_cols_in >= nc + 1
        has_row_labels = has_header or (n_rows_in == nr and n_cols_in == nc + 1)
        val_r0 = 1 if has_header else 0
        val_c0 = 1 if has_row_labels else 0
        row_fmt = _axis_fmt_func(self.table.row_label)
        col_fmt = _axis_fmt_func(self.table.col_label)
        for r in range(nr):
            sr = r + val_r0
            if sr >= n_rows_in:
                break
            if has_row_labels and val_c0 > 0:
                try:
                    rv = float(grid[sr][0])
                    self.table.row_axis[r] = rv
                    if r < len(self._row_axis_entries):
                        e = self._row_axis_entries[r]
                        e.delete(0, "end")
                        e.insert(0, row_fmt(rv))
                except (ValueError, IndexError):
                    pass
            for c in range(nc):
                sc = c + val_c0
                if sc >= len(grid[sr]):
                    break
                try:
                    v = float(grid[sr][sc])
                except ValueError:
                    continue
                self.table.values[r, c] = v
                e = self._cells.get((r, c))
                if e:
                    state = e.cget("state")
                    e.configure(state="normal")
                    e.delete(0, "end")
                    e.insert(0, f"{v:.4f}")
                    e.configure(state=state)
        if has_header:
            for c in range(nc):
                sc = c + (1 if has_row_labels else 0)
                if sc < len(grid[0]):
                    try:
                        cv = float(grid[0][sc])
                        self.table.col_axis[c] = cv
                        if c < len(self._col_axis_entries):
                            e = self._col_axis_entries[c]
                            e.delete(0, "end")
                            e.insert(0, col_fmt(cv))
                    except (ValueError, IndexError):
                        pass
        if self.on_edit:
            self.on_edit()
        return True

    def copy_to_clipboard(self):
        row_fmt = _axis_fmt_func(self.table.row_label)
        col_fmt = _axis_fmt_func(self.table.col_label)
        lines = []
        header = [""] + [col_fmt(c) for c in self.table.col_axis] + [self.table.col_label]
        lines.append("\t".join(header))
        for r in range(self.table.n_rows):
            row = [row_fmt(self.table.row_axis[r])]
            row += [f"{self.table.values[r, c]:.4f}" for c in range(self.table.n_cols)]
            lines.append("\t".join(row))
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))

    @staticmethod
    def _heat_color(v: float, scale: float = 1.0) -> str:
        """Map a signed delta to green(positive)/red(negative) gradient.
        Zero values return the dark cell background."""
        if v == 0 or np.isnan(v):
            return "#2d2d2d"  # dark cell background
        mag = min(abs(v) / scale, 1.0) if scale > 0 else min(abs(v), 1.0)
        if v > 0:
            r = int(255 - 180 * mag)
            b = int(255 - 180 * mag)
            return f"#{r:02x}ff{b:02x}"
        g = int(255 - 180 * mag)
        b = int(255 - 180 * mag)
        return f"#ff{g:02x}{b:02x}"  # red gradient (issue)

    def refresh(self, table=None, highlight: Optional[np.ndarray] = None):
        if table is not None:
            self.table = table
        if highlight is not None:
            self.highlight = highlight
        for (r, c), e in self._cells.items():
            val = self.table.values[r, c]
            state = e.cget("state")
            e.configure(state="normal")
            e.delete(0, "end")
            e.insert(0, f"{val:.4f}")
            if self.highlight is not None:
                color = self._heat_color(self.highlight[r, c], 1.0)
                e.configure(background=color, readonlybackground=color)
                self._cell_bgs[(r, c)] = color
            else:
                e.configure(background=self._BG_CELL,
                            readonlybackground=self._BG_CELL)
                self._cell_bgs[(r, c)] = self._BG_CELL
            e.configure(state=state)


class DriverDemandApp(tk.Tk):
    """Main application window for the Driver Demand Table Validator."""

    # Dark theme palette — matches the Torque Tuning Tool's dark appearance.
    _BG = "#1e1e1e"          # main window background
    _BG_FRAME = "#252525"    # frame / panel background
    _BG_ENTRY = "#2d2d2d"    # entry / cell background
    _BG_HEADER = "#333333"   # axis header background
    _FG = "#e0e0e0"          # primary text
    _FG_DIM = "#888888"      # dimmed text
    _SELECT = "#0a4a6e"      # selection highlight
    _GRID_LINE = "#3c3c3c"   # grid frame background

    def __init__(self):
        super().__init__()
        self.title("Coyote Driver Demand Table Validator")
        self.geometry("1280x820")
        self.configure(background=self._BG)
        self._setup_dark_theme()
        # tables
        self.demand_tables: dict[str, DriverDemandTable] = {}
        self.indicated: Optional[IndicatedTorqueTable] = None
        self.wot_curve: Optional[WotMaxCurve] = None
        self.cfg = ValidationConfig()
        self.current_table_name: str = "Normal"
        self._results: dict[str, ValidationResult] = {}
        # UI refs
        self.table_var: tk.StringVar
        self.table_combo: ttk.Combobox
        self.demand_grid_frame: ttk.Frame
        self.indicated_grid_frame: ttk.Frame
        self.demand_grid: Optional[TableGrid] = None
        self.indicated_grid: Optional[TableGrid] = None
        self.plot_frame: ttk.Frame
        self._cfg_vars: dict[str, tk.StringVar] = {}
        self.report_text: tk.Text
        self._build_ui()
        self._load_defaults()
        self._load_session()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_dark_theme(self):
        """Match the Torque Tuning Tool's dark appearance.

        The Torque Tuning Tool uses the default 'vista' ttk theme, which
        picks up Windows dark mode automatically. We do the same here —
        no ttk theme override. We only set the tk.* widget colors that
        don't follow ttk styles (tk.Frame, tk.Label, tk.Entry, tk.Text,
        tk.Canvas) to dark backgrounds with white text.
        """
        # Set the root window bg to dark
        self.configure(background=self._BG)

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.tab_tables = ttk.Frame(self.notebook)
        self.tab_report = ttk.Frame(self.notebook)
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tables, text="Tables")
        self.notebook.add(self.tab_report, text="Validation Report")
        self.notebook.add(self.tab_plot, text="Plots")
        self._build_tables_tab()
        self._build_report_tab()
        self._build_plot_tab()
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, relief="sunken",
                  anchor="w").pack(fill="x", side="bottom")

    def _build_tables_tab(self):
        tab = self.tab_tables
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Paste Demand from Clipboard",
                   command=self._paste_demand).pack(side="left")
        ttk.Button(toolbar, text="Paste Indicated from Clipboard",
                   command=self._paste_indicated).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(toolbar, text="Load WOT Log",
                   command=self._load_wot_log).pack(side="left")
        ttk.Button(toolbar, text="Copy Demand to Clipboard",
                   command=self._copy_demand).pack(side="left")
        ttk.Button(toolbar, text="Copy Indicated to Clipboard",
                   command=self._copy_indicated).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(toolbar, text="Run Validation",
                   command=self._run_validation).pack(side="left")
        # table selector
        ttk.Label(toolbar, text="Demand Table:").pack(side="left", padx=(8, 2))
        self.table_var = tk.StringVar(value="Normal")
        self.table_combo = ttk.Combobox(toolbar, textvariable=self.table_var,
                                        values=["Normal", "Sport", "Fault", "OSS"],
                                        width=10, state="readonly")
        self.table_combo.pack(side="left")
        self.table_combo.bind("<<ComboboxSelected>>", lambda _ev: self._on_table_changed())

        paned = ttk.PanedWindow(tab, orient="vertical")
        paned.pack(fill="both", expand=True, pady=4)
        self.demand_grid_frame = ttk.Frame(paned)
        self.indicated_grid_frame = ttk.Frame(paned)
        paned.add(self.demand_grid_frame, weight=3)
        paned.add(self.indicated_grid_frame, weight=2)

    def _build_report_tab(self):
        tab = self.tab_report
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Run Validation",
                   command=self._run_validation).pack(side="left")
        ttk.Button(toolbar, text="Clear Report",
                   command=self._clear_report).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)
        # config settings
        cfgf = ttk.LabelFrame(toolbar, text="Settings")
        cfgf.pack(side="left", fill="x", padx=4)
        for key, label, default in [
            ("max_torque_margin", "Max torque margin (N m)", str(self.cfg.max_torque_margin)),
            ("monotonicity_tolerance", "Monotonicity tol (N m)", str(self.cfg.monotonicity_tolerance)),
            ("gradient_change_threshold", "Gradient change threshold", str(self.cfg.gradient_change_threshold)),
            ("idle_max_torque", "Idle max torque (N m)", str(self.cfg.idle_max_torque)),
            ("idle_max_rpm", "Idle max RPM", str(self.cfg.idle_max_rpm)),
        ]:
            var = tk.StringVar(value=default)
            self._cfg_vars[key] = var
            ttk.Label(cfgf, text=label).pack(side="left", padx=2)
            ttk.Entry(cfgf, textvariable=var, width=8).pack(side="left", padx=2)

        self.report_text = tk.Text(tab, wrap="word", font=("Consolas", 10),
                                   bg="#1e1e1e", fg="#e0e0e0",
                                   insertbackground="#e0e0e0",
                                   selectbackground="#0a4a6e")
        self.report_text.pack(fill="both", expand=True, pady=4)

    def _build_plot_tab(self):
        tab = self.tab_plot
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Plot Demand Surface",
                   command=self._plot_surface).pack(side="left")
        ttk.Button(toolbar, text="Plot Demand vs Indicated Max",
                   command=self._plot_max_comparison).pack(side="left")
        ttk.Button(toolbar, text="Plot Gradient Heatmap",
                   command=self._plot_gradient).pack(side="left")
        self.plot_frame = ttk.Frame(tab)
        self.plot_frame.pack(fill="both", expand=True, pady=4)

    # ---- defaults / session --------------------------------------------

    def _load_defaults(self):
        try:
            demand = DriverDemandTable.from_grid_text(
                DEFAULT_DEMAND_NORMAL, table_name="Normal",
                row_label="Pedal", col_label="RPM", val_label="N m")
            self.demand_tables["Normal"] = demand
            # clone Normal as the starting point for Sport and Fault so
            # they are editable immediately rather than blank
            for mode in ("Sport", "Fault"):
                clone = demand.copy()
                clone.table_name = mode
                self.demand_tables[mode] = clone
            # OSS modifier table has different axes (pedal ADC x OSS RPM)
            # and stores multipliers, not torque
            oss = DriverDemandTable.from_grid_text(
                DEFAULT_OSS_MODIFIER, table_name="OSS",
                row_label="Pedal", col_label="OSS RPM", val_label="mult")
            self.demand_tables["OSS"] = oss
        except (ValueError, IndexError) as e:
            messagebox.showerror("Default load error", f"Could not load default demand table:\n{e}")
        try:
            self.indicated = IndicatedTorqueTable.from_grid_text(DEFAULT_INDICATED)
        except (ValueError, IndexError) as e:
            messagebox.showerror("Default load error", f"Could not load default indicated table:\n{e}")
        self._render_tables()

    def _load_session(self):
        if not SAVE_FILE.exists():
            return
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # restore config
            for key, var in self._cfg_vars.items():
                if key in data.get("config", {}):
                    var.set(str(data["config"][key]))
            self._apply_config()
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def _save_session(self):
        data = {"config": {k: v.get() for k, v in self._cfg_vars.items()}}
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _on_close(self):
        self._save_session()
        self.destroy()

    # ---- config ---------------------------------------------------------

    def _apply_config(self):
        try:
            for key, var in self._cfg_vars.items():
                val = float(var.get())
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, val)
        except (ValueError, TypeError):
            pass

    # ---- table rendering ------------------------------------------------

    def _render_tables(self, demand_highlight: Optional[np.ndarray] = None,
                       demand_highlight_label: str = "",
                       ind_highlight: Optional[np.ndarray] = None):
        # demand table
        for child in self.demand_grid_frame.winfo_children():
            child.destroy()
        demand = self.demand_tables.get(self.current_table_name)
        if demand is not None:
            self.demand_grid = TableGrid(
                self.demand_grid_frame, demand,
                f"Driver Demand — {self.current_table_name}",
                highlight=demand_highlight,
                highlight_label=demand_highlight_label,
                on_edit=self._on_demand_edit,
            )
            self.demand_grid.pack(fill="both", expand=True)
        # indicated table
        for child in self.indicated_grid_frame.winfo_children():
            child.destroy()
        if self.indicated is not None:
            self.indicated_grid = TableGrid(
                self.indicated_grid_frame, self.indicated,
                "Indicated Torque (max reference)",
                highlight=ind_highlight,
                editable=True,
                on_edit=self._on_indicated_edit,
            )
            self.indicated_grid.pack(fill="both", expand=True)

    def _on_demand_edit(self):
        pass  # table is edited in place; no action needed

    def _on_indicated_edit(self):
        pass  # table is edited in place; no action needed

    def _on_table_changed(self):
        self.current_table_name = self.table_var.get()
        self._render_tables()

    # ---- clipboard ------------------------------------------------------

    def _paste_demand(self):
        if self.demand_grid is None:
            return
        if self.demand_grid.paste_from_clipboard():
            self.status.set(f"Pasted {self.current_table_name} demand table from clipboard.")
        else:
            self.status.set("Clipboard is empty or invalid.")

    def _paste_indicated(self):
        if self.indicated_grid is None:
            return
        if self.indicated_grid.paste_from_clipboard():
            self.status.set("Pasted indicated torque table from clipboard.")
        else:
            self.status.set("Clipboard is empty or invalid.")

    def _load_wot_log(self):
        """Open a file dialog, load a WOT log (CSV or HPL), and extract
        the engine's max torque curve."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Load WOT Log",
            filetypes=[("Log files", "*.csv *.hpl"), ("All files", "*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(".hpl"):
                log = load_hpl(path)
            else:
                log = load_csv(path)
            self.wot_curve = extract_wot_max(log)
            self.status.set(
                f"Loaded WOT log: {len(self.wot_curve.rpm_bins)} RPM bins, "
                f"{self.wot_curve.source}")
        except (ValueError, OSError, KeyError, IndexError) as e:
            self.wot_curve = None
            messagebox.showerror("WOT log error", str(e))
            self.status.set("WOT log load failed.")

    def _copy_demand(self):
        if self.demand_grid is not None:
            self.demand_grid.copy_to_clipboard()
            self.status.set(f"Copied {self.current_table_name} demand table to clipboard.")

    def _copy_indicated(self):
        if self.indicated_grid is not None:
            self.indicated_grid.copy_to_clipboard()
            self.status.set("Copied indicated torque table to clipboard.")

    # ---- validation -----------------------------------------------------

    def _run_validation(self):
        self._apply_config()
        self.report_text.delete("1.0", "end")
        all_ok = True
        # Separate demand tables from the OSS modifier table.
        # OSS has different axes (pedal × OSS RPM, multipliers) and must not
        # be validated with the torque-table checks or included in
        # cross-table comparison.
        oss_table = self.demand_tables.get("OSS")
        demand_only = {k: v for k, v in self.demand_tables.items() if k != "OSS"}
        # Per-table validation (demand tables only)
        for name, table in demand_only.items():
            result = validate_demand_table(table, self.indicated, self.cfg)
            self._results[name] = result
            if not result.passed:
                all_ok = False
            report = format_report(result)
            self.report_text.insert("end", report + "\n\n")
        # OSS modifier self-check (multiplier bounds vs neutral 1.0)
        if oss_table is not None:
            oss_result = ValidationResult(table_name="OSS")
            check_oss_modifier(oss_table, self.cfg, oss_result)
            self._results["OSS"] = oss_result
            if not oss_result.passed:
                all_ok = False
            self.report_text.insert("end", format_report(oss_result) + "\n\n")
            # Effective-demand validation: base × OSS, separate report.
            # Uses the Normal table as the base (or the first available).
            base = demand_only.get("Normal")
            if base is None and demand_only:
                base = next(iter(demand_only.values()))
            if base is not None:
                eff_result = validate_with_oss(
                    base, oss_table, self.indicated, self.cfg)
                self._results["OSS_effective"] = eff_result
                if not eff_result.passed:
                    all_ok = False
                self.report_text.insert(
                    "end", format_report(eff_result) + "\n\n")
        # Cross-table comparison (demand tables only, excludes OSS)
        if len(demand_only) > 1:
            cross = compare_tables(demand_only, self.cfg)
            self.report_text.insert("end", format_report(cross) + "\n")
            if not cross.passed:
                all_ok = False
        # Logged engine max comparison (separate from indicated-table check)
        if self.wot_curve is not None:
            for name, table in demand_only.items():
                lm_result = validate_vs_logged_max(
                    table, self.wot_curve, self.cfg)
                self._results[f"{name}_vs_logged"] = lm_result
                if not lm_result.passed:
                    all_ok = False
                self.report_text.insert(
                    "end", format_report(lm_result) + "\n\n")
        # highlight the current demand table with issues
        self._highlight_issues()
        if all_ok:
            self.status.set("Validation complete — all tables PASS.")
        else:
            self.status.set("Validation complete — issues found. See report tab.")

    def _highlight_issues(self):
        """Highlight cells in the current demand table that have issues.

        Uses the same green/red heat gradient as the Torque Tuning Tool:
        red = critical (demand exceeds indicated), green = info.
        """
        result = self._results.get(self.current_table_name)
        if result is None or self.demand_grid is None:
            self._render_tables()
            return
        demand = self.demand_tables.get(self.current_table_name)
        if demand is None:
            self._render_tables()
            return
        # build a highlight array: negative = critical (red), positive = info (green)
        highlight = np.zeros_like(demand.values)
        for issue in result.issues:
            if issue.row_idx is not None and issue.col_idx is not None:
                if issue.severity == Severity.CRITICAL:
                    highlight[issue.row_idx, issue.col_idx] = -3.0
                elif issue.severity == Severity.WARNING:
                    highlight[issue.row_idx, issue.col_idx] = -2.0
                else:
                    highlight[issue.row_idx, issue.col_idx] = 1.0
        self._render_tables(
            demand_highlight=highlight,
            demand_highlight_label="red=critical/warning, green=info",
        )

    def _clear_report(self):
        self.report_text.delete("1.0", "end")
        self._render_tables()
        self.status.set("Report cleared.")

    # ---- plots ----------------------------------------------------------

    def _plot_surface(self):
        demand = self.demand_tables.get(self.current_table_name)
        if demand is None:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
        except ImportError:
            messagebox.showerror("Missing dependency", "matplotlib is required for plots.")
            return
        for child in self.plot_frame.winfo_children():
            child.destroy()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        P, R = np.meshgrid(demand.row_axis, demand.col_axis, indexing="ij")
        ax.plot_surface(P, R, demand.values, cmap="viridis", alpha=0.8)
        ax.set_xlabel("Pedal")
        ax.set_ylabel("RPM")
        ax.set_zlabel("Torque (N m)")
        ax.set_title(f"Driver Demand — {self.current_table_name}")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_max_comparison(self):
        demand = self.demand_tables.get(self.current_table_name)
        if demand is None or self.indicated is None:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
        except ImportError:
            messagebox.showerror("Missing dependency", "matplotlib is required for plots.")
            return
        for child in self.plot_frame.winfo_children():
            child.destroy()
        fig, ax = plt.subplots(figsize=(8, 5))
        rpm = demand.col_axis
        demand_max = demand.max_torque_per_rpm()
        ind_rpm = self.indicated.col_axis
        ind_max = self.indicated.max_torque_per_rpm()
        ax.plot(rpm, demand_max, "b-o", label="Demand max")
        ax.plot(ind_rpm, ind_max, "r-s", label="Indicated max")
        ax.fill_between(ind_rpm, 0, ind_max, alpha=0.1, color="red",
                        label="Engine capacity")
        ax.set_xlabel("RPM")
        ax.set_ylabel("Torque (N m)")
        ax.set_title(f"Demand max vs Indicated max — {self.current_table_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_gradient(self):
        demand = self.demand_tables.get(self.current_table_name)
        if demand is None:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
        except ImportError:
            messagebox.showerror("Missing dependency", "matplotlib is required for plots.")
            return
        for child in self.plot_frame.winfo_children():
            child.destroy()
        fig, ax = plt.subplots(figsize=(8, 5))
        gradients = np.diff(demand.values, axis=0)
        im = ax.imshow(gradients, aspect="auto", cmap="RdYlGn",
                       extent=[demand.col_axis[0], demand.col_axis[-1],
                               demand.row_axis[-1], demand.row_axis[0]])
        ax.set_xlabel("RPM")
        ax.set_ylabel("Pedal")
        ax.set_title(f"Torque gradient (d(Tq)/d(pedal)) — {self.current_table_name}")
        fig.colorbar(im, ax=ax, label="N m / pedal count")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


def main():
    app = DriverDemandApp()
    app.mainloop()


if __name__ == "__main__":
    main()
