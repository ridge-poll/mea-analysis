#!/usr/bin/env python3
"""
mea_gui.py — Tkinter GUI for HD-MEA trace visualization.

Depends on: mea_io.py, mea_plot.py (same directory or on PYTHONPATH)
Run:  python mea_gui.py [optional_recording.brw]
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import numpy as np

try:
    import mea_io
    import mea_plot
except ImportError as e:
    print(f"ERROR: {e}\nmea_io.py and mea_plot.py must be in the same directory.")
    sys.exit(1)


# ── Palette ────────────────────────────────────────────────────────────────────
CLR_BG           = "#1e1e2e"   # main window background
CLR_PANEL        = "#2a2a3e"   # left electrode panel
CLR_TOOLBAR      = "#16162a"   # top / bottom bars
CLR_BTN          = "#45475a"   # idle button / electrode cell fill
CLR_BTN_HOVER    = "#585b70"   # hover (not wired yet, available)
CLR_BTN_ACTIVE   = "#89b4fa"   # selected electrode fill  (bright blue)
CLR_TEXT         = "#000000"   # primary text  (light lavender — readable on dark)
CLR_TEXT_ON_ACTIVE = "#1e1e2e" # text printed ON a bright active button
CLR_SUBTEXT      = "#a6adc8"   # secondary / dim text
CLR_ACCENT       = "#89b4fa"   # accent (matches active electrode)
CLR_PLOT_BG      = "#ffffff"   # matplotlib axes background

# Range-slider colours
CLR_TRACK        = "#313244"   # unselected track
CLR_RANGE        = "#585b70"   # filled range between handles
CLR_HANDLE       = "#89b4fa"   # drag handle circles
CLR_HANDLE_GRAB  = "#b4befe"   # handle while dragging

FONT_UI   = ("Helvetica", 10)
FONT_MONO = ("Courier", 9)
FONT_TINY = ("Helvetica", 8)

DEBOUNCE_MS  = 600   # ms after handle release before auto-replot
GRID_CELL_PX = 8    # electrode cell size (px)
GRID_PAD_PX  = 2     # gap between cells


# ══════════════════════════════════════════════════════════════════════════════
class RangeSlider(tk.Canvas):
    """
    A dual-handle range slider drawn on a Canvas.

    Attributes (read from outside)
    --------------------------------
    start_var : tk.IntVar  — current left-handle value
    end_var   : tk.IntVar  — current right-handle value

    Callback
    --------
    command(start, end) is called whenever either handle moves.
    """

    _PAD   = 16   # horizontal padding so handles don't clip at edges
    _H     = 36   # canvas height
    _TH    =  4   # track half-height
    _HR    =  8   # handle radius

    def __init__(self, master, from_=0, to=100, command=None, **kw):
        kw.setdefault("bg", CLR_TOOLBAR)
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("height", self._H)
        super().__init__(master, **kw)

        self._from   = from_
        self._to     = to
        self._cmd    = command
        self._drag   = None   # "start" | "end" | None

        self.start_var = tk.IntVar(value=from_)
        self.end_var   = tk.IntVar(value=to)

        self._track   = self.create_rectangle(0, 0, 0, 0, fill=CLR_TRACK,   outline="")
        self._range   = self.create_rectangle(0, 0, 0, 0, fill=CLR_RANGE,   outline="")
        self._h_start = self.create_oval(0, 0, 0, 0,      fill=CLR_HANDLE,  outline="")
        self._h_end   = self.create_oval(0, 0, 0, 0,      fill=CLR_HANDLE,  outline="")
        self._lbl_s   = self.create_text(0, 0, fill=CLR_TEXT, font=FONT_TINY, anchor="n")
        self._lbl_e   = self.create_text(0, 0, fill=CLR_TEXT, font=FONT_TINY, anchor="n")

        self.bind("<Configure>",       self._redraw)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ── Public ─────────────────────────────────────────────────────────────────

    def set_range(self, from_, to, start=None, end=None):
        """Update total range and optionally reset handle positions."""
        self._from = from_
        self._to   = to
        self.start_var.set(from_ if start is None else start)
        self.end_var.set(to if end is None else end)
        self._redraw()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _val_to_x(self, val):
        w = self.winfo_width() or 400
        frac = (val - self._from) / max(self._to - self._from, 1)
        return self._PAD + frac * (w - 2 * self._PAD)

    def _x_to_val(self, x):
        w = self.winfo_width() or 400
        frac = (x - self._PAD) / max(w - 2 * self._PAD, 1)
        frac = max(0.0, min(1.0, frac))
        return round(self._from + frac * (self._to - self._from))

    def _redraw(self, _=None):
        w  = self.winfo_width() or 400
        cy = self._H // 2
        th = self._TH
        hr = self._HR

        xs = self._val_to_x(self.start_var.get())
        xe = self._val_to_x(self.end_var.get())

        # Track
        self.coords(self._track, self._PAD, cy - th, w - self._PAD, cy + th)
        # Filled range
        self.coords(self._range, xs, cy - th, xe, cy + th)
        # Handles
        self.coords(self._h_start, xs - hr, cy - hr, xs + hr, cy + hr)
        self.coords(self._h_end,   xe - hr, cy - hr, xe + hr, cy + hr)
        # Labels
        self.coords(self._lbl_s, xs, cy + hr + 1)
        self.coords(self._lbl_e, xe, cy + hr + 1)
        self.itemconfig(self._lbl_s, text=str(self.start_var.get()))
        self.itemconfig(self._lbl_e, text=str(self.end_var.get()))

        # Keep handles on top
        self.tag_raise(self._h_start)
        self.tag_raise(self._h_end)

    def _nearest_handle(self, x):
        xs = self._val_to_x(self.start_var.get())
        xe = self._val_to_x(self.end_var.get())
        return "start" if abs(x - xs) <= abs(x - xe) else "end"

    def _on_press(self, event):
        self._drag = self._nearest_handle(event.x)
        hid = self._h_start if self._drag == "start" else self._h_end
        self.itemconfig(hid, fill=CLR_HANDLE_GRAB)

    def _on_drag(self, event):
        if self._drag is None:
            return
        val = self._x_to_val(event.x)
        if self._drag == "start":
            val = min(val, self.end_var.get() - 1)
            val = max(val, self._from)
            self.start_var.set(val)
        else:
            val = max(val, self.start_var.get() + 1)
            val = min(val, self._to)
            self.end_var.set(val)
        self._redraw()
        if self._cmd:
            self._cmd(self.start_var.get(), self.end_var.get())

    def _on_release(self, event):
        if self._drag:
            hid = self._h_start if self._drag == "start" else self._h_end
            self.itemconfig(hid, fill=CLR_HANDLE)
        self._drag = None


# ══════════════════════════════════════════════════════════════════════════════
class MEAApp:
    def __init__(self, root, initial_file=None):
        self.root = root
        self.root.title("MEA Trace Viewer")
        self.root.configure(bg=CLR_BG)
        self.root.minsize(960, 600)

        # ── Application state ──────────────────────────────────────────────
        self.filepath     = None
        self.meta         = None
        self.selected_chs = set()
        self._cell_rects  = {}    # flat_idx -> canvas rect id
        self._grid_items  = {}    # rect id  -> flat_idx
        self._debounce_id = None

        # ── Build UI ───────────────────────────────────────────────────────
        self._build_toolbar()
        self._build_main_area()
        self._build_bottom_bar()

        if initial_file:
            self._load_file(initial_file)

    # ── Toolbar ────────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=CLR_TOOLBAR, pady=5)
        bar.pack(side=tk.TOP, fill=tk.X)

        _btn(bar, "Open file…", self._pick_file).pack(side=tk.LEFT, padx=(8, 4))

        self.lbl_file = tk.Label(
            bar, text="No file loaded", fg=CLR_SUBTEXT, bg=CLR_TOOLBAR,
            font=FONT_MONO, anchor="w",
        )
        self.lbl_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.lbl_meta = tk.Label(bar, text="", fg=CLR_SUBTEXT, bg=CLR_TOOLBAR, font=FONT_TINY)
        self.lbl_meta.pack(side=tk.RIGHT, padx=8)

        _btn(bar, "Save figure…", self._save_figure).pack(side=tk.RIGHT, padx=(4, 8))

    # ── Main area ──────────────────────────────────────────────────────────────

    def _build_main_area(self):
        pane = tk.Frame(self.root, bg=CLR_BG)
        pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: electrode panel
        left = tk.Frame(pane, bg=CLR_PANEL, width=360)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 3), pady=6)
        left.pack_propagate(False)

        tk.Label(
            left, text="Electrodes", fg=CLR_ACCENT, bg=CLR_PANEL,
            font=("Helvetica", 10, "bold"), pady=4,
        ).pack()

        sub = tk.Frame(left, bg=CLR_PANEL)
        sub.pack(fill=tk.X, padx=4, pady=(0, 4))
        _btn(sub, "All",  self._select_all,  tiny=True).pack(side=tk.LEFT, padx=2)
        _btn(sub, "None", self._select_none, tiny=True).pack(side=tk.LEFT, padx=2)
        self.lbl_sel = tk.Label(sub, text="0 selected", fg=CLR_SUBTEXT, bg=CLR_PANEL, font=FONT_TINY)
        self.lbl_sel.pack(side=tk.RIGHT, padx=4)

        grid_outer = tk.Frame(left, bg=CLR_PANEL)
        grid_outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self.grid_canvas = tk.Canvas(grid_outer, bg=CLR_PANEL, highlightthickness=0)
        vsb = tk.Scrollbar(grid_outer, orient=tk.VERTICAL,   command=self.grid_canvas.yview)
        hsb = tk.Scrollbar(grid_outer, orient=tk.HORIZONTAL, command=self.grid_canvas.xview)
        self.grid_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.grid_canvas.pack(fill=tk.BOTH, expand=True)
        self.grid_canvas.bind("<Button-1>",  self._on_grid_click)
        self.grid_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.grid_canvas.bind("<Button-4>",   self._on_mousewheel)
        self.grid_canvas.bind("<Button-5>",   self._on_mousewheel)

        # Right: matplotlib canvas
        right = tk.Frame(pane, bg=CLR_PLOT_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 6), pady=6)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.span = SpanSelector(
            self.ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor=CLR_ACCENT),
            interactive=False,
            drag_from_anywhere=True,
        )

        self._style_axes()
        self.root.after(50, self._draw_placeholder)
    

    def _on_span_select(self, xmin, xmax):
        if self.meta is None:
            return

        # Ensure left-to-right ordering
        xmin, xmax = sorted((xmin, xmax))

        sr = self.meta["sampling_rate"]

        # x-axis is currently in seconds
        start = int(xmin * sr)
        end   = int(xmax * sr)

        # Clamp
        start = max(0, start)
        end   = min(self.meta["n_frames"], end)

        # Prevent zero-width window
        if end <= start:
            return

        # Update slider
        self.range_slider.start_var.set(start)
        self.range_slider.end_var.set(end)

        self.range_slider._redraw()

        # Update text label
        self._update_window_label(start, end)

        # Optional auto replot
        self._do_plot()

    # ── Bottom bar (dual-handle range slider) ──────────────────────────────────

    def _build_bottom_bar(self):
        bar = tk.Frame(self.root, bg=CLR_TOOLBAR, pady=4)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Window info label (left side)
        self.lbl_window = tk.Label(bar, text="", fg=CLR_SUBTEXT, bg=CLR_TOOLBAR, font=FONT_TINY)
        self.lbl_window.pack(side=tk.LEFT, padx=(10, 6))

        # Plot button (right side — pack before slider so it anchors right)
        _btn(bar, "Plot", self._do_plot, accent=True).pack(side=tk.RIGHT, padx=(6, 10))

        # Dual-handle range slider fills the middle
        self.range_slider = RangeSlider(
            bar, from_=0, to=1000, command=self._on_range_change,
        )
        self.range_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=4)

    def _on_range_change(self, start, end):
        """Called continuously while dragging."""
        self._update_window_label(start, end)
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(DEBOUNCE_MS, self._do_plot)

    def _update_window_label(self, start=None, end=None):
        if start is None:
            start = self.range_slider.start_var.get()
        if end is None:
            end = self.range_slider.end_var.get()
        sr = self.meta["sampling_rate"] if self.meta else None
        if sr:
            self.lbl_window.config(
                text=f"{start / sr:.2f} s – {end / sr:.2f} s  ({(end - start) / sr:.2f} s)"
            )
        else:
            self.lbl_window.config(text=f"samples {start:,} – {end:,}")

    # ── File loading ───────────────────────────────────────────────────────────

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Open MEA recording",
            filetypes=[("BrainWave / HDF5", "*.brw *.h5 *.hdf5"), ("All files", "*.*")],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            meta = mea_io.open_recording(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.filepath     = path
        self.meta         = meta
        self.selected_chs = set()

        self.lbl_file.config(text=Path(path).name, fg=CLR_TEXT)
        sr  = meta["sampling_rate"]
        dur = f"{meta['n_frames'] / sr:.1f} s" if sr else f"{meta['n_frames']:,} frames"
        self.lbl_meta.config(text=f"{meta['n_channels']} ch  |  {dur}  |  {meta['layout']}")

        # Update range slider bounds; default window = first 3000 frames (or all)
        nf          = meta["n_frames"]
        default_end = min(nf, 3000)
        self.range_slider.set_range(0, nf, start=0, end=default_end)
        self._update_window_label(0, default_end)

        self._build_grid(meta)
        self._draw_placeholder()

    # ── Electrode grid ─────────────────────────────────────────────────────────

    def _build_grid(self, meta):
        chs  = meta["chs"]
        rows = chs["Row"].astype(int)
        cols = chs["Col"].astype(int)
        rmax = rows.max()
        cmax = cols.max()

        cell = GRID_CELL_PX
        pad  = GRID_PAD_PX
        step = cell + pad

        self.grid_canvas.delete("all")
        self._cell_rects.clear()
        self._grid_items.clear()

        for flat_idx, (r, c) in enumerate(zip(rows, cols)):
            r_plot = rmax - r          # flip: row 0 at bottom
            x0 = pad + c * step
            y0 = pad + r_plot * step
            rect = self.grid_canvas.create_rectangle(
                x0, y0, x0 + cell, y0 + cell,
                fill=CLR_BTN, outline=CLR_SUBTEXT, width=1,
                tags=("electrode",),
            )
            self._cell_rects[flat_idx] = rect
            self._grid_items[rect]     = flat_idx

        canvas_w = (cmax + 1) * step + pad
        canvas_h = (rmax + 1) * step + pad
        self.grid_canvas.config(scrollregion=(0, 0, canvas_w, canvas_h))
        self._update_sel_label()

    def _on_grid_click(self, event):
        cx = self.grid_canvas.canvasx(event.x)
        cy = self.grid_canvas.canvasy(event.y)
        for item in self.grid_canvas.find_overlapping(cx - 1, cy - 1, cx + 1, cy + 1):
            if item in self._grid_items:
                self._toggle_electrode(self._grid_items[item])
                break

    def _toggle_electrode(self, flat_idx):
        if flat_idx in self.selected_chs:
            self.selected_chs.discard(flat_idx)
            self.grid_canvas.itemconfig(self._cell_rects[flat_idx], fill=CLR_BTN)
        else:
            self.selected_chs.add(flat_idx)
            self.grid_canvas.itemconfig(self._cell_rects[flat_idx], fill=CLR_BTN_ACTIVE)
        self._update_sel_label()

    def _select_all(self):
        for flat_idx, rect in self._cell_rects.items():
            self.selected_chs.add(flat_idx)
            self.grid_canvas.itemconfig(rect, fill=CLR_BTN_ACTIVE)
        self._update_sel_label()

    def _select_none(self):
        for rect in self._cell_rects.values():
            self.grid_canvas.itemconfig(rect, fill=CLR_BTN)
        self.selected_chs.clear()
        self._update_sel_label()

    def _update_sel_label(self):
        self.lbl_sel.config(text=f"{len(self.selected_chs)} / {len(self._cell_rects)} selected")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.grid_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.grid_canvas.yview_scroll(1, "units")
        else:
            self.grid_canvas.yview_scroll(int(-event.delta / 60), "units")

    # ── Plotting ───────────────────────────────────────────────────────────────

    def _do_plot(self):
        self._debounce_id = None

        if self.meta is None:
            messagebox.showinfo("No file", "Please open a recording file first.")
            return

        channels = sorted(self.selected_chs)
        if not channels:
            messagebox.showinfo("No channels", "Toggle at least one electrode in the grid.")
            return

        start = self.range_slider.start_var.get()
        end   = self.range_slider.end_var.get()

        try:
            traces = mea_io.load_traces(
                filepath        = self.filepath,
                channel_indices = channels,
                start           = start,
                end             = end,
            )
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        x, xlabel = mea_plot.make_x_axis(
            n_samples        = traces.shape[0],
            start_sample     = start,
            sampling_rate_hz = self.meta["sampling_rate"],
        )

        mea_plot.render_traces(self.ax, traces, channels, x, xlabel)
        self._style_axes()
        self.ax.set_title(
            f"{len(channels)} channel(s)  |  {traces.shape[0]:,} samples  |  "
            f"{Path(self.filepath).name}",
            fontsize=8, color=CLR_TEXT,
        )
        self.fig.tight_layout()
        self.mpl_canvas.draw()

    def _style_axes(self):
        """Apply dark theme to axes (safe to call after cla())."""
        self.fig.patch.set_facecolor(CLR_PLOT_BG)
        self.ax.set_facecolor(CLR_PLOT_BG)
        self.ax.tick_params(colors=CLR_TEXT)
        self.ax.xaxis.label.set_color(CLR_TEXT)
        self.ax.yaxis.label.set_color(CLR_TEXT)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(CLR_SUBTEXT)

    def _draw_placeholder(self):
        self.ax.cla()
        self._style_axes()
        self.ax.text(
            0.5, 0.5, "Select electrodes and press Plot",
            ha="center", va="center", transform=self.ax.transAxes,
            color=CLR_SUBTEXT, fontsize=11,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # self.fig.tight_layout()
        self.mpl_canvas.draw()

    # ── Save ───────────────────────────────────────────────────────────────────

    def _save_figure(self):
        path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight",
                             facecolor=self.fig.get_facecolor())
            print(f"Saved → {path}")


# ── Shared button factory ──────────────────────────────────────────────────────

def _btn(parent, text, cmd, tiny=False, accent=False):
    """Consistent button styling in one place."""
    return tk.Button(
        parent, text=text, command=cmd, relief=tk.FLAT,
        font=FONT_TINY if tiny else FONT_UI,
        bg=CLR_ACCENT if accent else CLR_BTN,
        fg=CLR_TEXT_ON_ACTIVE if accent else CLR_TEXT,
        activebackground=CLR_BTN_HOVER,
        activeforeground=CLR_TEXT,
        padx=8 if not tiny else 4,
    )


# ══════════════════════════════════════════════════════════════════════════════

def main():
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    MEAApp(root, initial_file=initial)
    root.mainloop()


if __name__ == "__main__":
    main()