#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # backend GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 3D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Savitzky–Golay (optionnel)
try:
    from scipy.signal import savgol_filter
    HAVE_SG = True
except Exception:
    HAVE_SG = False


APP_TITLE = "Marker Plotter – CSV"
CSV_EXPECTED_COLS = [
    "frame", "marker_id", "marker_name", "x_px_raw", "y_px_raw", "likelihood",
    "depth_mm", "X_m", "Y_m", "Z_m"
]


class MarkerPlotterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x700")

        self.df = None
        self.markers = []  # list of marker names
        self.marker_to_df = {}  # name -> df sorted by frame

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        # Top bar: file controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.csv_path_var = tk.StringVar(value="")
        ttk.Label(top, text="CSV:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(top, textvariable=self.csv_path_var, width=60).pack(side=tk.LEFT)
        ttk.Button(top, text="Ouvrir…", command=self.on_open_csv).pack(side=tk.LEFT, padx=6)

        # Options pane
        opts = ttk.LabelFrame(self, text="Options")
        opts.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        # Marker selection
        ttk.Label(opts, text="Marqueur").pack(anchor="w", padx=8, pady=(6, 2))
        self.marker_list = tk.Listbox(opts, selectmode=tk.SINGLE, height=12, exportselection=False)
        self.marker_list.pack(fill=tk.X, padx=8)
        self.marker_list.insert(tk.END, "All markers")

        # Axes
        ttk.Label(opts, text="Axes").pack(anchor="w", padx=8, pady=(10, 2))
        self.axes_var = tk.StringVar(value="x-y")
        axes_cb = ttk.Combobox(opts, textvariable=self.axes_var, state="readonly",
                               values=["x-y", "y-z", "x-z", "3D"])
        axes_cb.pack(fill=tk.X, padx=8)

        # Coordinate source
        ttk.Label(opts, text="Source des coordonnées").pack(anchor="w", padx=8, pady=(10, 2))
        self.src_var = tk.StringVar(value="auto")
        ttk.Radiobutton(opts, text="Auto (XYZ si dispo)", variable=self.src_var, value="auto").pack(anchor="w", padx=10)
        ttk.Radiobutton(opts, text="Caméra XYZ (m)", variable=self.src_var, value="xyz").pack(anchor="w", padx=10)
        ttk.Radiobutton(opts, text="Pixels + profondeur (z=depth_mm/1000)", variable=self.src_var, value="pxd").pack(anchor="w", padx=10)
        ttk.Radiobutton(opts, text="Pixels 2D (x,y)", variable=self.src_var, value="px").pack(anchor="w", padx=10)

        # Likelihood threshold
        ttk.Label(opts, text="Seuil likelihood (0–1)").pack(anchor="w", padx=8, pady=(10, 2))
        self.lk_var = tk.DoubleVar(value=0.0)
        ttk.Entry(opts, textvariable=self.lk_var).pack(fill=tk.X, padx=8)

        # Smoothing
        self.smooth_var = tk.BooleanVar(value=HAVE_SG)
        ttk.Checkbutton(opts, text="Lissage Savitzky–Golay", variable=self.smooth_var,
                        state=("normal" if HAVE_SG else "disabled")).pack(anchor="w", padx=8, pady=(10, 2))

        row = ttk.Frame(opts)
        row.pack(fill=tk.X, padx=8)
        ttk.Label(row, text="fenêtre").pack(side=tk.LEFT)
        self.sg_win_var = tk.IntVar(value=9)
        ttk.Entry(row, textvariable=self.sg_win_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="ordre").pack(side=tk.LEFT)
        self.sg_order_var = tk.IntVar(value=2)
        ttk.Entry(row, textvariable=self.sg_order_var, width=6).pack(side=tk.LEFT, padx=4)

        # Downsample
        ttk.Label(opts, text="Sous-échantillonnage (step)").pack(anchor="w", padx=8, pady=(10, 2))
        self.step_var = tk.IntVar(value=1)
        ttk.Entry(opts, textvariable=self.step_var).pack(fill=tk.X, padx=8)

        # Frame range
        ttk.Label(opts, text="Frame range [start, end] (optionnel)").pack(anchor="w", padx=8, pady=(10, 2))
        range_fr = ttk.Frame(opts); range_fr.pack(fill=tk.X, padx=8)
        self.start_var = tk.StringVar(value="")
        self.end_var   = tk.StringVar(value="")
        ttk.Entry(range_fr, textvariable=self.start_var, width=8).pack(side=tk.LEFT)
        ttk.Label(range_fr, text="→").pack(side=tk.LEFT, padx=4)
        ttk.Entry(range_fr, textvariable=self.end_var, width=8).pack(side=tk.LEFT)

        # Buttons
        ttk.Button(opts, text="Tracer", command=self.on_plot).pack(fill=tk.X, padx=8, pady=(14, 4))
        ttk.Button(opts, text="Enregistrer figure…", command=self.on_save_fig).pack(fill=tk.X, padx=8)
        ttk.Button(opts, text="Quitter", command=self.destroy).pack(fill=tk.X, padx=8, pady=(6, 10))

        # Plot area
        self.fig = plt.Figure(figsize=(7.5, 5.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Status bar
        self.status_var = tk.StringVar(value="Ouvre un CSV pour commencer…")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))

    # -------------- Data loading --------------
    def on_open_csv(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le CSV:\n{e}")
            return

        self.csv_path_var.set(path)
        missing = [c for c in CSV_EXPECTED_COLS if c not in self.df.columns]
        if missing:
            messagebox.showwarning("Colonnes manquantes",
                                   f"Colonnes attendues absentes: {missing}\nLe script tentera de continuer.")
        # Types
        for c in ["frame", "marker_id"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        for c in ["x_px_raw", "y_px_raw", "likelihood", "depth_mm", "X_m", "Y_m", "Z_m"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        # Build per-marker views
        if "marker_name" in self.df.columns:
            self.markers = sorted(self.df["marker_name"].dropna().unique().tolist())
        else:
            # fallback
            self.markers = sorted(self.df["marker_id"].dropna().unique().astype(int).tolist())

        self.marker_list.delete(0, tk.END)
        self.marker_list.insert(tk.END, "All markers")
        for m in self.markers:
            self.marker_list.insert(tk.END, str(m))

        # dict
        self.marker_to_df.clear()
        if "marker_name" in self.df.columns:
            for m in self.markers:
                d = self.df[self.df["marker_name"] == m].copy()
                d.sort_values("frame", inplace=True)
                self.marker_to_df[m] = d
        else:
            for m in self.markers:
                d = self.df[self.df["marker_id"] == m].copy()
                d.sort_values("frame", inplace=True)
                self.marker_to_df[m] = d

        self.status_var.set(f"Chargé: {os.path.basename(path)} | {len(self.markers)} marqueurs")

    # -------------- Helpers --------------
    def _get_selection(self):
        # marker choice
        sel = self.marker_list.curselection()
        if not sel:
            marker_choice = "All markers"
        else:
            marker_choice = self.marker_list.get(sel[0])
        return marker_choice, self.axes_var.get(), self.src_var.get()

    def _prepare_series(self, d: pd.DataFrame, src: str):
        """
        Retourne (xs, ys, zs, frames) selon source choisie.
        xs, ys, zs sont des np.ndarray (peuvent être NaN si indispo).
        """
        # frames selection
        frames = d["frame"].to_numpy(dtype=float) if "frame" in d.columns else np.arange(len(d))

        # frame range filter
        s_txt, e_txt = self.start_var.get().strip(), self.end_var.get().strip()
        if s_txt or e_txt:
            try:
                s = float(s_txt) if s_txt else frames.min()
                e = float(e_txt) if e_txt else frames.max()
                sel = (frames >= s) & (frames <= e)
                d = d.loc[sel]
                frames = frames[sel]
            except Exception:
                pass

        # likelihood filter
        lk_thr = float(self.lk_var.get())
        mask_lk = (d["likelihood"].to_numpy() >= lk_thr) if "likelihood" in d.columns else np.ones(len(d), bool)

        # choose coordinate source
        # 1) auto -> xyz if available
        if src == "auto":
            if {"X_m", "Y_m", "Z_m"}.issubset(d.columns) and d[["X_m","Y_m","Z_m"]].notna().any().any():
                src = "xyz"
            elif {"depth_mm"}.issubset(d.columns):
                src = "pxd"
            else:
                src = "px"

        if src == "xyz":
            xs = d["X_m"].to_numpy(dtype=float)
            ys = d["Y_m"].to_numpy(dtype=float)
            zs = d["Z_m"].to_numpy(dtype=float)
        elif src == "pxd":
            xs = d["x_px_raw"].to_numpy(dtype=float)
            ys = d["y_px_raw"].to_numpy(dtype=float)
            # depth_mm -> m
            zm = d["depth_mm"].to_numpy(dtype=float)
            zs = np.where(np.isfinite(zm), zm / 1000.0, np.nan)
        else:  # "px"
            xs = d["x_px_raw"].to_numpy(dtype=float)
            ys = d["y_px_raw"].to_numpy(dtype=float)
            zs = np.full_like(xs, np.nan, dtype=float)

        # apply likelihood mask
        xs = np.where(mask_lk, xs, np.nan)
        ys = np.where(mask_lk, ys, np.nan)
        zs = np.where(mask_lk, zs, np.nan)

        # downsample
        step = max(1, int(self.step_var.get()))
        xs, ys, zs, frames = xs[::step], ys[::step], zs[::step], frames[::step]

        # smoothing
        if self.smooth_var.get() and HAVE_SG:
            win = int(self.sg_win_var.get())
            order = int(self.sg_order_var.get())
            if win % 2 == 0:
                win += 1
            if len(xs) >= win and win >= (order + 2):
                # Interp NaN then smooth
                def _fill_and_sg(y):
                    y = y.astype(float)
                    bad = ~np.isfinite(y)
                    if bad.any():
                        t = np.arange(len(y))
                        good = ~bad
                        if good.sum() >= 2:
                            y[bad] = np.interp(t[bad], t[good], y[good])
                    return savgol_filter(y, win, order, mode="interp")
                xs = _fill_and_sg(xs)
                ys = _fill_and_sg(ys)
                if np.isfinite(zs).any():
                    zs = _fill_and_sg(zs)

        return xs, ys, zs, frames

    # -------------- Plot --------------
    def on_plot(self):
        if self.df is None:
            messagebox.showinfo("Info", "Ouvre d'abord un CSV.")
            return

        marker_choice, axes_mode, src = self._get_selection()
        self.fig.clf()

        if axes_mode == "3D":
            ax = self.fig.add_subplot(111, projection="3d")
        else:
            ax = self.fig.add_subplot(111)

        palette = plt.cm.get_cmap("tab20", max(1, len(self.markers)))

        def plot_one(name, color=None):
            d = self.marker_to_df.get(name, None)
            if d is None or len(d) == 0:
                return
            xs, ys, zs, fr = self._prepare_series(d, src)
            if axes_mode == "3D":
                # prefer XYZ; if Z is NaN -> skip
                if not np.isfinite(zs).any():
                    return
                ax.plot(xs, ys, zs, lw=1.5, color=color, label=str(name))
                # last point
                idx_last = np.where(np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs))[0]
                if idx_last.size:
                    j = idx_last[-1]
                    ax.scatter(xs[j], ys[j], zs[j], s=25, color=color, marker="o")
            else:
                if axes_mode == "x-y":
                    x, y = xs, ys
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                elif axes_mode == "y-z":
                    if not np.isfinite(zs).any():
                        return
                    x, y = ys, zs
                    ax.set_xlabel("y")
                    ax.set_ylabel("z")
                elif axes_mode == "x-z":
                    if not np.isfinite(zs).any():
                        return
                    x, y = xs, zs
                    ax.set_xlabel("x")
                    ax.set_ylabel("z")
                else:
                    x, y = xs, ys
                ax.plot(x, y, lw=1.5, color=color, label=str(name))
                # last point
                idx_last = np.where(np.isfinite(x) & np.isfinite(y))[0]
                if idx_last.size:
                    j = idx_last[-1]
                    ax.scatter(x[j], y[j], s=25, color=color, marker="o")

        if marker_choice == "All markers":
            for k, name in enumerate(self.markers):
                plot_one(name, color=palette(k % palette.N))
            ax.legend(loc="best", fontsize=8, ncol=2)
            ttl = f"Tous les marqueurs – {axes_mode} – source={self.src_var.get()}"
        else:
            # value could be marker_name (string) or id-as-string
            # unify to marker_name when possible
            name = marker_choice
            if name not in self.marker_to_df and "marker_name" in self.df.columns:
                # maybe user picked ID and we have names; fallback to first that matches ID
                try:
                    mid = int(name)
                    d = self.df[self.df["marker_id"] == mid]
                    if "marker_name" in d.columns and not d.empty:
                        name = d["marker_name"].iloc[0]
                except Exception:
                    pass
            plot_one(name, color="C0")
            ttl = f"Marqueur: {name} – {axes_mode} – source={self.src_var.get()}"

        ax.set_title(ttl)
        if axes_mode != "3D":
            ax.grid(True, ls="--", alpha=0.4)
        else:
            # make axes a bit equal when XYZ available
            try:
                xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
                ranges = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])
                m = ranges.max()
                def mid(lim): return 0.5*(lim[0]+lim[1])
                ax.set_xlim3d(mid(xlim)-m/2, mid(xlim)+m/2)
                ax.set_ylim3d(mid(ylim)-m/2, mid(ylim)+m/2)
                ax.set_zlim3d(mid(zlim)-m/2, mid(zlim)+m/2)
            except Exception:
                pass

        self.canvas.draw()
        self.status_var.set("Trace terminé")

    # -------------- Save Figure --------------
    def on_save_fig(self):
        if self.fig is None:
            return
        path = filedialog.asksaveasfilename(
            title="Enregistrer la figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if not path:
            return
        try:
            self.fig.savefig(path, bbox_inches="tight", dpi=150)
            self.status_var.set(f"Figure enregistrée: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'enregistrer la figure:\n{e}")


def main():
    app = MarkerPlotterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
