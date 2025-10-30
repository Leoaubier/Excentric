#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

APP_TITLE = "Marker Plotter – MultiHeader CSV (XYZ en m)"

prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep

# Optionnel: mets ici un chemin par défaut vers ton CSV
DEFAULT_CSV = ""  # ex: r"/home/mickael/Documents/Leo/Excentric/markers_left_arm_xyz_m_filt.csv"

def _first_existing_col(df, candidates):
    """Retourne le premier tuple de colonne MultiIndex existant parmi 'candidates'."""
    for col in candidates:
        if col in df.columns:
            return col
    return None

def _find_xyz_columns(df, marker_name):
    """
    Cherche (X,Y,Z) pour un 'marker_name' dans un CSV à 3 niveaux (marker, axis, unit).
    Tolère:
      - axis: 'x' ou 'X', 'y'/'Y', 'z'/'Z'
      - unit: n'importe quoi (m, metre, '', etc.)
    Retourne dict {'x': col or None, 'y':..., 'z':...} où col = tuple MultiIndex.
    """
    lvl0 = marker_name
    axes_lower = ['X', 'Y', 'Z']
    out = {'X': None, 'Y': None, 'Z': None}

    # collecte de toutes les colonnes de ce marqueur
    cols = [c for c in df.columns if isinstance(c, tuple) and len(c) == 3 and c[0] == lvl0]
    if not cols:
        return out

    # on mappe par axis (insensible à la casse), unit libre
    for axis in axes_lower:
        # candidats = toutes combinaisons avec axis en 2e niveau (insensible casse)
        cands = [c for c in cols if str(c[1]).lower() == axis]
        # si aucune, tenter l'autre casse (déjà géré par lower), sinon versions "_m" éventuelles
        if not cands:
            continue
        # priorise une unité 'm' si présente
        pref = [c for c in cands if str(c[2]).lower() in ('m', 'meter', 'metre', 'meters', '')]
        out[axis] = pref[0] if pref else cands[0]

    return out

class MarkerPlotterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x700")

        self.df = None
        self.markers = []
        self.marker_to_xyz = {}

        self._build_ui()

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.csv_path_var = tk.StringVar(value=DEFAULT_CSV)
        ttk.Label(top, text="CSV:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(top, textvariable=self.csv_path_var, width=60).pack(side=tk.LEFT)
        ttk.Button(top, text="Ouvrir…", command=self.on_open_csv).pack(side=tk.LEFT, padx=6)

        # Options
        opts = ttk.LabelFrame(self, text="Options")
        opts.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        ttk.Label(opts, text="Marqueur").pack(anchor="w", padx=8, pady=(6, 2))
        self.marker_list = tk.Listbox(opts, selectmode=tk.SINGLE, height=12, exportselection=False)
        self.marker_list.pack(fill=tk.X, padx=8)
        self.marker_list.insert(tk.END, "All markers")

        ttk.Label(opts, text="Axes").pack(anchor="w", padx=8, pady=(10, 2))
        self.axes_var = tk.StringVar(value="x-y")
        axes_cb = ttk.Combobox(opts, textvariable=self.axes_var, state="readonly",
                               values=["x-y", "y-z", "x-z", "3D"])
        axes_cb.pack(fill=tk.X, padx=8)

        ttk.Label(opts, text="Sous-échantillonnage (step)").pack(anchor="w", padx=8, pady=(10, 2))
        self.step_var = tk.IntVar(value=1)
        ttk.Entry(opts, textvariable=self.step_var).pack(fill=tk.X, padx=8)

        ttk.Label(opts, text="Frame range [start, end] (optionnel)").pack(anchor="w", padx=8, pady=(10, 2))
        range_fr = ttk.Frame(opts); range_fr.pack(fill=tk.X, padx=8)
        self.start_var = tk.StringVar(value="")
        self.end_var   = tk.StringVar(value="")
        ttk.Entry(range_fr, textvariable=self.start_var, width=8).pack(side=tk.LEFT)
        ttk.Label(range_fr, text="→").pack(side=tk.LEFT, padx=4)
        ttk.Entry(range_fr, textvariable=self.end_var, width=8).pack(side=tk.LEFT)

        ttk.Button(opts, text="Tracer", command=self.on_plot).pack(fill=tk.X, padx=8, pady=(14, 4))
        ttk.Button(opts, text="Enregistrer figure…", command=self.on_save_fig).pack(fill=tk.X, padx=8)
        ttk.Button(opts, text="Quitter", command=self.destroy).pack(fill=tk.X, padx=8, pady=(6, 10))

        # Plot zone
        self.fig = plt.Figure(figsize=(7.5, 5.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Status
        self.status_var = tk.StringVar(value="Ouvre un CSV (3 lignes d’en-tête) pour commencer…")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))

        # si un CSV par défaut est fourni, tente de le charger
        if DEFAULT_CSV and os.path.isfile(DEFAULT_CSV):
            self._load_csv(DEFAULT_CSV)

    def on_open_csv(self):
        path = filedialog.askopenfilename(
            title="Choisir un CSV (3 lignes d’en-tête)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._load_csv(path)

    def _load_csv(self, path):
        try:
            df = pd.read_csv(path, header=[0, 1, 2])
            if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 3:
                raise ValueError("Ce CSV n’a pas 3 lignes d’en-tête (MultiIndex).")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le CSV comme MultiIndex (3 headers):\n{e}")
            return

        self.csv_path_var.set(path)
        self.df = df

        # Récupère markers (lvl0) hors 'meta'
        lvl0 = self.df.columns.get_level_values(0)
        markers = sorted(set([m for m in lvl0 if m != "meta"]))
        self.markers = markers

        # Frame (si présente)
        frames = None
        if ("meta", "frame", "idx") in self.df.columns:
            frames = pd.to_numeric(self.df[("meta", "frame", "idx")], errors="coerce").to_numpy()

        # Prépare dict séries
        self.marker_to_xyz.clear()
        for m in self.markers:
            xyz_cols = _find_xyz_columns(self.df, m)
            # si l’export a utilisé X/Y/Z en majuscules (nos CSV récents), ça passe;
            # sinon il trouvera x/y/z en minuscules.
            def _to_arr(col):
                return pd.to_numeric(self.df[col], errors="coerce").to_numpy() if col else np.full(len(self.df), np.nan)
            xs = _to_arr(xyz_cols['X'])
            ys = _to_arr(xyz_cols['Y'])
            zs = _to_arr(xyz_cols['Z'])
            self.marker_to_xyz[m] = {
                "x": xs, "y": ys, "z": zs,
                "frame": frames if frames is not None else np.arange(len(xs))
            }

        # UI list
        self.marker_list.delete(0, tk.END)
        self.marker_list.insert(tk.END, "All markers")
        for m in self.markers:
            self.marker_list.insert(tk.END, str(m))

        self.status_var.set(f"Chargé: {os.path.basename(path)} | {len(self.markers)} marqueurs")

    def _get_selection(self):
        sel = self.marker_list.curselection()
        choice = self.marker_list.get(sel[0]) if sel else "All markers"
        return choice, self.axes_var.get()

    def _slice_range_and_step(self, arrs):
        n = len(arrs[0])
        start_txt, end_txt = self.start_var.get().strip(), self.end_var.get().strip()
        idx = np.arange(n)
        if start_txt or end_txt:
            try:
                s = float(start_txt) if start_txt else idx.min()
                e = float(end_txt) if end_txt else idx.max()
                frames = arrs[-1]
                if frames is not None and frames.shape == idx.shape and np.issubdtype(frames.dtype, np.number):
                    sel = (frames >= s) & (frames <= e)
                else:
                    sel = (idx >= s) & (idx <= e)
                arrs = [a[sel] if a is not None else None for a in arrs]
            except Exception:
                pass
        step = max(1, int(self.step_var.get()))
        return [a[::step] if a is not None else None for a in arrs]

    def on_plot(self):
        if self.df is None:
            messagebox.showinfo("Info", "Ouvre d’abord un CSV (3 lignes d’en-tête).")
            return

        marker_choice, axes_mode = self._get_selection()
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection="3d") if axes_mode == "3D" else self.fig.add_subplot(111)

        # Matplotlib >=3.7 : nouvelle API colormaps
        # --- à la place de :
        # palette = matplotlib.colormaps.get_cmap("tab20", max(1, len(self.markers)))

        # --- mets :
        N = max(1, len(self.markers))
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(N)
        # fabrique une liste de N couleurs bien espacées dans la cmap
        colors = [cmap(i / (N - 1 if N > 1 else 1)) for i in range(N)]

        def plot_one(name, color=None):
            data = self.marker_to_xyz.get(name)
            if not data:
                return
            xs = data["x"].astype(float)
            ys = data["y"].astype(float)
            zs = data["z"].astype(float)
            fr = data["frame"].astype(float) if data["frame"] is not None else np.arange(len(xs), dtype=float)
            xs, ys, zs, fr = self._slice_range_and_step([xs, ys, zs, fr])

            if axes_mode == "3D":
                if np.isfinite(zs).any():
                    ax.plot(xs, ys, zs, lw=1.5, color=color, label=str(name))
                    idx_last = np.where(np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs))[0]
                    if idx_last.size:
                        j = idx_last[-1]
                        ax.scatter(xs[j], ys[j], zs[j], s=25, color=color, marker="o")
            else:
                if axes_mode == "x-y":
                    x, y = xs, ys
                    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
                elif axes_mode == "y-z":
                    if not np.isfinite(zs).any(): return
                    x, y = ys, zs
                    ax.set_xlabel("y (m)"); ax.set_ylabel("z (m)")
                elif axes_mode == "x-z":
                    if not np.isfinite(zs).any(): return
                    x, y = xs, zs
                    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
                else:
                    x, y = xs, ys
                    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

                ax.plot(x, y, lw=1.5, color=color, label=str(name))
                idx_last = np.where(np.isfinite(x) & np.isfinite(y))[0]
                if idx_last.size:
                    j = idx_last[-1]
                    ax.scatter(x[j], y[j], s=25, color=color, marker="o")

        if marker_choice == "All markers":
            for k, name in enumerate(self.markers):
                plot_one(name, color = colors[k % N])
            # légende seulement si nécessaire
            handles, labels = ax.get_legend_handles_labels()
            pairs = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
            if pairs and len(self.markers) <= 20:
                h, l = zip(*pairs)
                ax.legend(h, l, loc="best", fontsize=8, ncol=2)
            ttl = f"Tous les marqueurs – {axes_mode}"
        else:
            plot_one(marker_choice, color="C0")
            ttl = f"Marqueur: {marker_choice} – {axes_mode}"

        ax.set_title(ttl)
        if axes_mode != "3D":
            ax.grid(True, ls="--", alpha=0.4)
        else:
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
