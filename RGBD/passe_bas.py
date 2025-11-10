#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd

# ==== SciPy pour Butterworth ====
try:
    from scipy.signal import butter, filtfilt
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


APP_TITLE = "Filtrage CSV – Marqueurs (Butterworth + Hampel)"

# Groupes de colonnes candidates (priorité décroissante)
PREFERRED_NUM_COLS = [
    ["X_m", "Y_m", "Z_m"],                 # 3D caméra (m)
    ["x_px_raw", "y_px_raw", "depth_mm"],  # 2D pixels + profondeur (mm)
    ["x_px_raw", "y_px_raw"],              # 2D pixels
]


# ---------- Filtres ----------
def hampel_filter_1d(x, window=7, n_sigma=3.0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = int(window)
    if w < 3:
        return x.copy()
    half = w // 2
    y = x.copy()
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        vals = x[lo:hi]
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if not np.isfinite(mad):
            continue
        thresh = n_sigma * 1.4826 * mad
        if np.isfinite(x[i]) and np.abs(x[i] - med) > thresh:
            y[i] = med
    return y


def butter_lowpass_filtfilt(x, fs, cutoff, order=4):
    """Passe-bas Butterworth sans déphasage. Fallback moyenne glissante si SciPy absent."""
    if not HAVE_SCIPY:
        # moyenne glissante ~ fs/cutoff
        win = max(3, int(round(fs / max(cutoff, 0.1))))
        if win % 2 == 0:
            win += 1
        xin = x.astype(float)
        if not np.isfinite(xin).any():
            return xin
        xin = np.nan_to_num(xin, nan=np.nanmedian(xin))
        kernel = np.ones(win) / win
        return np.convolve(xin, kernel, mode="same")

    if cutoff <= 0 or fs <= 0:
        return x.copy()

    nyq = 0.5 * fs
    Wn = min(0.99, float(cutoff) / nyq)
    if Wn <= 0:
        return x.copy()

    b, a = butter(order, Wn, btype="low", analog=False)

    # interp NaN avant filtfilt
    t = np.arange(len(x))
    xin = x.astype(float)
    bad = ~np.isfinite(xin)
    if bad.any():
        good = ~bad
        if good.sum() >= 2:
            xin[bad] = np.interp(t[bad], t[good], xin[good])
        else:
            return x.copy()

    y = filtfilt(b, a, xin, method="pad")
    return y


# ---------- Utilitaires ----------
def choose_numeric_cols(df, mode):
    """Choisir colonnes à filtrer selon mode choisi."""
    numeric_cols = [c for c in df.columns
                    if (pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]))
                    and not pd.api.types.is_bool_dtype(df[c])]

    if mode == "xyz":
        group = ["X_m", "Y_m", "Z_m"]
        return [c for c in group if c in numeric_cols]
    elif mode == "pxd":
        group = ["x_px_raw", "y_px_raw", "depth_mm"]
        return [c for c in group if c in numeric_cols]
    elif mode == "px":
        group = ["x_px_raw", "y_px_raw"]
        return [c for c in group if c in numeric_cols]
    else:  # auto
        for group in PREFERRED_NUM_COLS:
            cand = [c for c in group if c in numeric_cols]
            if len(cand) >= 2:
                return cand
        return numeric_cols  # secours


def list_markers(df):
    """Retourne (key, values) où key est 'marker_name' ou 'marker_id' ou None."""
    if "marker_name" in df.columns:
        vals = sorted(df["marker_name"].dropna().unique().tolist())
        return "marker_name", vals
    elif "marker_id" in df.columns:
        vals = sorted(df["marker_id"].dropna().unique().tolist())
        return "marker_id", vals
    else:
        return None, []


def filter_one_file(csv_path, marker_key, selected_markers, cols_mode,
                    fps, cutoff, order, use_hampel, hampel_win, hampel_sigma, out_suffix, log_fn=print):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log_fn(f"❌ Lecture échouée: {csv_path} ({e})")
        return None

    cols = choose_numeric_cols(df, cols_mode)
    if not cols:
        log_fn(f"⚠️ {os.path.basename(csv_path)} : aucune colonne numérique pertinente. Copie brute.")
        out_path = os.path.splitext(csv_path)[0] + out_suffix + ".csv"
        try:
            df.to_csv(out_path, index=False)
            return out_path
        except Exception as e:
            log_fn(f"❌ Écriture impossible : {e}")
            return None

    # Masque des lignes à filtrer (par marqueur)
    if marker_key and selected_markers:
        mask = df[marker_key].isin(selected_markers)
    else:
        mask = np.ones(len(df), dtype=bool)

    out = df.copy()

    # Grouper par marqueur pour conserver la chronologie (frame)
    if marker_key and marker_key in df.columns:
        groups = df.groupby(marker_key)
    else:
        groups = [(None, df)]

    for gname, gdf in groups:
        if marker_key and selected_markers and gname not in set(selected_markers):
            continue

        idxs = gdf.index
        gwork = gdf.copy()
        if "frame" in gwork.columns:
            gwork = gwork.sort_values("frame")

        for c in cols:
            if c not in gwork.columns:
                continue

            x = gwork[c].to_numpy(dtype=float)
            local_mask = mask[idxs].copy()
            if not local_mask.any():
                continue

            x_sel = x[local_mask]

            if use_hampel:
                x_sel = hampel_filter_1d(x_sel, window=hampel_win, n_sigma=hampel_sigma)

            x_sel = butter_lowpass_filtfilt(x_sel, fs=fps, cutoff=cutoff, order=order)

            x_new = x.copy()
            x_new[local_mask] = x_sel
            out.loc[idxs, c] = x_new

    out_path = os.path.splitext(csv_path)[0] + out_suffix + ".csv"
    try:
        out.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        log_fn(f"❌ Écriture impossible vers {out_path}: {e}")
        return None


# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x640")

        self.input_dir = tk.StringVar(value="")
        self.glob_pattern = tk.StringVar(value="*.csv")
        self.out_suffix = tk.StringVar(value="_filtered")

        self.cols_mode = tk.StringVar(value="auto")   # auto / xyz / pxd / px
        self.fps = tk.DoubleVar(value=100.0)
        self.cutoff = tk.DoubleVar(value=6.0)
        self.order = tk.IntVar(value=4)

        self.use_hampel = tk.BooleanVar(value=True)
        self.hampel_win = tk.IntVar(value=7)
        self.hampel_sigma = tk.DoubleVar(value=3.0)

        self.marker_key = None
        self.marker_values = []

        self._build_ui()

    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="Dossier CSV :").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.input_dir, width=60).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Parcourir…", command=self.browse_dir).pack(side=tk.LEFT)

        ttk.Label(top, text="Motif :").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=self.glob_pattern, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Charger", command=self.load_first).pack(side=tk.LEFT, padx=6)

        # Options panel
        left = ttk.LabelFrame(self, text="Options de filtrage"); left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        # Colonnes / source
        ttk.Label(left, text="Colonnes à filtrer").pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Radiobutton(left, text="Auto (XYZ si dispo, sinon px+depth, sinon px)", variable=self.cols_mode, value="auto").pack(anchor="w", padx=12)
        ttk.Radiobutton(left, text="Caméra XYZ (X_m,Y_m,Z_m)", variable=self.cols_mode, value="xyz").pack(anchor="w", padx=12)
        ttk.Radiobutton(left, text="Pixels + profondeur (x,y,depth_mm)", variable=self.cols_mode, value="pxd").pack(anchor="w", padx=12)
        ttk.Radiobutton(left, text="Pixels 2D (x,y)", variable=self.cols_mode, value="px").pack(anchor="w", padx=12)

        # Paramètres filtre
        frm = ttk.Frame(left); frm.pack(fill=tk.X, padx=8, pady=(8, 2))
        ttk.Label(frm, text="fps").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fps, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(frm, text="cutoff (Hz)").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.cutoff, width=8).grid(row=1, column=1, padx=4)
        ttk.Label(frm, text="ordre").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.order, width=8).grid(row=2, column=1, padx=4)

        # Hampel
        ttk.Checkbutton(left, text="Hampel anti-pics", variable=self.use_hampel).pack(anchor="w", padx=8, pady=(8, 2))
        hrow = ttk.Frame(left); hrow.pack(fill=tk.X, padx=8)
        ttk.Label(hrow, text="fenêtre").grid(row=0, column=0, sticky="w")
        ttk.Entry(hrow, textvariable=self.hampel_win, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(hrow, text="sigma").grid(row=0, column=2, sticky="w", padx=(8,0))
        ttk.Entry(hrow, textvariable=self.hampel_sigma, width=8).grid(row=0, column=3, padx=4)

        # Suffixe sortie
        ttk.Label(left, text="Suffixe de sortie").pack(anchor="w", padx=8, pady=(10, 2))
        ttk.Entry(left, textvariable=self.out_suffix, width=14).pack(anchor="w", padx=8)

        # Marqueurs
        right = ttk.LabelFrame(self, text="Marqueurs à filtrer"); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.marker_list = tk.Listbox(right, selectmode=tk.EXTENDED, exportselection=False, height=18)
        self.marker_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        ttk.Button(right, text="Tout sélectionner", command=self.select_all_markers).pack(side=tk.LEFT, padx=8, pady=(0,8))
        ttk.Button(right, text="Tout désélectionner", command=self.clear_selection).pack(side=tk.LEFT, padx=8, pady=(0,8))

        # Bottom buttons + log
        bottom = ttk.Frame(self); bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        ttk.Button(bottom, text="Lancer le filtrage", command=self.run_batch).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Quitter", command=self.destroy).pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(self, text="Journal"); log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=8, pady=(0,8))
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ---- callbacks ----
    def browse_dir(self):
        d = filedialog.askdirectory(title="Choisir le dossier CSV")
        if d:
            self.input_dir.set(d)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def load_first(self):
        folder = self.input_dir.get().strip()
        pattern = self.glob_pattern.get().strip() or "*.csv"
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Dossier manquant", "Sélectionne un dossier CSV valide.")
            return
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not files:
            messagebox.showwarning("Aucun fichier", f"Aucun CSV trouvé dans {folder} avec le motif {pattern}.")
            return
        try:
            df = pd.read_csv(files[0])
        except Exception as e:
            messagebox.showerror("Erreur", f"Lecture du premier CSV impossible:\n{e}")
            return

        self.marker_key, self.marker_values = list_markers(df)
        self.marker_list.delete(0, tk.END)
        if self.marker_key is None:
            self.marker_list.insert(tk.END, "(Aucune colonne marker_name/marker_id) → Filtrage de TOUT le fichier")
        else:
            for m in self.marker_values:
                self.marker_list.insert(tk.END, str(m))
        self.log(f"Chargé {os.path.basename(files[0])} – clé marqueur: {self.marker_key or 'None'} – {len(self.marker_values)} marqueurs.")

    def get_selected_markers(self):
        if self.marker_key is None:
            return None  # filtrer tout
        sel = [self.marker_list.get(i) for i in self.marker_list.curselection()]
        if not sel:
            # si rien sélectionné → tout
            return None
        if self.marker_key == "marker_id":
            # convertir en int si possible
            out = []
            for s in sel:
                try:
                    out.append(int(s))
                except Exception:
                    pass
            return out or None
        return sel

    def select_all_markers(self):
        self.marker_list.select_set(0, tk.END)

    def clear_selection(self):
        self.marker_list.select_clear(0, tk.END)

    def run_batch(self):
        folder = self.input_dir.get().strip()
        pattern = self.glob_pattern.get().strip() or "*.csv"
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Dossier manquant", "Sélectionne un dossier CSV valide.")
            return
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not files:
            messagebox.showwarning("Aucun fichier", f"Aucun CSV trouvé dans {folder} avec le motif {pattern}.")
            return

        # paramètres
        cols_mode = self.cols_mode.get()
        fps = float(self.fps.get() or 100.0)
        cutoff = float(self.cutoff.get() or 6.0)
        order = int(self.order.get() or 4)
        order = max(1, min(order, 8))
        use_hampel = bool(self.use_hampel.get())
        hampel_win = int(self.hampel_win.get() or 7)
        if hampel_win % 2 == 0:
            hampel_win += 1
        hampel_win = max(3, hampel_win)
        hampel_sigma = float(self.hampel_sigma.get() or 3.0)
        out_suffix = self.out_suffix.get().strip() or "_filtered"

        selected_markers = self.get_selected_markers()

        if not HAVE_SCIPY:
            self.log("⚠️ SciPy indisponible : fallback moyenne glissante (moins fidèle).")

        ok = 0
        for p in files:
            self.log(f"→ Traitement: {os.path.basename(p)}")
            out = filter_one_file(
                p, self.marker_key, selected_markers, cols_mode,
                fps, cutoff, order, use_hampel, hampel_win, hampel_sigma, out_suffix,
                log_fn=self.log
            )
            if out:
                self.log(f"   ✅ Écrit: {out}")
                ok += 1
            else:
                self.log("   ❌ Échec")

        self.log(f"\nTerminé. {ok}/{len(files)} fichiers écrits.")

def main():
    app = App()
    app.title(APP_TITLE + ("  (SciPy OK)" if HAVE_SCIPY else "  (sans SciPy)"))
    app.mainloop()

if __name__ == "__main__":
    main()
