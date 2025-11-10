
"""
Filtrage passe-bas (Butterworth ordre 4, 8 Hz, zéro-lag) de trajectoires de marqueurs.
Produit pour chaque marqueur :
  - un plot (X,Y) avant/après
  - un plot (Y,Z) avant/après
Sauvegarde un CSV avec colonnes filtrées suffixées *_filt.

Compatible CSV :
  1) Multi-en-têtes (ligne1: marker, ligne2: X/Y/Z, ligne3: unité 'm' ou 'mm')
  2) Colonnes plates: marker_X[_m], marker_Y[_m], marker_Z[_m]

Dépendances : pandas, numpy, matplotlib, scipy
"""

import os
import io
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

# ===================== CONFIG =====================

prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep

CSV_IN    = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_left_arm_xyz_m.csv"
CSV_OUT  = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_left_arm_xyz_m_filt_2D.csv"
PLOTS_DIR = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data"      # dossier des PNG

FPS    = 100.0     # Hz
CUTOFF = 8.0       # Hz
ORDER  = 4
SHOW_FIG = False   # True pour afficher à l'écran (en plus des PNG)
# ==================================================


# ------------ utils lecture CSV (multi-header tolérant) ------------
def _sniff_delim(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        return dialect.delimiter
    except Exception:
        if sample.count(';') >= max(sample.count(','), sample.count('\t')): return ';'
        if sample.count('\t') >= sample.count(','): return '\t'
        return ','

def _detect_decimal(text: str) -> str:
    commas = len(re.findall(r'\b\d+,\d+\b', text))
    dots   = len(re.findall(r'\b\d+\.\d+\b', text))
    return ',' if commas > dots else '.'

def read_csv_flexible(path: str) -> pd.DataFrame:
    """Lit un CSV soit en MultiIndex (3 lignes d'en-tête), soit en simple header."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    lines = raw.splitlines()
    if len(lines) < 2:
        raise RuntimeError("CSV trop court.")
    sep = _sniff_delim("\n".join(lines[:3]))
    decimal = _detect_decimal("\n".join(lines[3:])) if len(lines) >= 4 else '.'

    # tentative multiheader (3 lignes)
    try:
        df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=decimal, header=[0,1,2], engine="python")
        # vérifie que les niveaux semblent corrects
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
            return df
    except Exception:
        pass

    # fallback: simple header
    df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=decimal, engine="python")
    return df


# ------------ sélection colonnes par marqueur ------------
def list_markers(df: pd.DataFrame):
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
        # niveau 0 = nom marqueur (on ignore éventuels 'meta')
        return [m for m in sorted({c[0] for c in df.columns}) if m.lower() != "meta"]
    else:
        # colonnes plates : on extrait les préfixes avant _X/_Y/_Z
        mks = set()
        for c in df.columns:
            m = re.match(r"(.+?)_(X|Y|Z)(?:_m)?$", str(c))
            if m:
                mks.add(m.group(1))
        return sorted(mks)

def get_cols_for_marker(df: pd.DataFrame, marker: str):
    """
    Retourne les noms de colonnes pour X,Y,Z et facteur d'unité vers mètres (1 ou 1/1000).
    Accepte :
      - MultiIndex: (marker, X|Y|Z, unité m|mm|...)
      - Flat: marker_X(_m), marker_Y(_m), marker_Z(_m)
    """
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
        cols = df.columns

        def pick(ax):
            # priorité à unité 'm' si dispo, sinon 'mm'
            cands = [c for c in cols if c[0]==marker and str(c[1]).upper()==ax.upper()]
            if not cands:
                # tolère variantes X_corr/X_kf : on prend la première qui commence par X/y/z
                cands = [c for c in cols if c[0]==marker and str(c[1]).upper().startswith(ax.upper())]
                if not cands:
                    return None, 1.0
            # choisit 'm' si présent
            m_cols = [c for c in cands if str(c[2]).lower() in ("m","meter","metre","meters","")]
            if m_cols:
                return m_cols[0], 1.0
            mm_cols = [c for c in cands if str(c[2]).lower()=="mm"]
            if mm_cols:
                return mm_cols[0], 1/1000.0
            # sinon défaut m
            return cands[0], 1.0

        cx, fx = pick("X")
        cy, fy = pick("Y")
        cz, fz = pick("Z")
        return cx, fx, cy, fy, cz, fz

    else:
        # colonnes plates
        def pick(ax):
            # préférer *_m si présent
            name_m = f"{marker}_{ax}_m"
            name   = f"{marker}_{ax}"
            if name_m in df.columns:
                return name_m, 1.0
            if name in df.columns:
                return name, 1.0  # on suppose déjà en m; change à 1/1000 si c'est des mm
            # tolère X_corr/X_kf
            for suffix in ("_corr","_kf"):
                cand = f"{marker}_{ax}{suffix}"
                if cand in df.columns:
                    return cand, 1.0
            return None, 1.0

        cx, fx = pick("X")
        cy, fy = pick("Y")
        cz, fz = pick("Z")
        return cx, fx, cy, fy, cz, fz


# ------------ filtre passe-bas ------------
def butter_lowpass_filtfilt(x: np.ndarray, fs: float, cutoff: float, order: int):
    if x is None:
        return None
    x = np.asarray(x, float)
    if len(x) < max(15, 3*order):
        return x  # trop court pour un filtrage fiable
    nyq = 0.5 * fs
    wn = min(0.99, float(cutoff)/nyq)
    b, a = butter(order, wn, btype="low", analog=False)
    # interpoler NaN si besoin
    t = np.arange(len(x))
    bad = ~np.isfinite(x)
    if bad.any():
        good = ~bad
        if good.sum() >= 2:
            x[bad] = np.interp(t[bad], t[good], x[good])
        else:
            return x
    return filtfilt(b, a, x, method="pad")


# ------------ pipeline principal ------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = read_csv_flexible(CSV_IN).copy()
    markers = list_markers(df)
    if not markers:
        raise RuntimeError("Aucun marqueur détecté dans le CSV.")

    out = df.copy()

    for mk in markers:
        cx, fx, cy, fy, cz, fz = get_cols_for_marker(df, mk)
        # (X,Y)
        if cx is not None and cy is not None:
            X = df[cx].to_numpy(dtype=float) * fx
            Y = df[cy].to_numpy(dtype=float) * fy
            Xf = butter_lowpass_filtfilt(X, FPS, CUTOFF, ORDER)
            Yf = butter_lowpass_filtfilt(Y, FPS, CUTOFF, ORDER)

            # Ajoute colonnes filtrées (suffixe _filt)
            col_x_f = (f"{mk}_X_filt" if not isinstance(df.columns, pd.MultiIndex) else (mk,"X_filt","m"))
            col_y_f = (f"{mk}_Y_filt" if not isinstance(df.columns, pd.MultiIndex) else (mk,"Y_filt","m"))
            out[col_x_f] = Xf
            out[col_y_f] = Yf

            # Plot XY (trajectoire)
            plt.figure(figsize=(6,6))
            plt.plot(X,  Y,  lw=1, alpha=0.5, label="Brut")
            plt.plot(Xf, Yf, lw=2, label="Filtré")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel("X (m)"); plt.ylabel("Y (m)")
            plt.title(f"{mk} — Trajectoire (X,Y)")
            plt.legend()
            plt.grid(True, ls=':')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{mk}_XY.png"), dpi=150)
            if SHOW_FIG: plt.show()
            else: plt.close()

        # (Y,Z)
        if cy is not None and cz is not None:
            Y = df[cy].to_numpy(dtype=float) * fy
            Z = df[cz].to_numpy(dtype=float) * fz
            Yf = butter_lowpass_filtfilt(Y, FPS, CUTOFF, ORDER)
            Zf = butter_lowpass_filtfilt(Z, FPS, CUTOFF, ORDER)

            col_y_f2 = (f"{mk}_Y_filt" if not isinstance(df.columns, pd.MultiIndex) else (mk,"Y_filt","m"))
            col_z_f  = (f"{mk}_Z_filt" if not isinstance(df.columns, pd.MultiIndex) else (mk,"Z_filt","m"))
            out[col_y_f2] = Yf  # si déjà écrit plus haut, cela remplace par la même version
            out[col_z_f]  = Zf

            # Plot YZ (trajectoire)
            plt.figure(figsize=(6,6))
            plt.plot(Y,  Z,  lw=1, alpha=0.5, label="Brut")
            plt.plot(Yf, Zf, lw=2, label="Filtré")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel("Y (m)"); plt.ylabel("Z (m)")
            plt.title(f"{mk} — Trajectoire (Y,Z)")
            plt.legend()
            plt.grid(True, ls=':')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{mk}_YZ.png"), dpi=150)
            if SHOW_FIG: plt.show()
            else: plt.close()

    # Sauvegarde CSV plat (simple) pour compatibilité
    if isinstance(out.columns, pd.MultiIndex):
        # aplatit les colonnes multi-index "mk|ax|unit"
        out_flat = out.copy()
        out_flat.columns = ["|".join([str(x) for x in c]) for c in out.columns]
        out_flat.to_csv(CSV_OUT, index=False)
    else:
        out.to_csv(CSV_OUT, index=False)

    print(f"[OK] CSV filtré : {CSV_OUT}")
    print(f"[OK] Plots par marqueur dans : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
