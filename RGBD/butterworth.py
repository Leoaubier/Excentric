# -*- coding: utf-8 -*-
# Filtre Butterworth passe-bas ordre=4, fc=8 Hz, zero-lag (filtfilt) pour un CSV "en m"
# CSV attendu : 3 lignes d'en-tête -> MultiIndex colonnes (marker, axis, unit)
# Colonnes typiques : ('meta','frame','idx'), ('M1','x','m'), ('M1','y','m'), ('M1','z','m'), ...

import os
import numpy as np
import pandas as pd

prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"

INPUT  = prefix + f"/Projet_hand_bike_markerless/Leo/P0/demo_20-10-2025_10_47_44/data/markers_left_arm_xyz_m_cor_mod-2.csv"
OUTPUT = prefix + f"/Projet_hand_bike_markerless/Leo/P0/demo_20-10-2025_10_47_44/data/markers_left_arm_xyz_m_cor_mod_filt-2.csv"

FPS    = 100.0   # Hz
CUTOFF = 8.0     # Hz
ORDER  = 4       # Butterworth order

# --- filtre butterworth zero-lag ---
try:
    from scipy.signal import butter, filtfilt  # type: ignore
except Exception as e:
    raise RuntimeError("SciPy est requis pour filtfilt (pip install scipy)") from e

def butterworth_zero_lag(x: np.ndarray, fs: float, cutoff: float, order: int):
    """Passe-bas Butterworth zero-lag; interpole les NaN avant filtrage."""
    if len(x) == 0:
        return x
    nyq = 0.5 * fs
    Wn = min(0.99, float(cutoff) / nyq)
    if Wn <= 0:
        return x.copy()
    b, a = butter(order, Wn, btype="low", analog=False)

    t = np.arange(len(x))
    xin = x.astype(float)
    bad = ~np.isfinite(xin)
    if bad.any():
        good = ~bad
        if good.sum() >= 2:
            xin[bad] = np.interp(t[bad], t[good], xin[good])
        else:
            return x.copy()

    return filtfilt(b, a, xin, method="pad")

# --- lecture CSV avec 3 lignes d'en-tête ---
#    Si jamais le fichier n'a pas 3 en-têtes, on tombe en mode "simple".
try:
    df = pd.read_csv(INPUT, header=[0, 1, 2])
    assert isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3
except Exception:
    # fallback : lecture classique (1 ligne d'en-tête)
    df = pd.read_csv(INPUT)
    raise RuntimeError(
        "Le fichier ne semble pas avoir 3 lignes d'en-tête MultiIndex. "
        "Vérifie qu'il a été écrit avec (marker, axis, unit)."
    )

out = df.copy()

# --- filtrage par marqueur/axe ---
# niveau 0 = marker (ex: 'M1', 'M2'...), niveau 1 = axis ('x','y','z'), niveau 2 = unit ('m')
lvl0 = df.columns.get_level_values(0)
markers = sorted(set(l for l in lvl0 if l != "meta"))  # on ignore la colonne 'meta'

for mk in markers:
    for axis in ("x", "y", "z"):
        col = (mk, axis, "m")
        if col not in df.columns:
            continue
        series = df[col].to_numpy(dtype=float)
        series_f = butterworth_zero_lag(series, fs=FPS, cutoff=CUTOFF, order=ORDER)
        out[col] = series_f

# --- s'assurer que le dossier cible existe ---
out_dir = os.path.dirname(OUTPUT)
if out_dir and not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# --- écriture du CSV en conservant les 3 lignes d'en-tête ---
# pandas sait écrire un MultiIndex de colonnes en CSV ; le relire nécessitera header=[0,1,2]
out.to_csv(OUTPUT, index=False)

print(f"✅ écrit : {OUTPUT}")
print(f"   fps={FPS} Hz, fc={CUTOFF} Hz, ordre={ORDER}, zero-lag filtfilt")
