#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, csv, re
import numpy as np
import pandas as pd

# ====================== CONFIG ======================
prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep

INPUT = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_left_arm_xyz_m_cor_mod_filt-2.csv"
OUTPUT = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_xyz_m_kf.csv"

FPS    = 100.0
SIGMA_MEAS = 0.05   # m
SIGMA_ACC  = 3     # m/s^2
OUTLIER_P  = 0.001
USE_RTS    = True
# =========================

# ---------- A) LECTURE ROBUSTE -> MultiIndex (marker, axis, unit) ----------
def sniff_delimiter(sample: str):
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        return dialect.delimiter
    except Exception:
        # fallback: privilégie ';' (CSV européen), sinon '\t', sinon ','
        if sample.count(';') >= max(sample.count(','), sample.count('\t')):
            return ';'
        if sample.count('\t') >= sample.count(','):
            return '\t'
        return ','

def detect_decimal(body_text: str):
    # Si on voit des nombres du style 12,34 dans le corps → décimale = ','
    return ',' if re.search(r'\d+,\d+', body_text) else '.'

def read_multiheader_csv_robust(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    lines = raw.splitlines()
    if len(lines) < 4:
        raise RuntimeError("CSV trop court : il faut au moins 3 lignes d’en-tête + données.")

    # détecter séparateur sur la zone d’en-tête
    header_blob = "\n".join(lines[:3])
    sep = sniff_delimiter(header_blob)

    # en-têtes
    h0 = [c.strip() for c in lines[0].split(sep)]
    h1 = [c.strip() for c in lines[1].split(sep)]
    h2 = [c.strip() for c in lines[2].split(sep)]
    if not (len(h0)==len(h1)==len(h2)):
        raise RuntimeError("Les 3 lignes d’en-tête n’ont pas le même nombre de colonnes.")

    # corps
    body_text = "\n".join(lines[3:])
    decimal = detect_decimal(body_text)

    df = pd.read_csv(io.StringIO(body_text), sep=sep, decimal=decimal, engine="python")
    if df.shape[1] != len(h0):
        # dernier filet de sécurité : relire avec un autre sep si nécessaire
        for alt in [';', '\t', ',']:
            if alt == sep: continue
            try:
                df_alt = pd.read_csv(io.StringIO(body_text), sep=alt, decimal=decimal, engine="python")
                if df_alt.shape[1] == len(h0):
                    df = df_alt; sep = alt; break
            except Exception:
                pass

    # assigne le MultiIndex
    df.columns = pd.MultiIndex.from_tuples(list(zip(h0, h1, h2)))
    print(f"[read] OK (sep='{sep}', decimal='{decimal}')  → {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def header_report(df):
    top   = sorted({c[0] for c in df.columns})
    axes  = sorted({str(c[1]) for c in df.columns})
    units = sorted({str(c[2]).lower() for c in df.columns})
    print(f"[headers] markers: {top}")
    print(f"[headers] axes   : {axes}")
    print(f"[headers] units  : {units}")

# ---------- B) Sélection stricte des colonnes XYZ (PRIORITÉ aux brutes) ----------
def get_xyz_cols_strict(df, marker):
    cols = df.columns

    # On veut l’axe “X” exact (pas "X_corr" / "X_kf"), unité m
    def pick_exact(axis_name):
        cands = [c for c in cols if len(c)==3 and c[0]==marker and str(c[1])==axis_name]
        cands_m = [c for c in cands if str(c[2]).lower() in ("m","meter","metre","meters","")]
        return cands_m

    # priorité: X / Y / Z (brut) ; si absent, on tolère X_corr (ou Y_corr, Z_corr) comme mesure
    for ax in ['X','Y','Z']:
        pass
    cx = pick_exact('X')
    cy = pick_exact('Y')
    cz = pick_exact('Z')

    if not cx:
        cx = [c for c in cols if len(c)==3 and c[0]==marker and str(c[1]).lower() in ('x','x_corr') and str(c[2]).lower() in ('m','meter','metre','meters','')]
    if not cy:
        cy = [c for c in cols if len(c)==3 and c[0]==marker and str(c[1]).lower() in ('y','y_corr') and str(c[2]).lower() in ('m','meter','metre','meters','')]
    if not cz:
        cz = [c for c in cols if len(c)==3 and c[0]==marker and str(c[1]).lower() in ('z','z_corr') and str(c[2]).lower() in ('m','meter','metre','meters','')]

    def choose(one_list, label):
        if len(one_list)==1:
            return one_list[0]
        elif len(one_list)>1:
            # si plusieurs, préfère l’exact (X plutôt que X_corr)
            exact = [c for c in one_list if str(c[1]) in ('X','Y','Z')]
            return exact[0] if exact else one_list[0]
        else:
            raise RuntimeError(f"Colonnes absentes pour {marker} {label} (unités m).")

    return choose(cx, 'X'), choose(cy, 'Y'), choose(cz, 'Z')

def list_markers(df):
    return sorted({c[0] for c in df.columns if c[0].lower() != "meta"})

# ---------- C) Temps -> dt ----------
def infer_dt(df, fps=FPS):
    if ("meta","time","s") in df.columns:
        t = pd.to_numeric(df[("meta","time","s")], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(t)):
            idx = np.arange(len(t)); good = np.isfinite(t)
            t[~good] = np.interp(idx[~good], idx[good], t[good]) if good.sum()>=2 else np.arange(len(t))/fps
        dt = np.diff(t, prepend=t[0]); dt[dt<=0] = 1.0/fps
        return dt
    return np.full(len(df), 1.0/fps, float)

# ---------- D) Kalman 3D vitesse constante + gating ----------
def chi2_threshold(p=0.001, dof=3):
    table = {0.05:7.815, 0.01:11.345, 0.005:12.838, 0.001:16.266}
    return table.get(p, 16.266)

def kalman_cv3d(z, dt, r_std=0.005, acc_std=1.0, gate_p=0.001, use_rts=True):
    N = len(z)
    x = np.zeros(6); P = np.eye(6)*1e3
    R = np.eye(3)*(r_std**2)
    x_hist = np.zeros((N,6)); P_hist = np.zeros((N,6,6))
    outlier = np.zeros(N, dtype=bool)
    thr = chi2_threshold(gate_p, 3)

    for k in range(N):
        h = float(max(1e-6, dt[k]))
        A = np.eye(6); A[0,3]=A[1,4]=A[2,5]=h
        G = np.array([[0.5*h*h,0,0],[0,0.5*h*h,0],[0,0,0.5*h*h],[h,0,0],[0,h,0],[0,0,h]])
        Q = G @ (np.eye(3)*(acc_std**2)) @ G.T
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1.0

        # Predict
        x = A@x; P = A@P@A.T + Q

        meas = z[k]
        if np.all(np.isfinite(meas)):
            y = meas - (H@x)
            S = H@P@H.T + R
            try: Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError: Sinv = np.linalg.pinv(S)
            d2 = float(y.T@Sinv@y)
            if d2 <= thr:
                K = P@H.T@Sinv
                x = x + K@y
                P = (np.eye(6)-K@H)@P
            else:
                outlier[k] = True
        x_hist[k] = x; P_hist[k] = P

    if use_rts and N>=2:
        xs = x_hist.copy(); Ps = P_hist.copy()
        for k in range(N-2,-1,-1):
            h = float(max(1e-6, dt[k+1]))
            A = np.eye(6); A[0,3]=A[1,4]=A[2,5]=h
            try: C = P_hist[k] @ A.T @ np.linalg.inv(P_hist[k+1])
            except np.linalg.LinAlgError: C = P_hist[k] @ A.T @ np.linalg.pinv(P_hist[k+1])
            xs[k] = x_hist[k] + C @ (xs[k+1] - A@x_hist[k])
            Ps[k] = P_hist[k] + C @ (Ps[k+1] - P_hist[k+1]) @ C.T
        x_hist = xs
    return x_hist[:, :3], outlier

# ================= RUN =================
df = read_multiheader_csv_robust(INPUT)
header_report(df)

markers = list_markers(df)
print(f"[markers] détectés: {markers}")

dt = infer_dt(df, FPS)
out = df.copy()

for m in markers:
    cx, cy, cz = get_xyz_cols_strict(df, m)   # accepte X ou X_corr si X absent
    X = pd.to_numeric(df[cx], errors="coerce").to_numpy(float)
    Y = pd.to_numeric(df[cy], errors="coerce").to_numpy(float)
    Z = pd.to_numeric(df[cz], errors="coerce").to_numpy(float)

    meas = np.stack([X, Y, Z], axis=1)
    Xk, outl = kalman_cv3d(meas, dt, r_std=SIGMA_MEAS, acc_std=SIGMA_ACC,
                           gate_p=OUTLIER_P, use_rts=USE_RTS)

    out[(m, "X_kf", "m")] = Xk[:,0]
    out[(m, "Y_kf", "m")] = Xk[:,1]
    out[(m, "Z_kf", "m")] = Xk[:,2]
    out[(m, "is_outlier", "")] = outl.astype(int)

out = out.sort_index(axis=1)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
out.to_csv(OUTPUT, index=False)
print(f"✅ Écrit: {OUTPUT}")
