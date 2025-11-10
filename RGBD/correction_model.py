#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# ================== CONFIG ==================
prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep

INPUT_CSV  = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_left_arm_xyz_m.csv"
OUTPUT_CSV = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_left_arm_xyz_m_cor_mod-2.csv"

# Liste SANS M1,M2,M3
MARKERS = ['xiph','ster','clavsc','clavac','delt',
           'arm_l','epic_l','larm_l','styl_r','styl_u']

# Graphe de distances (sans M1/M2/M3)
EDGES = [
    # thorax / clavicule / épaule
    ('xiph','ster'), ('ster','clavsc'), ('clavsc','clavac'), ('clavac','delt'),
    # bras gauche (humérus): épaule -> bras -> épicondyle
    ('clavac','arm_l'), ('arm_l','epic_l'),
    # avant-bras (radius/ulna): épicondyle -> larm_l -> styloïdes
    ('epic_l','larm_l'), ('larm_l','styl_r'), ('larm_l','styl_u'),
    # poignet (largeur) et cohérence
    ('styl_r','styl_u'),
]

# ancres thorax/épaule (stabilise globalement)
ANCHORS = ['clavsc','clavac','ster']

N_ITERS = 20
OBS_ALPHA = 0.05
ANCHOR_ALPHA = 0.20
# ============================================

def _get_xyz(df, name):
    cols = df.columns
    def pick_axis(ax):
        cands = [c for c in cols if isinstance(c, tuple) and len(c)==3
                 and c[0]==name and str(c[1]).lower()==ax]
        if not cands: return None
        pref = [c for c in cands if str(c[2]).lower() in ('m','meter','metre','meters','')]
        return pref[0] if pref else cands[0]
    cx = pick_axis('x'); cy = pick_axis('y'); cz = pick_axis('z')
    if any(c is None for c in (cx,cy,cz)): return None, None, None
    X = pd.to_numeric(df[cx], errors="coerce").to_numpy(float)
    Y = pd.to_numeric(df[cy], errors="coerce").to_numpy(float)
    Z = pd.to_numeric(df[cz], errors="coerce").to_numpy(float)
    return X, Y, Z

def _stack_xyz(X, Y, Z):
    return np.vstack([X, Y, Z]).T

def estimate_rest_lengths(positions, edges):
    L = {}
    for a,b in edges:
        Pa, Pb = positions.get(a), positions.get(b)
        if Pa is None or Pb is None:
            L[(a,b)] = np.nan; continue
        good = np.all(np.isfinite(Pa),1) & np.all(np.isfinite(Pb),1)
        if not np.any(good):
            L[(a,b)] = np.nan; continue
        dist = np.linalg.norm(Pb[good]-Pa[good], axis=1)
        L[(a,b)] = np.median(dist)
    return L

def project_edge(Pa, Pb, L):
    v = Pb - Pa
    n = np.linalg.norm(v)
    if not np.isfinite(L) or L<=0 or n<1e-12:
        return Pa, Pb
    corr = (L - n)
    dir = v / n
    return Pa - 0.5*corr*dir, Pb + 0.5*corr*dir

def correct_frame_pbd(obs, edges, rest_len, anchors, obs_alpha=0.2, anchor_alpha=0.5, n_iters=20):
    names = list(obs.keys())
    P = {k: (obs[k].copy() if np.all(np.isfinite(obs[k])) else np.array([np.nan,np.nan,np.nan],float))
         for k in names}
    # init des NaN par moyenne des voisins si possible
    for k in names:
        if not np.all(np.isfinite(P[k])):
            neigh = [n for (a,b) in edges for n in ([a,b]) if (a==k or b==k)]
            neigh = sorted(set(neigh))
            vals = [obs[n] for n in neigh if n in obs and np.all(np.isfinite(obs[n]))]
            P[k] = np.mean(vals,0) if vals else np.zeros(3)

    for _ in range(int(max(1,n_iters))):
        # projection de distance
        for (a,b) in edges:
            L = rest_len.get((a,b), np.nan)
            if not np.isfinite(L): L = rest_len.get((b,a), np.nan)
            Pa_new, Pb_new = project_edge(P[a], P[b], L)
            wa = 0.0 if a in anchors else 1.0
            wb = 0.0 if b in anchors else 1.0
            P[a] = P[a] if wa==0.0 else Pa_new
            P[b] = P[b] if wb==0.0 else Pb_new
        # rappel vers observations
        for k in names:
            if np.all(np.isfinite(obs[k])):
                alpha = anchor_alpha if k in anchors else obs_alpha
                if alpha>0:
                    P[k] = (1.0-alpha)*P[k] + alpha*obs[k]
    return P

def main():
    df = pd.read_csv(INPUT_CSV, header=[0,1,2])
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels!=3:
        raise RuntimeError("CSV attendu avec 3 lignes d’en-tête (marker/axis/unit).")

    pos = {}
    for m in MARKERS:
        X,Y,Z = _get_xyz(df, m)
        pos[m] = None if any(v is None for v in (X,Y,Z)) else _stack_xyz(X,Y,Z)

    rest_len = estimate_rest_lengths(pos, EDGES)
    print("[model] longueurs médianes (m):")
    for (a,b), L in rest_len.items():
        print(f"  {a:>7s} - {b:<7s}: {L:.4f}" if np.isfinite(L) else f"  {a:>7s} - {b:<7s}: NaN")

    N = len(df)
    corr = {m: np.full((N,3), np.nan, float) for m in MARKERS}
    for i in range(N):
        obs_i = {m: (pos[m][i] if pos[m] is not None else np.array([np.nan,np.nan,np.nan],float))
                 for m in MARKERS}
        Pc = correct_frame_pbd(obs_i, EDGES, rest_len, ANCHORS,
                               obs_alpha=OBS_ALPHA, anchor_alpha=ANCHOR_ALPHA, n_iters=N_ITERS)
        for m in MARKERS:
            corr[m][i] = Pc[m]

    out = df.copy()
    for m in MARKERS:
        Pm = corr[m]
        out[(m,'X_corr','m')] = Pm[:,0]
        out[(m,'Y_corr','m')] = Pm[:,1]
        out[(m,'Z_corr','m')] = Pm[:,2]

    out = out.sort_index(axis=1)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV corrigé écrit: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
