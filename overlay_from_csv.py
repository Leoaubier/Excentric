#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#permet de compiler une video a partir des images et des positions de markers contenus dans un csv
#permet une visualisation apres une correction des positions


import os, glob, re, math, json
import cv2
import numpy as np
import pandas as pd

# =============== CONFIG ===============
prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep
ZED_JSON   = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}config_camera_files{sep}config_camera_P0.json"      # intrinsics ZED
CSV_PATH    = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_xyz_m_kf.csv"
IMAGE_DIR   = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44"     # contient color_####.png
IMAGE_GLOB  = "color_*.png"                           # motif
# Si vous avez des indices entiers dans le nom de fichier:
#   "color_12.png" => index 12
FRAME_REGEX = re.compile(r"(\d+)")                     # pour extraire le numéro

OUT_VIDEO   = f"{sep}home{sep}mickael{sep}Documents{sep}Leo{sep}overlay_from_csv_test_3_filt-2.mp4"
FPS_OUT     = 25





# Marqueurs à tracer (None -> tous ceux du CSV)
MARKERS_TO_DRAW = None  # ex: ['arm_l','epic_l','styl_r','styl_u']

# Si les images ont été recadrées/redimensionnées, ajuster intrinsics:
CROP   = None     # [x0,y0,x1,y1] ou None
ASPECT = 1.0      # facteur d’échelle (resize), ex 0.5 ou 1.0

# Bornes de frames (selon "meta/frame/idx" s’il existe; sinon selon l’ordre des fichiers)
FRAME_START = None
FRAME_END   = None
# =============================================

_frame_num_re = re.compile(r"(\d+)")

def load_multiheader_xyz_m(path):
    df = pd.read_csv(path, header=[0,1,2])
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 3:
        raise RuntimeError("CSV attendu avec 3 en-têtes (marker / axis / unit).")
    return df

def available_markers(df):
    return sorted({c[0] for c in df.columns if c[0] != "meta"})

def get_axis_col(df, marker, axis_letter):
    cols = df.columns
    cands = [c for c in cols if len(c)==3 and c[0]==marker and str(c[1]).lower()==axis_letter]
    if not cands:
        return None
    # préfère l’unité 'm' si dispo
    pref = [c for c in cands if str(c[2]).lower() in ("m","meter","metre","meters","")]
    return pref[0] if pref else cands[0]

def get_xyz_cols(df, marker):
    cx = get_axis_col(df, marker, 'x_kf')
    cy = get_axis_col(df, marker, 'y_kf')
    cz = get_axis_col(df, marker, 'z_kf')
    return cx, cy, cz

def build_frame_index(img_dir, pattern):
    paths = sorted(glob.glob(os.path.join(img_dir, pattern)))
    frames = {}
    for p in paths:
        m = _frame_num_re.search(os.path.basename(p))
        if m: frames[int(m.group(1))] = p
    return frames, paths

def normalize_out_path(path):
    root, ext = os.path.splitext(path)
    if ext == "":
        path = path + ".mp4"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def open_video_writer(path, wh, fps):
    W, H = wh
    path = normalize_out_path(path)
    trials = [
        ("mp4v", os.path.splitext(path)[0] + ".mp4"),
        ("avc1", os.path.splitext(path)[0] + ".mp4"),
        ("XVID", os.path.splitext(path)[0] + ".avi"),
        ("MJPG", os.path.splitext(path)[0] + ".avi"),
    ]
    for fourcc_tag, ptry in trials:
        vw = cv2.VideoWriter(ptry, cv2.VideoWriter_fourcc(*fourcc_tag), float(fps), (W, H), True)
        if vw.isOpened():
            print(f"[VideoWriter] OK → {ptry} ({fourcc_tag}, {W}x{H}@{fps})")
            return vw, ptry
    raise RuntimeError("Impossible d’ouvrir un VideoWriter (essayez .avi/MJPG).")

# ---------- Intrinsics ZED ----------
def load_intrinsics(json_path):
    if not json_path or not os.path.isfile(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fx = fy = cx = cy = None
    # format “camera.depth_*”
    try:
        fx, fy = data["camera"]["depth_fx_fy"]
        cx, cy = data["camera"]["depth_ppx_ppy"]
    except Exception:
        pass
    # format “left_color_*”
    if fx is None or fy is None or cx is None or cy is None:
        try:
            fx, fy = data["left_color_fx_fy"]
            cx, cy = data["left_color_ppx_ppy"]
        except Exception:
            pass
    # fallback plat
    if fx is None: fx = data.get("fx") or data.get("depth_fx") or data.get("camera", {}).get("fx")
    if fy is None: fy = data.get("fy") or data.get("depth_fy") or data.get("camera", {}).get("fy")
    if cx is None: cx = data.get("cx") or data.get("depth_cx") or data.get("camera", {}).get("cx")
    if cy is None: cy = data.get("cy") or data.get("depth_cy") or data.get("camera", {}).get("cy")

    if None in (fx, fy, cx, cy):
        return None
    return [float(fx), float(fy), float(cx), float(cy)]

def adjust_intrinsics_for_crop_aspect(K, crop, aspect):
    if K is None: return None
    fx, fy, cx, cy = K
    if crop is not None:
        x0, y0, _, _ = crop
        cx -= x0; cy -= y0
    if aspect and aspect != 1.0:
        fx *= aspect; fy *= aspect; cx *= aspect; cy *= aspect
    return [fx, fy, cx, cy]

def project_xyz_to_uv(X, Y, Z, K):
    """Projette (X,Y,Z) (m) -> (u,v) (px) en supposant repère caméra. Ignorer si Z<=0."""
    fx, fy, cx, cy = K
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return u, v

def main():
    # Lire CSV
    df = load_multiheader_xyz_m(CSV_PATH)
    markers = MARKERS_TO_DRAW or available_markers(df)
    print(f"[CSV] {len(markers)} marqueurs à traiter")

    # Colonnes XYZ en m
    xyz_cols = {}
    for m in markers:
        cx, cy, cz = get_xyz_cols(df, m)
        if None in (cx,cy,cz):
            print(f"[WARN] {m}: colonnes (X,Y,Z,m) introuvables → ignoré")
            continue
        xyz_cols[m] = (cx, cy, cz)
    if not xyz_cols:
        raise RuntimeError("Aucun marqueur avec colonnes (X,Y,Z) en m trouvées.")

    # Frames depuis CSV (si dispo)
    has_meta_frame = ("meta","frame","idx") in df.columns
    if has_meta_frame:
        frame_series = pd.to_numeric(df[("meta","frame","idx")], errors="coerce").astype("Int64")
    else:
        frame_series = pd.Series(np.arange(len(df)), dtype="Int64")

    # Images
    frame2path, paths_fallback = build_frame_index(IMAGE_DIR, IMAGE_GLOB)
    use_map = has_meta_frame and len(frame2path) > 0
    print("[IMG] mapping via meta/frame/idx" if use_map else "[IMG] ordre des fichiers (fallback)")

    # Intrinsics
    K = load_intrinsics(ZED_JSON)
    if K is None:
        raise RuntimeError("Intrinsèques ZED introuvables/incomplets dans le JSON.")
    K = adjust_intrinsics_for_crop_aspect(K, CROP, ASPECT)
    fx, fy, cx, cy = K
    print(f"[K] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # Liste des frames à traiter
    todo = []
    if use_map:
        for i in range(len(df)):
            fr = frame_series.iat[i]
            if pd.isna(fr):
                continue
            fr = int(fr)
            if (FRAME_START is not None and fr < FRAME_START) or (FRAME_END is not None and fr > FRAME_END):
                continue
            path = frame2path.get(fr, None)
            if path: todo.append((i, fr, path))
    else:
        for i, p in enumerate(paths_fallback):
            if (FRAME_START is not None and i < FRAME_START) or (FRAME_END is not None and i > FRAME_END):
                continue
            m = _frame_num_re.search(os.path.basename(p))
            fr = int(m.group(1)) if m else i
            todo.append((i, fr, p))
    if not todo:
        raise RuntimeError("Aucune frame à traiter avec les bornes actuelles.")

    # Prépare vidéo
    img0 = cv2.imread(todo[0][2], cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Impossible de lire {todo[0][2]}")
    H, W = img0.shape[:2]
    vw, out_path = open_video_writer(OUT_VIDEO, (W, H), FPS_OUT)

    # Couleurs pour les marqueurs
    colors = [
        (40,255,40), (0,200,255), (255,180,0), (180,0,255),
        (255,60,60), (60,200,255), (150,150,255), (0,140,255),
        (255,120,180), (120,255,120), (220,220,0), (255,0,120),
        (128,128,255), (255,128,64), (64,255,128), (255,64,192)
    ]

    for row_i, fr, path in todo:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        # (Optionnel) si tes images ont été effectivement recadrées/redimensionnées en amont,
        # il faut que CROP/ASPECT ici reflètent la même opération — mais on ne touche pas à img ici.

        k = 0
        for m, (cx_col, cy_col, cz_col) in xyz_cols.items():
            X = pd.to_numeric(df.at[row_i, cx_col], errors="coerce")
            Y = pd.to_numeric(df.at[row_i, cy_col], errors="coerce")
            Z = pd.to_numeric(df.at[row_i, cz_col], errors="coerce")
            if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z) and Z > 0):
                continue

            u, v = project_xyz_to_uv(float(X), float(Y), float(Z), (fx, fy, cx, cy))
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                col = colors[k % len(colors)]; k += 1
                cv2.circle(img, (ui, vi), 6, col, -1, lineType=cv2.LINE_AA)
                cv2.putText(img, str(m), (ui+7, vi+7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.putText(img, f"frame: {fr}", (12, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame: {fr}", (12, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        vw.write(img)

    vw.release()
    print(f"✅ Vidéo écrite: {out_path}")

if __name__ == "__main__":
    main()
