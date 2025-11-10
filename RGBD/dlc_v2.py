#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, cv2, time, csv, re, json
import numpy as np
from dlclive import DLCLive, Processor

try:
    import yaml
except Exception:
    yaml = None  # si yaml pas installé, pip install pyyaml

def load_dlc_marker_names(model_dir: str):
    """
    Retourne la liste ALL_MARKERS dans l'ordre d'entraînement DLC.
    Cherche 'pose_cfg.yaml' ou 'config.yaml' qui contiennent 'all_joints_names' (DLC classique)
    ou 'bodyparts' (certaines versions).
    """
    if yaml is None:
        raise RuntimeError("PyYAML manquant: pip install pyyaml")

    ymls = []
    for pat in ("**/pose_cfg.yaml", "**/config.yaml", "pose_cfg.yaml", "config.yaml"):
        ymls.extend(glob.glob(os.path.join(model_dir, pat), recursive=True))
    if not ymls:
        raise FileNotFoundError(f"Aucun pose_cfg.yaml/config.yaml trouvé dans {model_dir}")

    # On prend le premier plausible
    with open(ymls[0], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Clés fréquentes
    if "all_joints_names" in cfg and isinstance(cfg["all_joints_names"], list):
        names = cfg["all_joints_names"]
    elif "bodyparts" in cfg and isinstance(cfg["bodyparts"], list):
        names = cfg["bodyparts"]
    elif "multianimalbodyparts" in cfg and isinstance(cfg["multianimalbodyparts"], list):
        names = cfg["multianimalbodyparts"]
    else:
        raise KeyError("Impossible de trouver la liste des marqueurs (all_joints_names/bodyparts).")

    # Nettoyage en str simples
    names = [str(n) for n in names]
    if not names:
        raise ValueError("Liste de marqueurs vide dans la config DLC.")
    return names

def indices_for(desired_names, all_names):
    """Mappe une liste de noms cibles vers leurs indices (ignore ceux introuvables)."""
    idx = []
    for n in desired_names:
        if n in all_names:
            idx.append(all_names.index(n))
    return idx

# =================== CONFIG ===================
prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep

INPUT_DIR   = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44"   # contient depth_XXXX.png (uint16 en mm)
DLC_MODEL   = fr"{pref_bis}Users{sep}User{sep}Documents{sep}Amedeo{sep}DLC_projects{sep}P12_excluded_normal_500{sep}exported-models{sep}DLC_test_mobilenet_v2_0.5_iteration-0_shuffle-1"
ZED_JSON = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}config_camera_files{sep}config_camera_P0.json"

OUT_MP4     = f"{sep}home{sep}mickael{sep}Documents{sep}Leo{sep}overlay_test_3.mp4"
OUT_CSV_M     = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_left_arm_xyz_m.csv"
OUT_CSV_PX     = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44{sep}data{sep}markers_left_arm_px_mm.csv"


FPS_OUT     = 25
PCUTOFF     = 0.70
# seuil visu vert; sinon jaune
# --- CONFIG Modèle ---

# Charge la liste complète (et dans le bon ordre) directement depuis le modèle
ALL_MARKERS = load_dlc_marker_names(DLC_MODEL)
print(f"[DLC] {len(ALL_MARKERS)} marqueurs:", ALL_MARKERS)

# --- CONFIG — section bras gauche ---
# Mets ici EXACTEMENT les noms tels qu’ils apparaissent dans ALL_MARKERS :
DESIRED_LEFT = ['xiph', 'ster', 'clavsc', 'clavac', 'delt', 'arm_l', 'epic_l', 'larm_l', 'styl_r', 'styl_u']  # adapte à tes étiquettes réelles
LEFT_IDX     = indices_for(DESIRED_LEFT, ALL_MARKERS)
LEFT_NAMES   = [ALL_MARKERS[i] for i in LEFT_IDX]
print("[DLC] LEFT_IDX:", LEFT_IDX)
print("[DLC] LEFT_NAMES:", LEFT_NAMES)

# Pré-traitement depth -> mêmes choix que ton script
CROP        = None                            # [x0,y0,x1,y1] ou None
ASPECT      = 1.0                             # facteur d'échelle (float)

# Bornes de frames (inclusives). Si None -> pas de borne.
FRAME_START = 600
FRAME_END   = 800
USE_NUMERIC_FROM_NAME = True  # True: borne sur numéro dans le nom; False: sur l'index trié
# =============================================

_num_re = re.compile(r"(\d+)")

# ======= Surface normales (reprend ton approche) =======  :contentReference[oaicite:1]{index=1}
def compute_surface_normals(depth_map, empty_mat=None):
    border_type = cv2.BORDER_DEFAULT
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, borderType=border_type)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, borderType=border_type)
    normal = empty_mat if empty_mat is not None else np.empty(
        (depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32
    )
    normal[..., 2] = -dx
    normal[..., 1] = -dy
    normal[..., 0] = 1.0
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    norm[norm == 0] = 1
    normal /= norm
    normal = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return normal

def make_normals_from_depth(depth_mm_uint16):
    """Clamp (0.2–1.5 m), puis normals 3 canaux 8-bit pour DLC (même esprit que ton script)."""
    d = depth_mm_uint16.astype(np.float32)
    d = np.where((d > 1500) | (d <= 200), 0, d)  # à ajuster si range différent
    return compute_surface_normals(d)

# ======= I/O utilitaires =======
def extract_num_from_name(path: str):
    m = _num_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def filter_paths_by_bounds(paths):
    if FRAME_START is None and FRAME_END is None:
        return list(enumerate(paths)), None
    if USE_NUMERIC_FROM_NAME:
        keep = []
        for i, p in enumerate(paths):
            n = extract_num_from_name(p)
            if n is None:
                continue
            if (FRAME_START is None or n >= FRAME_START) and (FRAME_END is None or n <= FRAME_END):
                keep.append((i, p))
        return keep, "name_num"
    else:
        start = FRAME_START if FRAME_START is not None else 0
        end   = FRAME_END   if FRAME_END   is not None else len(paths)-1
        start = max(0, int(start)); end = min(len(paths)-1, int(end))
        return list(enumerate(paths))[start:end+1], "index"

# ======= ZED intrinsics & conversion =======
def load_intrinsics(json_path):
    """Renvoie (K=(fx,fy,cx,cy), depth_scale (m/unit), size_depth ou None)."""
    if not json_path or not os.path.isfile(json_path):
        return None, None, None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fx = fy = cx = cy = None
    depth_scale = data.get("depth_scale", None)
    size_depth = tuple(data.get("size_depth", [])) if isinstance(data.get("size_depth", []), list) else None

    # schéma "camera.depth_*" (au cas où)
    try:
        fx, fy = data["camera"]["depth_fx_fy"]
        cx, cy = data["camera"]["depth_ppx_ppy"]
    except Exception:
        pass

    # schéma "left_color_*" (souvent présent avec ZED alignée couleur)
    if fx is None or fy is None or cx is None or cy is None:
        try:
            fx, fy = data["left_color_fx_fy"]
            cx, cy = data["left_color_ppx_ppy"]
        except Exception:
            pass

    # schéma plat en dernier recours
    if fx is None: fx = data.get("fx") or data.get("depth_fx") or data.get("camera", {}).get("fx")
    if fy is None: fy = data.get("fy") or data.get("depth_fy") or data.get("camera", {}).get("fy")
    if cx is None: cx = data.get("cx") or data.get("depth_cx") or data.get("camera", {}).get("cx")
    if cy is None: cy = data.get("cy") or data.get("depth_cy") or data.get("camera", {}).get("cy")

    if None in (fx, fy, cx, cy):
        return None, depth_scale, size_depth
    return (float(fx), float(fy), float(cx), float(cy)), depth_scale, size_depth

def adjust_intrinsics_for_crop_aspect(K, crop, aspect):
    """Décale cx,cy si CROP; multiplie fx,fy,cx,cy par ASPECT si redimensionnement."""
    if K is None:
        return None
    fx, fy, cx, cy = K
    if crop is not None:
        x0, y0, x1, y1 = crop
        cx -= x0; cy -= y0
    if aspect and aspect != 1.0:
        fx *= aspect; fy *= aspect; cx *= aspect; cy *= aspect
    return (fx, fy, cx, cy)

def uv_depth_to_xyz(u, v, depth_value, intrinsics, depth_scale):
    """(u,v,depth_raw) -> (X,Y,Z) en m. depth_scale = m par unité (ex: 0.001 pour mm)."""
    if depth_scale is None or not np.isfinite(depth_value):
        Z = np.nan
    else:
        Z = float(depth_value) * float(depth_scale)
    if intrinsics is None or not np.isfinite(Z):
        return np.nan, np.nan, Z
    fx, fy, cx, cy = intrinsics
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

# ======= Overlay =======
def draw_markers_on_depth(depth_mm: np.ndarray, poses: np.ndarray):
    scale = 255.0 / 1500.0
    depth_u8 = cv2.convertScaleAbs(depth_mm, alpha=scale)
    show = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    H, W = depth_mm.shape[:2]
    for i, name in enumerate(LEFT_NAMES):
        idx = LEFT_IDX[i] if i < len(LEFT_IDX) else None
        if idx is None or idx >= poses.shape[0]:
            continue
        u, v, p = float(poses[idx, 0]), float(poses[idx, 1]), float(poses[idx, 2])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        col = (40, 255, 40) if p >= PCUTOFF else (0, 200, 255)
        cv2.circle(show, (ui, vi), 6, col, -1, lineType=cv2.LINE_AA)
        cv2.putText(show, name, (ui + 7, vi + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return show

# ======= Main =======
def main():
    os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV_PX), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV_M), exist_ok=True)

    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "depth_*.png")))
    if not paths:
        raise FileNotFoundError(f"Aucune image depth_*.png dans: {INPUT_DIR}")

    kept, mode = filter_paths_by_bounds(paths)
    if not kept:
        raise RuntimeError("Aucune frame dans la plage demandée (FRAME_START/END).")
    paths_kept = [p for _, p in kept]

    # Première frame -> taille & init DLC
    depth0 = cv2.imread(paths_kept[0], cv2.IMREAD_ANYDEPTH)
    if depth0 is None:
        raise RuntimeError("Impossible de lire la première image sélectionnée.")
    if CROP is not None:
        x0, y0, x1, y1 = CROP; depth0 = depth0[y0:y1, x0:x1]
    if ASPECT and ASPECT != 1.0:
        H0, W0 = depth0.shape[:2]
        depth0 = cv2.resize(depth0, (int(W0*ASPECT), int(H0*ASPECT)))

    normals0 = make_normals_from_depth(depth0)  # *** entrée DLC = surface normales ***
    dlc = DLCLive(DLC_MODEL, processor=Processor())
    dlc.init_inference(normals0)

    # ZED intrinsics
    K_raw, depth_scale, _ = load_intrinsics(ZED_JSON)
    K = adjust_intrinsics_for_crop_aspect(K_raw, CROP, ASPECT)
    if depth_scale is None:
        depth_scale = 0.001  # fallback: profondeur en mm
        print("ℹ️ depth_scale absent → fallback 0.001 (mm → m).")
    if K is None:
        print("⚠️ Intrinsics incomplets → X_m,Y_m seront NaN (Z_m ok).")
    else:
        print(f"Intrinsics (après CROP/ASPECT): fx={K[0]:.2f}, fy={K[1]:.2f}, cx={K[2]:.2f}, cy={K[3]:.2f}")

    # Video writer
    H, W = depth0.shape[:2]
    vw = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (W, H), True)
    if not vw.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo: {OUT_MP4}")

    # CSV headers (3 lignes)
    header_px_l1 = ["meta"] + sum(([n, n, n] for n in LEFT_NAMES), [])
    header_px_l2 = ["frame"] + sum((["x", "y", "z"] for _ in LEFT_NAMES), [])
    header_px_l3 = ["idx"]   + sum((["px", "px", "mm"] for _ in LEFT_NAMES), [])
    rows_px = []

    header_m_l1 = ["meta"] + sum(([n, n, n] for n in LEFT_NAMES), [])
    header_m_l2 = ["frame"] + sum((["X", "Y", "Z"] for _ in LEFT_NAMES), [])
    header_m_l3 = ["idx"]   + sum((["m", "m", "m"] for _ in LEFT_NAMES), [])
    rows_m = []

    t0 = time.time()
    for k, p in kept:
        depth = cv2.imread(p, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            continue
        if CROP is not None:
            x0, y0, x1, y1 = CROP; depth = depth[y0:y1, x0:x1]
        if ASPECT and ASPECT != 1.0:
            Hc, Wc = depth.shape[:2]
            depth = cv2.resize(depth, (int(Wc*ASPECT), int(Hc*ASPECT)))

        normals = make_normals_from_depth(depth)  # *** comme ton code ***
        poses = dlc.get_pose(normals)             # (N,3): x,y,likelihood

        frame_id = extract_num_from_name(p) if (mode == "name_num") else k

        # CSV rows
        row_px = [frame_id]
        row_m  = [frame_id]
        H_img, W_img = depth.shape[:2]

        for i in range(len(LEFT_NAMES)):
            idx = LEFT_IDX[i] if i < len(LEFT_IDX) else None
            if (idx is None) or (idx >= poses.shape[0]):
                row_px += [np.nan, np.nan, np.nan]
                row_m  += [np.nan, np.nan, np.nan]
                continue

            u, v, pconf = float(poses[idx, 0]), float(poses[idx, 1]), float(poses[idx, 2])
            # px
            x_px = u if np.isfinite(u) else np.nan
            y_px = v if np.isfinite(v) else np.nan
            # z(mm) au pixel
            if np.isfinite(u) and np.isfinite(v):
                ui, vi = int(round(u)), int(round(v))
                z_mm = float(depth[vi, ui]) if (0 <= ui < W_img and 0 <= vi < H_img) else np.nan
            else:
                z_mm = np.nan
            row_px += [x_px, y_px, z_mm]

            # mètres
            X_m, Y_m, Z_m = uv_depth_to_xyz(u, v, z_mm, K, depth_scale)
            row_m += [X_m, Y_m, Z_m]

        rows_px.append(row_px)
        rows_m.append(row_m)

        # overlay
        frame_vis = draw_markers_on_depth(depth, poses)
        vw.write(frame_vis)

    vw.release()

    # write CSVs (3 lignes d’en-tête)
    for out_path, h1, h2, h3, rows in [
        (OUT_CSV_PX, header_px_l1, header_px_l2, header_px_l3, rows_px),
        (OUT_CSV_M,  header_m_l1,  header_m_l2,  header_m_l3,  rows_m),
    ]:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(h1); w.writerow(h2); w.writerow(h3)
            w.writerows(rows)

    dt = time.time() - t0
    print(f"✅ Vidéo : {OUT_MP4}")
    print(f"✅ CSV px/mm : {OUT_CSV_PX}")
    print(f"✅ CSV m     : {OUT_CSV_M}")
    print(f"   Frames: {len(paths_kept)} | Sélection: {('num fichier' if mode=='name_num' else 'index')} | {dt:.2f}s")

if __name__ == "__main__":
    main()
