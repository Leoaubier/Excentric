import os
import cv2
from dlclive import DLCLive, Processor
import time
import glob
import pandas as pd
import pickle
import json
import numpy as np
import csv
from pathlib import Path

# ========================= Utils =========================

def compute_surface_normals(depth_map, empty_mat = None):
    border_type = cv2.BORDER_DEFAULT
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, borderType=border_type)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, borderType=border_type)
    normal = empty_mat if empty_mat is not None else np.empty(
        (depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
    normal[..., 2] = -dx
    normal[..., 1] = -dy
    normal[..., 0] = 1.0
    norm = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    norm[norm == 0] = 1  # Prevent division by zero
    normal /= norm
    normal = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return normal


def process_image(path, idx, as_depth=False, first_max=-1, crop = None, aspect_ratio = None, empty_mat=None):
    if as_depth:
        try:
            depth = cv2.imread(path + f"{sep}depth_{idx}.png", cv2.IMREAD_ANYDEPTH)
            color = None
        except:
            return None, None, first_max, 0
        if depth is None:
            return None, None, first_max, 0
        tic = time.time()
        depth = depth[crop[1]:crop[3], crop[0]:crop[2]] if crop is not None else depth
        depth = np.where(
            (depth > 1.5*1000) | (depth <= 0.2*1000),
            0,
            depth,
        )
        h, w = depth.shape[:2]
        if aspect_ratio is not None:
            depth = cv2.resize(depth, (int(w * aspect_ratio), int(h * aspect_ratio)))
        depth = compute_surface_normals(depth, empty_mat)
        time_ti_proce = time.time() - tic
        return color, depth, first_max, time_ti_proce
    else:
        depth = cv2.imread(path + f"\depth_{idx}.png")
        color = depth.copy()
        if depth is None or color is None:
            return None, None
        return color, depth


def get_image_idx(path, from_depth=False):
    all_depth_files = glob.glob(path + ("/depth*.png" if from_depth else "/color*.png"))
    idx = []
    for f in all_depth_files:
        idx.append(int(f.split("\\")[-1].split("_")[-1].removesuffix(".png")))
    idx.sort()
    return idx


def load_zed_intrinsics(json_path, target_w, target_h):
    """
    Lit les intrinsèques ZED (caméra gauche) et les remet à l'échelle vers (target_w, target_h).
    Retourne: fx, fy, cx, cy, o3d_depth_scale  (où o3d_depth_scale = 1/json_depth_scale)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fx, fy = cfg["left_color_fx_fy"]
    cx, cy = cfg["left_color_ppx_ppy"]

    ref_w, ref_h = cfg.get("size_depth", cfg.get("size_color", [target_w, target_h]))
    sx = float(target_w) / float(ref_w)
    sy = float(target_h) / float(ref_h)
    fx, fy = fx * sx, fy * sy
    cx, cy = cx * sx, cy * sy

    depth_scale_json = float(cfg.get("depth_scale", 0.001))  # typiquement 0.001 (mm->m)
    o3d_depth_scale  = 1.0 / depth_scale_json               # Open3D attend "mm" => 1000

    return fx, fy, cx, cy, o3d_depth_scale

def get_resized_shape_for_o3d():
    d = first_depth_raw
    if crop is not None:
        d = d[crop[1]:crop[3], crop[0]:crop[2]]
    if ratio:
         d = cv2.resize(d, (int(d.shape[1]*ratio), int(d.shape[0]*ratio)), interpolation=cv2.INTER_NEAREST)
    return d.shape[1], d.shape[0]  # (W,H)

def normalize_xyz_from_list(xyz_list):
    """
    xyz_list: liste de longueur T
      - chaque élément peut être shape (N,3) ou (3,N)
    Retourne XYZ shape (3, N, T) en float.
    """
    if xyz_list is None or len(xyz_list) == 0:
        return None
    first = np.asarray(xyz_list[0])
    if first.ndim != 2 or (first.shape[1] not in (3,) and first.shape[0] not in (3,)):
        raise ValueError("xyz_list: chaque item doit être (N,3) ou (3,N).")

    # empile
    arrs = [np.asarray(a, dtype=float) for a in xyz_list]
    if arrs[0].shape[1] == 3:  # (N,3) -> transpose en (3,N)
        arrs = [a.T for a in arrs]  # (3,N)
    # maintenant chaque frame: (3,N)
    XYZ = np.stack(arrs, axis=-1)  # (3,N,T)
    return XYZ

def normalize_px_from_list(markers_px_list):
    """
    markers_px_list: liste de longueur T
      - chaque item peut être (N,3) = [x_px, y_px, likelihood] par marker
      - ou (3,N) = [x;y;likelihood]
    Retourne Xpx, Ypx shape (N,T) en float.
    """
    if markers_px_list is None or len(markers_px_list) == 0:
        return None, None
    first = np.asarray(markers_px_list[0])
    if first.ndim != 2:
        raise ValueError("markers_px_list: chaque item doit être 2D (N,3) ou (3,N).")

    arrs = [np.asarray(a, dtype=float) for a in markers_px_list]
    if arrs[0].shape[1] == 3:  # (N,3)
        Xpx = np.stack([a[:, 0] for a in arrs], axis=-1)  # (N,T)
        Ypx = np.stack([a[:, 1] for a in arrs], axis=-1)  # (N,T)
    elif arrs[0].shape[0] == 3:  # (3,N)
        Xpx = np.stack([a[0, :] for a in arrs], axis=-1)  # (N,T)
        Ypx = np.stack([a[1, :] for a in arrs], axis=-1)  # (N,T)
    else:
        raise ValueError("markers_px_list: attendu (N,3) ou (3,N).")
    return Xpx, Ypx

def build_df_xyz(markers_names, frames_np, XYZ):
    """
    Construit un DataFrame à colonnes MultiIndex (marker, axis, unit) pour XYZ (m).
    XYZ: (3,N,T) (X,Y,Z)
    """
    T = len(frames_np)
    cols = [("meta", "frame", "idx")]
    data_cols = [frames_np.astype(int)]
    if XYZ is None:
        # fabrique colonnes vides si pas d'XYZ
        for name in markers_names:
            cols.extend([(name, "x", "m"), (name, "y", "m"), (name, "z", "m")])
            data_cols.extend([np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)])
    else:
        _, N, TT = XYZ.shape
        assert TT == T, "Incohérence frames entre XYZ et frames_np"
        for m, name in enumerate(markers_names):
            cols.extend([(name, "x", "m"), (name, "y", "m"), (name, "z", "m")])
            data_cols.extend([XYZ[0, m, :], XYZ[1, m, :], XYZ[2, m, :]])
    mi = pd.MultiIndex.from_tuples(cols, names=["marker", "axis", "unit"])
    df_xyz = pd.DataFrame(np.column_stack(data_cols), columns=mi)
    return df_xyz

def build_df_px(markers_names, frames_np, Xpx, Ypx):
    """
    Construit un DataFrame à colonnes MultiIndex (marker, axis, unit) pour pixels (px).
    Xpx, Ypx: (N,T)
    """
    T = len(frames_np)
    cols = [("meta", "frame", "idx")]
    data_cols = [frames_np.astype(int)]
    if Xpx is None or Ypx is None:
        for name in markers_names:
            cols.extend([(name, "x", "px"), (name, "y", "px")])
            data_cols.extend([np.full(T, np.nan), np.full(T, np.nan)])
    else:
        N, TT = Xpx.shape
        assert TT == T, "Incohérence frames entre Xpx/Ypx et frames_np"
        for m, name in enumerate(markers_names):
            cols.extend([(name, "x", "px"), (name, "y", "px")])
            data_cols.extend([Xpx[m, :], Ypx[m, :]])
    mi = pd.MultiIndex.from_tuples(cols, names=["marker", "axis", "unit"])
    df_px = pd.DataFrame(np.column_stack(data_cols), columns=mi)
    return df_px

def write_multiheader_csv(path: Path, df_multi: pd.DataFrame):
    """
    Écrit un CSV avec 3 lignes d’en-tête (marker / axis / unit) puis les données (sans index).
    """
    cols = list(df_multi.columns)  # tuples (marker, axis, unit)
    header_l1 = [c[0] for c in cols]
    header_l2 = [c[1] for c in cols]
    header_l3 = [c[2] for c in cols]
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_l1)
        w.writerow(header_l2)
        w.writerow(header_l3)
        for row in df_multi.to_numpy():
            w.writerow(row)

# ========================= Main =========================

if __name__ == '__main__':
    prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
    pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
    sep = os.sep

    # --------- DLC init ---------
    dlc_proc = Processor()
    path_to_model =  fr"{pref_bis}Users{sep}User{sep}Documents{sep}Amedeo{sep}DLC_projects{sep}P12_excluded_normal_500{sep}exported-models{sep}DLC_test_mobilenet_v2_0.5_iteration-0_shuffle-1"
    dlc_live = DLCLive(path_to_model, processor=dlc_proc)

    # --------- Données + JSON camera ---------
    CAMERA_JSON = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}config_camera_files{sep}config_camera_P0.json"
    path_image = prefix + f"{sep}Projet_hand_bike_markerless{sep}Leo{sep}P0{sep}demo_20-10-2025_10_47_44"

    # Sorties
    save_dir = Path(path_image); save_dir.mkdir(parents=True, exist_ok=True)
    out_bio = save_dir / "markers_positions_multi_frames.bio"
    csv_xyz_path = Path(save_dir) / "markers_xyz_m.csv"
    csv_px_path = Path(save_dir) / "pixels_xy.csv"

    markers_px_list = []  # (3, N, T)   => x_px_raw, y_px_raw, likelihood
    depth_mm_list   = []  # (N, T)
    frame_list      = []  # (T,)

    # Visualisation Open3D périodique (0 = désactivé)
    DRAW_3D_EVERY_K = 50  # dessine le nuage et les marqueurs toutes les K frames ; mettre 0 pour désactiver

    # Crop & indices
    crop = [113, 33, 483, 337]
    idx = get_image_idx(path_image, from_depth=False)

    # Intrinsèques et buffers init
    first_depth_raw = cv2.imread(path_image+f"{sep}depth_{idx[0]}.png", cv2.IMREAD_ANYDEPTH)
    if first_depth_raw is None:
        raise RuntimeError("Impossible de lire la depth initiale.")
    raw_h, raw_w = first_depth_raw.shape[:2]

    # Ratio de resize appliqué aux images "traitées" (pour DLC et pour Open3D ensuite)
    ratio = 1.0

    # Charger intrinsics du JSON à la résolution après crop/ratio pour Open3D
    # (on lira et recadrera la depth/color identiquement avant Open3D)
    if not Path(CAMERA_JSON).is_file():
        raise FileNotFoundError(f"JSON camera introuvable: {CAMERA_JSON}")

    # Buffer "normales" pour DLC init
    depth_init = first_depth_raw.copy()
    if crop is not None:
        depth_init = depth_init[crop[1]:crop[3], crop[0]:crop[2]]
    if ratio:
        depth_init = cv2.resize(depth_init, (int(depth_init.shape[1]*ratio), int(depth_init.shape[0]*ratio)),
                                interpolation=cv2.INTER_NEAREST)
    normal = np.empty((depth_init.shape[0], depth_init.shape[1], 3), dtype=np.float32)
    _, depth_image, _, _ = process_image(path_image, idx[0], as_depth=True, first_max=-1,
                                         crop=crop, aspect_ratio=ratio, empty_mat=normal)
    dlc_live.init_inference(depth_image)

    # Listes pour XYZ (caméra) si on veut placer des sphères 3D
    xyz_list = []
    fx = fy = cx = cy = None
    o3d_depth_scale = None

    # Calcule (W,H) de la profondeur/couleur après crop/ratio (utile pour set_intrinsics Open3D)
    W_o3d, H_o3d = get_resized_shape_for_o3d()
    fx, fy, cx, cy, o3d_depth_scale = load_zed_intrinsics(CAMERA_JSON, W_o3d, H_o3d)

    # Mesures de perf (optionnel)
    all_process_time, all_dlc_time = [], []

    # ======================= Boucle =======================
    for i in range(200, min(len(idx), 200 + 200)):  # limite à 200 frames pour test
        # 1) Lecture depth brute (mm) pour ce frame
        depth_raw_mm = cv2.imread(path_image + f"{sep}depth_{idx[i]}.png", cv2.IMREAD_ANYDEPTH)
        if depth_raw_mm is None:
            continue

        # 2) Image "normales" (pour DLC)
        _, depth_image, first_max, time_to_process_tmp = process_image(
            path_image, idx[i], as_depth=True, first_max=-1, crop=crop, aspect_ratio=ratio, empty_mat=normal)
        all_process_time.append(time_to_process_tmp)
        if depth_image is None:
            continue

        # 3) Détection DLC
        tic = time.time()
        poses = dlc_live.get_pose(depth_image)
        all_dlc_time.append(time.time() - tic)

        # 4) Sauvegarde des positions + profondeur au pixel
        N = poses.shape[0]
        x_raw = (poses[:, 0] / ratio) + (crop[0] if crop is not None else 0)
        y_raw = (poses[:, 1] / ratio) + (crop[1] if crop is not None else 0)
        conf  = poses[:, 2] if poses.shape[1] > 2 else np.ones(N, dtype=float)

        h_raw, w_raw = depth_raw_mm.shape[:2]
        xi = np.clip(np.rint(x_raw).astype(int), 0, w_raw - 1)
        yi = np.clip(np.rint(y_raw).astype(int), 0, h_raw - 1)

        z_mm = depth_raw_mm[yi, xi].astype(np.int32)
        # Remplir 0 par médiane 3x3 locale
        if np.any(z_mm <= 0):
            bad_idx = np.where(z_mm <= 0)[0]
            for k in bad_idx:
                x0, x1 = max(0, xi[k] - 1), min(w_raw, xi[k] + 2)
                y0, y1 = max(0, yi[k] - 1), min(h_raw, yi[k] + 2)
                patch = depth_raw_mm[y0:y1, x0:x1]
                nz = patch[patch > 0]
                z_mm[k] = int(np.median(nz)) if nz.size > 0 else 0

        markers_px_list.append(np.vstack([x_raw, y_raw, conf]))
        depth_mm_list.append(z_mm.copy())
        frame_list.append(idx[i])

        # 5) XYZ caméra (m) pour les sphères de visualisation
        if o3d_depth_scale is not None:
            # json depth_scale = 0.001 (mm->m)  => Z_m = z_mm * 0.001
            Z_m = z_mm.astype(float) * (1.0 / o3d_depth_scale)  # puisque o3d_depth_scale = 1/json_scale
            X_m = (x_raw - cx) / fx * Z_m
            Y_m = (y_raw - cy) / fy * Z_m
            xyz_list.append(np.vstack([X_m, Y_m, Z_m]))

        # 6) Aperçu 2D (OpenCV)
        depth_to_show = depth_image.copy()
        for p in range(N):
            cv2.circle(depth_to_show, poses[p, :2].astype(int), 5, (0,255,0), -1)
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.imshow("depth", depth_to_show)
        cv2.waitKey(1)


    # ======================= Sauvegarde =======================
    markers_px = np.stack(markers_px_list, axis=-1)  # (3, N, T)
    depth_mm   = np.stack(depth_mm_list,   axis=-1)  # (N, T)

    n_markers = markers_px.shape[1]
    T = markers_px.shape[-1]
    markers_names = [f"M{k + 1}" for k in range(n_markers)]
    frames_np = np.array(frame_list, dtype=int)

    out = {
        "markers_in_pixels": markers_px,   # x_px_raw, y_px_raw, likelihood
        "depth_mm": depth_mm,              # mm au pixel pour chaque marker
        "markers_names": markers_names,
        "frame_idx": frames_np,
    }
    if xyz_list:
        out["markers_in_meters"] = np.stack(xyz_list, axis=-1)  # (3, N, T)

    with open(out_bio, "wb") as f:
        pickle.dump(out, f)
    print(f"✅ Enregistré {T} frames dans {out_bio}")

    # 2) Convertis les données en matrices normalisées
    #    - Si tu as déjà markers_in_meters (3,N,T), réutilise-le
    #    - Sinon, crée-le à partir de xyz_list :
    try:
        XYZ = out.get("markers_in_meters", None)  # si tu as dict 'out'
    except Exception:
        XYZ = None

    if XYZ is None:
        # Tente depuis xyz_list (liste par frame)
        try:
            XYZ = normalize_xyz_from_list(xyz_list)  # -> (3,N,T)
        except Exception:
            XYZ = None  # restera NaN dans le CSV XYZ

    # Pixels : depuis structure existante si tu as (3,N,T)
    try:
        markers_px = out.get("markers_in_pixels", None)  # (3,N,T) si dispo
        if markers_px is not None and markers_px.ndim == 3:
            Xpx, Ypx = markers_px[0, :, :], markers_px[1, :, :]
        else:
            raise KeyError
    except Exception:
        # Sinon, depuis markers_px_list (liste par frame)
        try:
            Xpx, Ypx = normalize_px_from_list(markers_px_list)  # (N,T) chacun
        except Exception:
            Xpx, Ypx = None, None  # restera NaN dans le CSV pixels

    # 3) Construis les DataFrames MultiIndex
    df_xyz = build_df_xyz(markers_names, frames_np, XYZ)  # MultiIndex (marker, axis, unit)
    df_px = build_df_px(markers_names, frames_np, Xpx, Ypx)

    # 4) Écris les CSV avec 3 lignes d’entête
    write_multiheader_csv(csv_xyz_path, df_xyz)
    write_multiheader_csv(csv_px_path, df_px)

    print(f"📄 CSV écrit : {csv_xyz_path}")
    print(f"📄 CSV écrit : {csv_px_path}")

