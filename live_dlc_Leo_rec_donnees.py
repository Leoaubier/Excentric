import os
import cv2
from dlclive import DLCLive, Processor
import time
import glob
import open3d as o3d
import pickle
from pathlib import Path
import json
import numpy as np

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
    out_csv = save_dir / "markers_positions_multi_frames.csv"

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
    def get_resized_shape_for_o3d():
        d = first_depth_raw
        if crop is not None:
            d = d[crop[1]:crop[3], crop[0]:crop[2]]
        if ratio:
            d = cv2.resize(d, (int(d.shape[1]*ratio), int(d.shape[0]*ratio)), interpolation=cv2.INTER_NEAREST)
        return d.shape[1], d.shape[0]  # (W,H)

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

        # 7) Dessin 3D Open3D (nuage RGB-D + sphères markers) toutes les K frames
        # if DRAW_3D_EVERY_K and (len(frame_list) % DRAW_3D_EVERY_K == 0):
        #     # Construire depth (mm) et color (BGR->RGB) avec le même crop/ratio
        #     depth_for_o3d = depth_raw_mm
        #     if crop is not None:
        #         depth_for_o3d = depth_for_o3d[crop[1]:crop[3], crop[0]:crop[2]]
        #     if ratio:
        #         depth_for_o3d = cv2.resize(depth_for_o3d, (int(depth_for_o3d.shape[1]*ratio),
        #                                                    int(depth_for_o3d.shape[0]*ratio)),
        #                                    interpolation=cv2.INTER_NEAREST)
        #     depth_o3d = o3d.geometry.Image(depth_for_o3d.astype(np.uint16))
        #
        #     color_bgr = cv2.imread(path_image + f"{sep}color_{idx[i]}.png", cv2.IMREAD_COLOR)
        #     if color_bgr is None:
        #         # si pas de color, fabrique une image grise
        #         color_bgr = np.zeros((H_o3d, W_o3d, 3), np.uint8)
        #     if crop is not None:
        #         color_bgr = color_bgr[crop[1]:crop[3], crop[0]:crop[2], :]
        #     if ratio:
        #         color_bgr = cv2.resize(color_bgr, (int(color_bgr.shape[1]*ratio),
        #                                            int(color_bgr.shape[0]*ratio)),
        #                                interpolation=cv2.INTER_LINEAR)
        #     color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        #     color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
        #
        #     intrinsics = o3d.camera.PinholeCameraIntrinsic()
        #     intrinsics.set_intrinsics(width=W_o3d, height=H_o3d, fx=fx, fy=fy, cx=cx, cy=cy)
        #
        #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #         color_o3d, depth_o3d,
        #         depth_scale=o3d_depth_scale,   # ex: 1000 si depth en mm
        #         depth_trunc=10.0,
        #         convert_rgb_to_intensity=False
        #     )
        #     pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        #
        #     # Repère vision -> Open3D (y up, z vers l'observateur)
        #     T_flip = np.array([[1,0,0,0],
        #                        [0,-1,0,0],
        #                        [0,0,-1,0],
        #                        [0,0,0,1]], float)
        #     pcd1.transform(T_flip)
        #
        #     # Sphères aux positions 3D du frame courant (si calculées)
        #     sphere_list = []
        #     if xyz_list:
        #         P = xyz_list[-1]  # (3,N)
        #         for m in range(P.shape[1]):
        #             if not np.all(np.isfinite(P[:, m])) or P[2, m] <= 0:
        #                 continue
        #             s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #             s.translate(P[:, m])
        #             s.compute_vertex_normals()
        #             s.paint_uniform_color([0.8, 0.2, 0.2])
        #             s.transform(T_flip)
        #             sphere_list.append(s)
        #
        #     # Axes
        #     def axis(vec, col):
        #         ls = o3d.geometry.LineSet()
        #         ls.points = o3d.utility.Vector3dVector(np.array([[0,0,0], vec], float))
        #         ls.lines  = o3d.utility.Vector2iVector(np.array([[0,1]], int))
        #         ls.colors = o3d.utility.Vector3dVector(np.array([col], float))
        #         return ls
        #     axes = [axis([1,0,0],[1,0,0]), axis([0,1,0],[0,1,0]), axis([0,0,1],[0,0,1])]
        #
        #     # IMPORTANT : fermer les fenêtres OpenCV pour éviter conflit focus
        #     try:
        #         cv2.destroyAllWindows()
        #     except:
        #         pass
        #     o3d.visualization.draw_geometries([pcd1, *axes, *sphere_list])

    # ======================= Sauvegarde =======================
    markers_px = np.stack(markers_px_list, axis=-1)  # (3, N, T)
    depth_mm   = np.stack(depth_mm_list,   axis=-1)  # (N, T)

    n_markers = markers_px.shape[1]
    markers_names = [f"M{k + 1}" for k in range(n_markers)]

    out = {
        "markers_in_pixels": markers_px,   # x_px_raw, y_px_raw, likelihood
        "depth_mm": depth_mm,              # mm au pixel pour chaque marker
        "markers_names": markers_names,
        "frame_idx": np.array(frame_list, dtype=int),
    }
    if xyz_list:
        out["markers_in_meters"] = np.stack(xyz_list, axis=-1)  # (3, N, T)

    with open(out_bio, "wb") as f:
        pickle.dump(out, f)
    print(f"✅ Enregistré {markers_px.shape[-1]} frames dans {out_bio}")

    # Option CSV
    try:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "marker_id", "marker_name", "x_px_raw", "y_px_raw", "likelihood", "depth_mm",
                        "X_m","Y_m","Z_m"])
            for t, fr in enumerate(frame_list):
                for m in range(n_markers):
                    x, y, lk = markers_px[:, m, t]
                    z = int(depth_mm[m, t])
                    if "markers_in_meters" in out:
                        X, Y, Z = out["markers_in_meters"][:, m, t]
                    else:
                        X = Y = Z = ""
                    w.writerow([fr, m, markers_names[m], float(x), float(y), float(lk), z, X, Y, Z])
        print(f"📄 CSV écrit : {out_csv}")
    except Exception as e:
        print(f"CSV non écrit ({e}).")
